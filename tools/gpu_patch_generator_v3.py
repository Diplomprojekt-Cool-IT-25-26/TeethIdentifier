"""GPU-Accelerated Patch Generation using Custom Rasterizer (V3).

Replaces PyTorch3D mesh duplication with a single-mesh-upload approach.
Projects faces into camera space using pure tensor math, rasterizes via
barycentric sampling + z-buffer scatter_reduce.
"""

import numpy as np
import torch
import trimesh
from scipy.spatial import KDTree
from tqdm import tqdm


class GPUPatchGeneratorV3:
    """GPU-accelerated patch generation without mesh duplication."""

    def __init__(self, mesh, annotations=None, patch_size=100, patch_radius=5.0, device='cuda'):
        if not torch.cuda.is_available() and device == 'cuda':
            device = 'cpu'

        self.device = torch.device(device)
        self.patch_size = patch_size
        self.patch_radius = patch_radius
        self.camera_distance = patch_radius * 1.5
        self.scale = patch_size / (2 * patch_radius)

        # Upload mesh data ONCE
        self.vertices = torch.tensor(mesh.vertices, dtype=torch.float32, device=self.device)
        self.faces = torch.tensor(mesh.faces, dtype=torch.int64, device=self.device)
        self.normals = torch.tensor(mesh.vertex_normals, dtype=torch.float32, device=self.device)

        # Per-vertex curvature (B channel): measures how much the surface bends at each point
        curv_np = self._compute_vertex_curvature(mesh)
        self.vertex_curvature = torch.tensor(curv_np, dtype=torch.float32, device=self.device)

        # Pre-compute face geometry on GPU
        self.face_verts = self.vertices[self.faces]  # (F, 3, 3)
        self.face_centroids = self.face_verts.mean(dim=1)  # (F, 3)

        # Spatial index for face centroids (CPU)
        centroids_np = self.face_centroids.cpu().numpy()
        self.face_kdtree = KDTree(centroids_np)

        # Pre-compute k-NN candidate faces and upload to GPU immediately.
        # Keeping them on GPU avoids per-batch numpy→GPU transfer overhead.
        _faces_np, _valid_np = self._precompute_candidate_faces(mesh.vertices)
        self._candidate_faces_gpu = torch.as_tensor(_faces_np, device=self.device)   # int32, (N, k)
        self._candidate_valid_gpu = torch.as_tensor(_valid_np, device=self.device)   # bool,  (N, k)
        del _faces_np, _valid_np  # free CPU RAM (~930 MB)

        # Pre-compute pixel offset grid for rasterization
        # max_bbox=10: handles triangles up to ~1mm edge at 10px/unit scale.
        # Larger than 6 avoids black gaps where triangle > 6px wide.
        max_bbox = 10
        oy, ox = torch.meshgrid(
            torch.arange(max_bbox, device=self.device, dtype=torch.float32),
            torch.arange(max_bbox, device=self.device, dtype=torch.float32),
            indexing='ij'
        )
        self._offsets = torch.stack([ox.reshape(-1), oy.reshape(-1)], dim=-1)  # (S, 2)
        self._max_bbox = max_bbox


    def _compute_vertex_curvature(self, mesh):
        """Compute per-vertex curvature as normal variation across the 1-ring neighbourhood.

        For each vertex, measures how much adjacent face normals deviate from it.
        High value = strongly curved (tooth cusp). Low value = flat (gingiva surface).
        Normalised to [0, 1] using the 2nd/98th percentile to ignore outliers.
        """
        normals = mesh.vertex_normals   # (N, 3)
        faces   = mesh.faces            # (F, 3)
        N = len(normals)

        # Build all directed edges from face triangles (6 edges per face)
        i0, i1, i2 = faces[:, 0], faces[:, 1], faces[:, 2]
        src = np.concatenate([i0, i1, i2, i1, i2, i0])
        dst = np.concatenate([i1, i2, i0, i0, i1, i2])

        # Dot product between each vertex normal and its neighbour's normal
        dot = np.einsum('ij,ij->i', normals[src], normals[dst])

        # Average dot per vertex
        dot_sum = np.zeros(N, dtype=np.float64)
        count   = np.zeros(N, dtype=np.float64)
        np.add.at(dot_sum, src, dot)
        np.add.at(count,   src, 1.0)
        avg_dot = dot_sum / np.maximum(count, 1)

        # Curvature proxy: 0 = perfectly flat, 1 = maximally curved
        curvature = (1.0 - avg_dot).astype(np.float32)

        lo, hi = np.percentile(curvature, [2, 98])
        normed = np.clip((curvature - lo) / (hi - lo + 1e-8), 0.0, 1.0)
        return normed.astype(np.float32)

    def _precompute_candidate_faces(self, vertices_np):
        """Pre-compute k-NN candidate faces for every vertex.

        Uses k-nearest instead of radius search so memory is bounded regardless
        of mesh density. Stores as uniform numpy arrays (no Python list overhead).
        """
        n_faces = len(self.face_centroids)
        k = min(2000, n_faces)
        distances, face_idx_arr = self.face_kdtree.query(vertices_np, k=k, workers=-1)

        # Boolean mask: include only faces whose centroid is within view radius
        valid = (distances <= self.patch_radius * 1.1).astype(np.bool_)  # (N, k)
        face_idx_arr = face_idx_arr.astype(np.int32)  # save memory vs int64

        counts = valid.sum(axis=1)
        mem_mb = (face_idx_arr.nbytes + valid.nbytes) / 1024 / 1024
        print(f"  k-NN candidates: k={k}, within-radius max={counts.max()}, mean={counts.mean():.0f}, RAM={mem_mb:.0f}MB")
        return face_idx_arr, valid

    def _compute_tangent_basis_batch(self, normals):
        batch_size = normals.shape[0]
        temp = torch.where(
            normals[:, 0].abs().unsqueeze(1) < 0.9,
            torch.tensor([1, 0, 0], device=self.device, dtype=torch.float32).expand(batch_size, 3),
            torch.tensor([0, 1, 0], device=self.device, dtype=torch.float32).expand(batch_size, 3)
        )
        u = torch.cross(normals, temp, dim=1)
        u = torch.nn.functional.normalize(u, dim=1)
        v = torch.cross(normals, u, dim=1)
        v = torch.nn.functional.normalize(v, dim=1)
        return u, v

    def _gather_candidate_faces(self, vertex_indices):
        """Slice pre-computed k-NN arrays from GPU tensors — zero CPU overhead."""
        # vertex_indices is already a GPU tensor; direct indexing stays on GPU.
        face_idx = self._candidate_faces_gpu[vertex_indices].long()  # (C, k) int64
        valid_mask = self._candidate_valid_gpu[vertex_indices]        # (C, k) bool
        return face_idx, valid_mask

    def _project_faces(self, face_idx, cam_origins, u, v, n):
        """Project candidate faces into each camera's screen space."""
        fv = self.face_verts[face_idx]  # (C, F_max, 3, 3)
        rel = fv - cam_origins[:, None, None, :]

        proj_u = (rel * u[:, None, None, :]).sum(dim=-1)  # (C, F_max, 3)
        proj_v = (rel * v[:, None, None, :]).sum(dim=-1)
        proj_z = (rel * n[:, None, None, :]).sum(dim=-1)

        px = (proj_u + self.patch_radius) * self.scale
        py = (proj_v + self.patch_radius) * self.scale

        screen_xy = torch.stack([px, py], dim=-1)  # (C, F_max, 3, 2)
        return screen_xy, proj_z

    def _rasterize(self, screen_xy, depth, face_idx, valid_mask, cam_n):
        """Rasterize projected triangles into images via z-buffer.

        Uses early filtering: computes barycentric coords, filters to valid
        samples (~5% of total), then does z-buffer and color only on those.
        """
        C, F_max = valid_mask.shape
        H = W = self.patch_size
        S = self._offsets.shape[0]

        v0 = screen_xy[:, :, 0, :]  # (C, F_max, 2)
        v1 = screen_xy[:, :, 1, :]
        v2 = screen_xy[:, :, 2, :]

        d0 = depth[:, :, 0]
        d1 = depth[:, :, 1]
        d2 = depth[:, :, 2]

        # Back-face culling — softened threshold to keep edge/boundary faces.
        # cross_z > 0 is strict front-face; > -0.3 * area keeps grazing faces
        # that are still partially visible (avoids black holes at mesh boundaries).
        edge1 = v1 - v0
        edge2 = v2 - v0
        cross_z = edge1[..., 0] * edge2[..., 1] - edge1[..., 1] * edge2[..., 0]
        valid_mask = valid_mask & (cross_z > -0.3)

        # Bounding boxes
        all_x = torch.stack([v0[..., 0], v1[..., 0], v2[..., 0]], dim=-1)
        all_y = torch.stack([v0[..., 1], v1[..., 1], v2[..., 1]], dim=-1)
        min_x = all_x.min(dim=-1).values.floor().clamp(0, W - 1)
        min_y = all_y.min(dim=-1).values.floor().clamp(0, H - 1)

        # Filter faces outside image
        max_x = all_x.max(dim=-1).values
        max_y = all_y.max(dim=-1).values
        valid_mask = valid_mask & (max_x >= 0) & (max_y >= 0) & (min_x < W) & (min_y < H)

        # Barycentric constants
        v0x, v0y = v0[..., 0].unsqueeze(-1), v0[..., 1].unsqueeze(-1)  # (C, F_max, 1)
        v1x, v1y = v1[..., 0].unsqueeze(-1), v1[..., 1].unsqueeze(-1)
        v2x, v2y = v2[..., 0].unsqueeze(-1), v2[..., 1].unsqueeze(-1)
        det = (v1y - v2y) * (v0x - v2x) + (v2x - v1x) * (v0y - v2y)
        det = det.clamp(min=1e-8)

        # Sample points at pixel centers within each face's bbox
        offsets = self._offsets  # (S, 2)
        sx = min_x.unsqueeze(-1) + offsets[:, 0] + 0.5  # (C, F_max, S)
        sy = min_y.unsqueeze(-1) + offsets[:, 1] + 0.5
        px_int = (min_x.unsqueeze(-1) + offsets[:, 0]).long()
        py_int = (min_y.unsqueeze(-1) + offsets[:, 1]).long()

        in_bounds = (px_int >= 0) & (px_int < W) & (py_int >= 0) & (py_int < H)

        # Barycentric coordinates
        w0 = ((v1y - v2y) * (sx - v2x) + (v2x - v1x) * (sy - v2y)) / det
        w1 = ((v2y - v0y) * (sx - v2x) + (v0x - v2x) * (sy - v2y)) / det
        w2 = 1.0 - w0 - w1

        inside = (w0 >= 0) & (w1 >= 0) & (w2 >= 0)
        valid_sample = inside & in_bounds & valid_mask.unsqueeze(-1)

        # --- Filter to valid samples only ---
        valid_flat = valid_sample.reshape(-1).nonzero(as_tuple=True)[0]  # indices into flat tensor
        n_valid = valid_flat.shape[0]

        if n_valid == 0:
            return torch.zeros(C, H, W, 3, device=self.device)

        w0_v = w0.reshape(-1)[valid_flat]
        w1_v = w1.reshape(-1)[valid_flat]
        w2_v = w2.reshape(-1)[valid_flat]

        # Free the big (C, F_max, S) tensors
        del w0, w1, w2, inside, in_bounds, sx, sy, valid_sample

        # Pixel indices
        cam_idx_expanded = torch.arange(C, device=self.device).reshape(C, 1, 1).expand(C, F_max, S)
        flat_pixel = (cam_idx_expanded.reshape(-1)[valid_flat] * H * W +
                      py_int.reshape(-1)[valid_flat] * W +
                      px_int.reshape(-1)[valid_flat])
        del cam_idx_expanded, px_int, py_int

        # Depth for valid samples
        d0_exp = d0.unsqueeze(-1).expand(C, F_max, S).reshape(-1)[valid_flat]
        d1_exp = d1.unsqueeze(-1).expand(C, F_max, S).reshape(-1)[valid_flat]
        d2_exp = d2.unsqueeze(-1).expand(C, F_max, S).reshape(-1)[valid_flat]
        sample_depth = w0_v * d0_exp + w1_v * d1_exp + w2_v * d2_exp
        del d0_exp, d1_exp, d2_exp

        # Z-buffer
        z_buffer = torch.full((C * H * W,), float('inf'), device=self.device)
        z_buffer.scatter_reduce_(0, flat_pixel, sample_depth, reduce='amin', include_self=True)

        # Depth match
        depth_match = (sample_depth - z_buffer[flat_pixel]).abs() < 1e-3
        del sample_depth

        if not depth_match.any():
            return torch.zeros(C, H, W, 3, device=self.device)

        # Filter to depth-matched samples — keep weights for normal interpolation
        matched_pixel = flat_pixel[depth_match]
        matched_depth = z_buffer[matched_pixel]
        matched_w0 = w0_v[depth_match]
        matched_w1 = w1_v[depth_match]
        matched_w2 = w2_v[depth_match]
        face_lookup = face_idx.unsqueeze(-1).expand(C, F_max, S).reshape(-1)[valid_flat][depth_match]
        del w0_v, w1_v, w2_v, flat_pixel, z_buffer, depth_match, valid_flat

        # --- Channel R: depth ---
        # proj_z > 0 = convex (tooth cusp), < 0 = concave (gingival valley)
        depth_ch = (matched_depth / self.patch_radius).clamp(-1.0, 1.0) * 0.5 + 0.5
        del matched_depth

        # --- Channel G: normal shading ---
        # Interpolate vertex normals at each matched pixel, then dot with camera direction.
        # Front-facing surface (normal toward camera) → bright; tangential/away → dark.
        vn0 = self.normals[self.faces[face_lookup, 0]]   # (N_matched, 3)
        vn1 = self.normals[self.faces[face_lookup, 1]]
        vn2 = self.normals[self.faces[face_lookup, 2]]
        interp_n = (matched_w0.unsqueeze(-1) * vn0 +
                    matched_w1.unsqueeze(-1) * vn1 +
                    matched_w2.unsqueeze(-1) * vn2)
        interp_n = interp_n / interp_n.norm(dim=-1, keepdim=True).clamp(min=1e-8)
        del vn0, vn1, vn2

        # --- Channel B: curvature ---
        # Interpolate per-vertex curvature. Bright = strongly curved (tooth cusp).
        # Dark = flat (gingiva). Sharp drop at the boundary gives the model a clear edge signal.
        vc0 = self.vertex_curvature[self.faces[face_lookup, 0]]
        vc1 = self.vertex_curvature[self.faces[face_lookup, 1]]
        vc2 = self.vertex_curvature[self.faces[face_lookup, 2]]
        curv_ch = (matched_w0 * vc0 + matched_w1 * vc1 + matched_w2 * vc2).clamp(0.0, 1.0)
        del matched_w0, matched_w1, matched_w2, face_lookup, vc0, vc1, vc2

        # cam_n is (C, 3); select the right camera normal per matched pixel
        cam_idx_per_pixel = torch.div(matched_pixel, H * W, rounding_mode='trunc')
        shading_ch = (interp_n * cam_n[cam_idx_per_pixel]).sum(dim=-1).clamp(0.0, 1.0)
        del interp_n, cam_idx_per_pixel

        # Assemble: R=depth, G=shading, B=curvature
        images = torch.zeros(C * H * W, 3, device=self.device)
        images[matched_pixel, 0] = depth_ch
        images[matched_pixel, 1] = shading_ch
        images[matched_pixel, 2] = curv_ch

        return images.reshape(C, H, W, 3)

    def _contrast_stretch(self, images):
        """Per-patch auto-levels: stretch actual depth range to [0, 1].

        Background pixels (exactly 0.0) are excluded from the min/max
        calculation and remain black after stretching.
        """
        C = images.shape[0]
        bg = (images.sum(dim=-1) == 0.0)          # (C, H, W) — background mask

        flat = images[..., 0].reshape(C, -1)       # one channel, (C, H*W)
        bg_flat = bg.reshape(C, -1)

        # Compute per-patch min/max over foreground pixels only
        flat_fg = flat.clone()
        flat_fg[bg_flat] = float('inf')
        lo = flat_fg.min(dim=1).values             # (C,)
        flat_fg[bg_flat] = float('-inf')
        hi = flat_fg.max(dim=1).values             # (C,)

        # Skip stretch for patches with no foreground or near-zero range
        valid = (hi - lo) > 0.01
        lo = torch.where(valid, lo, torch.zeros_like(lo))
        hi = torch.where(valid, hi, torch.ones_like(hi))

        lo = lo.reshape(C, 1, 1, 1)
        hi = hi.reshape(C, 1, 1, 1)
        stretched = ((images - lo) / (hi - lo)).clamp(0.0, 1.0)
        stretched[bg.unsqueeze(-1).expand_as(images)] = 0.0   # keep background black
        return stretched

    def _apply_gaussian_blur(self, images):
        import kornia
        images = images.permute(0, 3, 1, 2)
        blurred = kornia.filters.gaussian_blur2d(images, kernel_size=(3, 3), sigma=(0.5, 0.5))
        return blurred.permute(0, 2, 3, 1)

    def _safe_batch_size(self, requested):
        """Cap batch size to a known-safe value given max_bbox=10, F_max=2000.

        Peak VRAM per batch is dominated by the C×F_max×S rasterization tensors:
          C=256, F_max=2000, S=100 → 51M floats × ~6 tensors ≈ 1.2 GB peak.
        Cap at 256; candidate-face arrays (931 MB) are pre-loaded on GPU already.
        """
        return min(requested, 256)

    @torch.no_grad()
    def generate_patches_batch(self, vertex_indices, batch_size=256):
        batch_size = self._safe_batch_size(batch_size)
        all_patches = []
        n_vertices = len(vertex_indices)

        # Keep indices on GPU to avoid repeated CPU→GPU transfers per batch.
        if not isinstance(vertex_indices, torch.Tensor):
            vertex_indices_gpu = torch.as_tensor(
                vertex_indices.astype(np.int64) if hasattr(vertex_indices, 'astype')
                else vertex_indices,
                dtype=torch.long, device=self.device
            )
        else:
            vertex_indices_gpu = vertex_indices.to(self.device)

        for batch_start in tqdm(range(0, n_vertices, batch_size), desc="GPU rendering"):
            batch_end = min(batch_start + batch_size, n_vertices)
            batch_idx = vertex_indices_gpu[batch_start:batch_end]

            positions = self.vertices[batch_idx]
            norms = self.normals[batch_idx]
            u, v = self._compute_tangent_basis_batch(norms)

            face_idx, valid_mask = self._gather_candidate_faces(batch_idx)
            screen_xy, depth = self._project_faces(face_idx, positions, u, v, norms)
            images = self._rasterize(screen_xy, depth, face_idx, valid_mask, norms)
            images = self._contrast_stretch(images)

            patches = (images * 255).clamp(0, 255).to(torch.uint8).cpu().numpy()
            all_patches.append(patches)

            del positions, norms, u, v, face_idx, valid_mask, screen_xy, depth, images
            # Do NOT call empty_cache() per batch — the caching allocator reuses
            # freed blocks for the next batch without OS round-trips.

        if self.device.type == 'cuda':
            torch.cuda.empty_cache()

        return np.concatenate(all_patches, axis=0)
