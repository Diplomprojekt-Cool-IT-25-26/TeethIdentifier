"""GPU-Accelerated Patch Generation (Math Approach)."""

import numpy as np
import torch
import trimesh
from scipy.spatial import KDTree
from tqdm import tqdm


class GPUPatchGeneratorV2:
    """GPU-accelerated patch generation using point projection math."""

    def __init__(self, mesh, annotations, patch_size=100, patch_radius=5.0, device='cuda', precompute_neighbors=True):
        if not torch.cuda.is_available() and device == 'cuda':
            device = 'cpu'

        self.device = torch.device(device)
        self.patch_size = patch_size
        self.patch_radius = patch_radius
        self.mesh = mesh
        self.annotations = annotations
        self.precompute_neighbors = precompute_neighbors

        if precompute_neighbors:
            self.neighbors = self._precompute_neighbors(mesh.vertices, patch_radius)
            self.kdtree = None
        else:
            self.neighbors = None
            self.kdtree = KDTree(mesh.vertices)

        self.vertices_gpu = torch.tensor(mesh.vertices, device=self.device, dtype=torch.float32)
        self.normals_gpu = torch.tensor(mesh.vertex_normals, device=self.device, dtype=torch.float32)

        if hasattr(mesh.visual, 'vertex_colors') and mesh.visual.vertex_colors is not None:
            self.colors_gpu = torch.tensor(mesh.visual.vertex_colors[:, :3] / 255.0, device=self.device, dtype=torch.float32)
        else:
            labels = np.array(annotations['labels'])
            colors = np.zeros((len(mesh.vertices), 3), dtype=np.float32)
            colors[labels == 0] = [255/255, 180/255, 180/255]
            colors[labels != 0] = [200/255, 200/255, 255/255]
            self.colors_gpu = torch.tensor(colors, device=self.device, dtype=torch.float32)

    def _precompute_neighbors(self, vertices, radius):
        kdtree = KDTree(vertices)
        neighbors = {}
        for i in range(len(vertices)):
            neighbor_indices = kdtree.query_ball_point(vertices[i], radius)
            if len(neighbor_indices) < 3:
                neighbor_indices = kdtree.query_ball_point(vertices[i], radius * 2)
            neighbors[i] = neighbor_indices
        return neighbors

    def generate_patches_batch(self, vertex_indices, batch_size=256):
        all_patches = []
        n_vertices = len(vertex_indices)

        for batch_start in tqdm(range(0, n_vertices, batch_size), desc="GPU processing"):
            batch_end = min(batch_start + batch_size, n_vertices)
            batch_idx = vertex_indices[batch_start:batch_end]
            patches = self._generate_batch(batch_idx)
            all_patches.append(patches)

        return np.concatenate(all_patches, axis=0)

    def _generate_batch(self, vertex_indices):
        batch_size = len(vertex_indices)
        patches = torch.zeros(batch_size, self.patch_size, self.patch_size, 3, device=self.device, dtype=torch.float32)

        for i, vertex_idx in enumerate(vertex_indices):
            if self.neighbors is not None:
                neighbor_indices = self.neighbors.get(vertex_idx, [])
            else:
                center_point = self.mesh.vertices[vertex_idx]
                neighbor_indices = self.kdtree.query_ball_point(center_point, self.patch_radius)
                if len(neighbor_indices) < 3:
                    neighbor_indices = self.kdtree.query_ball_point(center_point, self.patch_radius * 2)

            if len(neighbor_indices) < 3:
                continue

            pixel_coords, colors = self._project_neighbors_gpu(vertex_idx, neighbor_indices)
            if pixel_coords is None:
                continue

            patches[i] = self._rasterize_points_gpu(pixel_coords, colors)

            if self.device.type == 'cuda' and (i + 1) % 512 == 0:
                torch.cuda.empty_cache()

        patches = self._blur_gpu(patches)
        patches_np = (patches * 255).clamp(0, 255).to(torch.uint8).cpu().numpy()

        if self.device.type == 'cuda':
            torch.cuda.empty_cache()

        return patches_np

    def _project_neighbors_gpu(self, vertex_idx, neighbor_indices):
        center_pos = self.vertices_gpu[vertex_idx]
        normal = self.normals_gpu[vertex_idx]
        u, v = self._compute_tangent_basis(normal)

        neighbor_indices_tensor = torch.tensor(neighbor_indices, device=self.device, dtype=torch.long)
        neighbor_pos = self.vertices_gpu[neighbor_indices_tensor]
        neighbor_colors = self.colors_gpu[neighbor_indices_tensor]

        relative_pos = neighbor_pos - center_pos
        u_coords = torch.matmul(relative_pos, u)
        v_coords = torch.matmul(relative_pos, v)

        scale = self.patch_size / (2 * self.patch_radius)
        pixel_u = ((u_coords + self.patch_radius) * scale).long()
        pixel_v = ((v_coords + self.patch_radius) * scale).long()

        valid_mask = (pixel_u >= 0) & (pixel_u < self.patch_size) & (pixel_v >= 0) & (pixel_v < self.patch_size)

        if valid_mask.sum() == 0:
            return None, None

        pixel_coords = torch.stack([pixel_u[valid_mask], pixel_v[valid_mask]], dim=1)
        colors = neighbor_colors[valid_mask]
        return pixel_coords, colors

    def _compute_tangent_basis(self, normal):
        if abs(normal[0].item()) < 0.9:
            temp = torch.tensor([1, 0, 0], device=self.device, dtype=torch.float32)
        else:
            temp = torch.tensor([0, 1, 0], device=self.device, dtype=torch.float32)

        u = torch.cross(normal, temp)
        u = u / torch.norm(u)
        v = torch.cross(normal, u)
        v = v / torch.norm(v)
        return u, v

    def _rasterize_points_gpu(self, pixel_coords, colors):
        patch = torch.zeros(self.patch_size, self.patch_size, 3, device=self.device, dtype=torch.float32)
        pixel_x = pixel_coords[:, 0]
        pixel_y = pixel_coords[:, 1]
        patch[pixel_y, pixel_x, :] = colors
        return patch

    def _blur_gpu(self, patches):
        import kornia
        patches = patches.permute(0, 3, 1, 2)
        blurred = kornia.filters.gaussian_blur2d(patches, kernel_size=(3, 3), sigma=(0.5, 0.5))
        return blurred.permute(0, 2, 3, 1)


def test_gpu_availability():
    if not torch.cuda.is_available():
        print("CUDA not available")
        return False
    print(f"CUDA: {torch.cuda.get_device_name(0)}, {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    return True


if __name__ == "__main__":
    test_gpu_availability()
