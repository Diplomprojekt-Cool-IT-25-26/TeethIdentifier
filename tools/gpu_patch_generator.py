"""GPU-Accelerated Patch Generation using PyTorch3D."""

import numpy as np
import torch
import trimesh
from scipy.spatial import KDTree
from tqdm import tqdm


class GPUPatchGenerator:
    """GPU-accelerated patch generation using PyTorch3D rendering."""

    def __init__(self, mesh, annotations, patch_size=100, patch_radius=5.0, device='cuda'):
        if not torch.cuda.is_available() and device == 'cuda':
            device = 'cpu'

        self.device = torch.device(device)
        self.patch_size = patch_size
        self.patch_radius = patch_radius
        self.mesh = mesh
        self.annotations = annotations

        self.pytorch3d_mesh = self._create_pytorch3d_mesh()
        self.kdtree = KDTree(mesh.vertices)

    def _create_pytorch3d_mesh(self):
        from pytorch3d.structures import Meshes
        from pytorch3d.renderer import TexturesVertex

        verts = torch.tensor(self.mesh.vertices, dtype=torch.float32, device=self.device)
        faces = torch.tensor(self.mesh.faces, dtype=torch.int64, device=self.device)

        if hasattr(self.mesh.visual, 'vertex_colors') and self.mesh.visual.vertex_colors is not None:
            colors = self.mesh.visual.vertex_colors[:, :3] / 255.0
        else:
            labels = np.array(self.annotations['labels'])
            colors = np.zeros((len(self.mesh.vertices), 3), dtype=np.float32)
            colors[labels == 0] = [255/255, 180/255, 180/255]
            colors[labels != 0] = [200/255, 200/255, 255/255]

        vertex_colors = torch.tensor(colors, dtype=torch.float32, device=self.device)
        textures = TexturesVertex(verts_features=[vertex_colors])
        return Meshes(verts=[verts], faces=[faces], textures=textures)

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

    def _setup_cameras_batch(self, vertex_positions, normals):
        from pytorch3d.renderer import look_at_view_transform, FoVOrthographicCameras

        u, v = self._compute_tangent_basis_batch(normals)
        n = normals

        camera_distance = self.patch_radius * 1.5
        camera_positions = vertex_positions + n * camera_distance

        R, T = look_at_view_transform(eye=camera_positions, at=vertex_positions, up=v, device=self.device)

        cameras = FoVOrthographicCameras(
            R=R, T=T,
            znear=camera_distance * 0.5, zfar=camera_distance * 2.0,
            max_x=self.patch_radius * 1.2, max_y=self.patch_radius * 1.2,
            device=self.device
        )
        return cameras

    def _rasterize_batch(self, cameras):
        from pytorch3d.renderer import RasterizationSettings, MeshRasterizer
        from pytorch3d.ops import interpolate_face_attributes

        batch_size = cameras.R.shape[0]
        meshes_batch = self.pytorch3d_mesh.extend(batch_size)

        raster_settings = RasterizationSettings(image_size=self.patch_size, blur_radius=0.0, faces_per_pixel=1)
        rasterizer = MeshRasterizer(cameras=cameras, raster_settings=raster_settings)
        fragments = rasterizer(meshes_batch)

        verts_colors = self.pytorch3d_mesh.textures.verts_features_packed()
        faces = self.pytorch3d_mesh.faces_packed()
        faces_verts_colors = verts_colors[faces]

        pixel_colors = interpolate_face_attributes(fragments.pix_to_face, fragments.bary_coords, faces_verts_colors)
        images = pixel_colors[:, :, :, 0, :]

        background_mask = (fragments.pix_to_face[:, :, :, 0] == -1).unsqueeze(-1)
        images = torch.where(background_mask, torch.zeros_like(images), images)
        return images

    def _apply_gaussian_blur_gpu(self, images):
        import kornia
        images = images.permute(0, 3, 1, 2)
        blurred = kornia.filters.gaussian_blur2d(images, kernel_size=(3, 3), sigma=(0.5, 0.5))
        return blurred.permute(0, 2, 3, 1)

    def generate_patches_batch(self, vertex_indices, batch_size=1024):
        all_patches = []
        n_vertices = len(vertex_indices)

        for batch_start in tqdm(range(0, n_vertices, batch_size), desc="GPU rendering"):
            batch_end = min(batch_start + batch_size, n_vertices)
            batch_idx = vertex_indices[batch_start:batch_end]

            positions = torch.tensor(self.mesh.vertices[batch_idx], dtype=torch.float32, device=self.device)
            normals = torch.tensor(self.mesh.vertex_normals[batch_idx], dtype=torch.float32, device=self.device)

            cameras = self._setup_cameras_batch(positions, normals)
            images = self._rasterize_batch(cameras)
            images = self._apply_gaussian_blur_gpu(images)

            patches = (images * 255).clamp(0, 255).to(torch.uint8).cpu().numpy()
            all_patches.append(patches)

            del positions, normals, cameras, images
            torch.cuda.empty_cache()

        return np.concatenate(all_patches, axis=0)


def test_gpu_availability():
    if not torch.cuda.is_available():
        print("CUDA not available")
        return False
    print(f"CUDA: {torch.cuda.get_device_name(0)}, {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    return True


if __name__ == "__main__":
    test_gpu_availability()
