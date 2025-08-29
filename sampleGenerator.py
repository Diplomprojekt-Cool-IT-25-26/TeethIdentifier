"""
hell yeah python file

this is where the sample taking functionality should be
"""

import numpy as np
import trimesh
import cv2
from scipy.spatial import KDTree
from sklearn.neighbors import NearestNeighbors
import matplotlib.pyplot as plt
from typing import List, Tuple, Optional, Dict
import random
from dataclasses import dataclass

@dataclass
class TrainingSample:
    """Container for a single training sample"""
    image_patch: np.ndarray  # The I(p_i) - image patch around point
    label: int              # The O(p_i) - binary classification (0=gingiva, 1=tooth)
    point_index: int        # Original vertex index
    point_coords: np.ndarray # 3D coordinates of the point

class DentalMeshSampler:
    """
    Generates training samples for dental segmentation from annotated mesh data
    """
    
    def __init__(self, mesh_file: str, patch_size: int = 100, patch_radius: float = 5.0):
        """
        Initialize the sampler
        
        Args:
            mesh_file: Path to mesh file (supports .ply, .obj, etc.)
            patch_size: Size of generated image patches (patch_size x patch_size)
            patch_radius: Radius around each point to include in patch generation
        """
        self.mesh = trimesh.load(mesh_file)
        self.patch_size = patch_size
        self.patch_radius = patch_radius
        
        # Build KDTree for efficient neighbor queries
        self.kdtree = KDTree(self.mesh.vertices)
        
        # Vertex labels (0=gingiva, 1=tooth) - should be loaded from mesh attributes
        self.vertex_labels = self._load_vertex_labels()
        
        print(f"Loaded mesh with {len(self.mesh.vertices)} vertices")
        print(f"Label distribution - Gingiva: {np.sum(self.vertex_labels == 0)}, Teeth: {np.sum(self.vertex_labels == 1)}")
    
    def _load_vertex_labels(self) -> np.ndarray:
        """
        Load vertex labels from mesh attributes
        In practice, this should load the annotation data from your segmentation tool
        For now, creates dummy labels for demonstration
        """
        # Try to load from mesh vertex attributes if available
        if hasattr(self.mesh.visual, 'vertex_attributes'):
            if 'labels' in self.mesh.visual.vertex_attributes:
                return np.array(self.mesh.visual.vertex_attributes['labels'])
        
        # Fallback: create dummy labels for demonstration
        # In real implementation, replace this with actual label loading
        print("Warning: No labels found in mesh, creating dummy labels for demonstration")
        labels = np.random.choice([0, 1], size=len(self.mesh.vertices), p=[0.3, 0.7])
        return labels
    
    def sample_training_points(self, 
                             n_samples: int, 
                             sampling_strategy: str = 'random',
                             balance_classes: bool = True) -> List[int]:
        """
        Sample training points from the mesh
        
        Args:
            n_samples: Number of samples to generate
            sampling_strategy: 'random', 'uniform_surface', or 'boundary_focused'
            balance_classes: Whether to balance tooth/gingiva samples
            
        Returns:
            List of vertex indices to use for training
        """
        if sampling_strategy == 'random':
            return self._random_sampling(n_samples, balance_classes)
        elif sampling_strategy == 'uniform_surface':
            return self._uniform_surface_sampling(n_samples, balance_classes)
        elif sampling_strategy == 'boundary_focused':
            return self._boundary_focused_sampling(n_samples, balance_classes)
        else:
            raise ValueError(f"Unknown sampling strategy: {sampling_strategy}")
    
    def _random_sampling(self, n_samples: int, balance_classes: bool) -> List[int]:
        """Simple random sampling of vertices"""
        if balance_classes:
            tooth_indices = np.where(self.vertex_labels == 1)[0]
            gingiva_indices = np.where(self.vertex_labels == 0)[0]
            
            n_tooth = n_samples // 2
            n_gingiva = n_samples - n_tooth
            
            selected_tooth = np.random.choice(tooth_indices, 
                                            min(n_tooth, len(tooth_indices)), 
                                            replace=False)
            selected_gingiva = np.random.choice(gingiva_indices, 
                                              min(n_gingiva, len(gingiva_indices)), 
                                              replace=False)
            
            return np.concatenate([selected_tooth, selected_gingiva]).tolist()
        else:
            return np.random.choice(len(self.mesh.vertices), n_samples, replace=False).tolist()
    
    def _uniform_surface_sampling(self, n_samples: int, balance_classes: bool) -> List[int]:
        """Sample points more uniformly across the surface area"""
        # Weight by face area to get more uniform surface distribution
        face_areas = self.mesh.area_faces
        vertex_weights = np.zeros(len(self.mesh.vertices))
        
        # Accumulate area weights for each vertex
        for i, face in enumerate(self.mesh.faces):
            for vertex_idx in face:
                vertex_weights[vertex_idx] += face_areas[i] / 3  # Distribute face area equally
        
        vertex_weights /= np.sum(vertex_weights)
        
        if balance_classes:
            # Separate sampling for each class
            tooth_indices = np.where(self.vertex_labels == 1)[0]
            gingiva_indices = np.where(self.vertex_labels == 0)[0]
            
            tooth_weights = vertex_weights[tooth_indices] / np.sum(vertex_weights[tooth_indices])
            gingiva_weights = vertex_weights[gingiva_indices] / np.sum(vertex_weights[gingiva_indices])
            
            n_tooth = n_samples // 2
            n_gingiva = n_samples - n_tooth
            
            selected_tooth = np.random.choice(tooth_indices, 
                                            min(n_tooth, len(tooth_indices)), 
                                            replace=False, p=tooth_weights)
            selected_gingiva = np.random.choice(gingiva_indices, 
                                              min(n_gingiva, len(gingiva_indices)), 
                                              replace=False, p=gingiva_weights)
            
            return np.concatenate([selected_tooth, selected_gingiva]).tolist()
        else:
            return np.random.choice(len(self.mesh.vertices), n_samples, 
                                  replace=False, p=vertex_weights).tolist()
    
    def _boundary_focused_sampling(self, n_samples: int, balance_classes: bool) -> List[int]:
        """Focus sampling near tooth-gingiva boundaries"""
        # Find boundary vertices (vertices with both tooth and gingiva neighbors)
        boundary_vertices = []
        
        for i, vertex in enumerate(self.mesh.vertices):
            neighbors = self._get_vertex_neighbors(i)
            neighbor_labels = self.vertex_labels[neighbors]
            
            # Check if this vertex has neighbors of different classes
            if len(np.unique(neighbor_labels)) > 1:
                boundary_vertices.append(i)
        
        boundary_vertices = np.array(boundary_vertices)
        
        if len(boundary_vertices) < n_samples // 2:
            # If not enough boundary vertices, supplement with random sampling
            remaining_samples = n_samples - len(boundary_vertices)
            non_boundary = np.setdiff1d(np.arange(len(self.mesh.vertices)), boundary_vertices)
            additional = np.random.choice(non_boundary, remaining_samples, replace=False)
            return np.concatenate([boundary_vertices, additional]).tolist()
        else:
            # Sample from boundary vertices
            selected_boundary = np.random.choice(boundary_vertices, n_samples // 2, replace=False)
            # Add some random samples
            remaining_samples = n_samples - len(selected_boundary)
            non_boundary = np.setdiff1d(np.arange(len(self.mesh.vertices)), boundary_vertices)
            additional = np.random.choice(non_boundary, remaining_samples, replace=False)
            return np.concatenate([selected_boundary, additional]).tolist()
    
    def _get_vertex_neighbors(self, vertex_idx: int) -> np.ndarray:
        """Get neighboring vertices within a small radius"""
        neighbors = self.kdtree.query_ball_point(self.mesh.vertices[vertex_idx], 
                                                r=self.patch_radius * 0.3)
        return np.array(neighbors)
    
    def generate_image_patch(self, vertex_idx: int) -> np.ndarray:
        """
        Generate image patch around a vertex through texture projection
        
        Args:
            vertex_idx: Index of the vertex to generate patch for
            
        Returns:
            Image patch as numpy array of shape (patch_size, patch_size, 3)
        """
        center_point = self.mesh.vertices[vertex_idx]
        
        # Find all vertices within radius
        neighbor_indices = self.kdtree.query_ball_point(center_point, self.patch_radius)
        
        if len(neighbor_indices) < 3:
            # Not enough neighbors, return black patch
            return np.zeros((self.patch_size, self.patch_size, 3), dtype=np.uint8)
        
        # Get local coordinate system
        normal = self.mesh.vertex_normals[vertex_idx]
        
        # Create two orthogonal vectors in the tangent plane
        # Choose an arbitrary vector not parallel to normal
        if abs(normal[0]) < 0.9:
            temp = np.array([1, 0, 0])
        else:
            temp = np.array([0, 1, 0])
        
        # Create orthogonal basis
        u = np.cross(normal, temp)
        u = u / np.linalg.norm(u)
        v = np.cross(normal, u)
        v = v / np.linalg.norm(v)
        
        # Project neighbor vertices onto local 2D coordinate system
        neighbor_points = self.mesh.vertices[neighbor_indices]
        relative_points = neighbor_points - center_point
        
        # Project to 2D local coordinates
        u_coords = np.dot(relative_points, u)
        v_coords = np.dot(relative_points, v)
        
        # Convert to image coordinates
        scale = self.patch_size / (2 * self.patch_radius)
        img_u = ((u_coords + self.patch_radius) * scale).astype(int)
        img_v = ((v_coords + self.patch_radius) * scale).astype(int)
        
        # Create image patch
        patch = np.zeros((self.patch_size, self.patch_size, 3), dtype=np.uint8)
        
        # Simple texture mapping - in practice, use actual texture coordinates
        for i, (iu, iv) in enumerate(zip(img_u, img_v)):
            if 0 <= iu < self.patch_size and 0 <= iv < self.patch_size:
                # Use vertex color or generate based on position/normal
                if hasattr(self.mesh.visual, 'vertex_colors'):
                    color = self.mesh.visual.vertex_colors[neighbor_indices[i]][:3]
                else:
                    # Generate color based on vertex normal (for visualization)
                    vertex_normal = self.mesh.vertex_normals[neighbor_indices[i]]
                    color = ((vertex_normal + 1) * 127.5).astype(np.uint8)
                
                patch[iv, iu] = color
        
        # Apply Gaussian blur to smooth the patch
        patch = cv2.GaussianBlur(patch, (5, 5), 1.0)
        
        return patch
    
    def generate_training_samples(self, 
                                sample_indices: List[int],
                                save_patches: bool = False,
                                patch_save_dir: str = "patches") -> List[TrainingSample]:
        """
        Generate training samples for given vertex indices
        
        Args:
            sample_indices: List of vertex indices to generate samples for
            save_patches: Whether to save image patches to disk
            patch_save_dir: Directory to save patches if save_patches=True
            
        Returns:
            List of TrainingSample objects
        """
        if save_patches:
            import os
            os.makedirs(patch_save_dir, exist_ok=True)
        
        training_samples = []
        
        for i, vertex_idx in enumerate(sample_indices):
            # Generate image patch
            patch = self.generate_image_patch(vertex_idx)
            
            # Get label
            label = self.vertex_labels[vertex_idx]
            
            # Create training sample
            sample = TrainingSample(
                image_patch=patch,
                label=label,
                point_index=vertex_idx,
                point_coords=self.mesh.vertices[vertex_idx].copy()
            )
            
            training_samples.append(sample)
            
            # Save patch if requested
            if save_patches:
                class_name = "tooth" if label == 1 else "gingiva"
                filename = f"{patch_save_dir}/patch_{i:05d}_{class_name}_v{vertex_idx}.png"
                cv2.imwrite(filename, cv2.cvtColor(patch, cv2.COLOR_RGB2BGR))
            
            if (i + 1) % 100 == 0:
                print(f"Generated {i + 1}/{len(sample_indices)} samples")
        
        return training_samples
    
    def save_training_data(self, 
                          training_samples: List[TrainingSample], 
                          output_file: str):
        """Save training samples to disk"""
        import pickle
        
        # Convert to format suitable for ML training
        data = {
            'images': np.array([sample.image_patch for sample in training_samples]),
            'labels': np.array([sample.label for sample in training_samples]),
            'point_indices': np.array([sample.point_index for sample in training_samples]),
            'point_coords': np.array([sample.point_coords for sample in training_samples])
        }
        
        with open(output_file, 'wb') as f:
            pickle.dump(data, f)
        
        print(f"Saved {len(training_samples)} training samples to {output_file}")
    
    def visualize_samples(self, training_samples: List[TrainingSample], n_show: int = 16):
        """Visualize a subset of training samples"""
        n_show = min(n_show, len(training_samples))
        samples_to_show = random.sample(training_samples, n_show)
        
        fig, axes = plt.subplots(4, 4, figsize=(12, 12))
        axes = axes.flatten()
        
        for i, sample in enumerate(samples_to_show):
            if i >= 16:
                break
                
            axes[i].imshow(sample.image_patch)
            class_name = "Tooth" if sample.label == 1 else "Gingiva"
            axes[i].set_title(f"{class_name} (v{sample.point_index})")
            axes[i].axis('off')
        
        plt.tight_layout()
        plt.show()

# Example usage
if __name__ == "__main__":
    # Initialize sampler
    sampler = DentalMeshSampler("path/to/your/mesh.ply", patch_size=100, patch_radius=5.0)
    
    # Sample training points
    sample_indices = sampler.sample_training_points(
        n_samples=1000, 
        sampling_strategy='boundary_focused',
        balance_classes=True
    )
    
    # Generate training samples
    training_samples = sampler.generate_training_samples(
        sample_indices,
        save_patches=True,
        patch_save_dir="training_patches"
    )
    
    # Save training data
    sampler.save_training_data(training_samples, "dental_training_data.pkl")
    
    # Visualize samples
    sampler.visualize_samples(training_samples)
    
    print(f"Generated {len(training_samples)} training samples")
    print(f"Class distribution: {np.bincount([s.label for s in training_samples])}")