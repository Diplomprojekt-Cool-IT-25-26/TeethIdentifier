"""
Module for generating training samples from dental 3D mesh scans.

This module provides functionality to convert 3D dental scans (meshes) into 
training samples suitable for neural network training. It extracts local image 
patches around sampled points for tooth/gingiva classification.

The main approach:
- Sample points from the mesh surface
- Generate local 2D image patches through texture projection
- Create labeled training data for binary classification
"""

# Standard library imports
import os
import random
import pickle
from typing import List, Tuple, Optional, Dict
from dataclasses import dataclass

# Third-party imports
import numpy as np
import trimesh
import cv2
import matplotlib.pyplot as plt
from scipy.spatial import KDTree
from sklearn.neighbors import NearestNeighbors


# Constants
DEFAULT_PATCH_SIZE = 100
DEFAULT_PATCH_RADIUS = 5.0
LABEL_GINGIVA = 0
LABEL_TOOTH = 1
BOUNDARY_NEIGHBOR_RADIUS_FACTOR = 0.3
GAUSSIAN_KERNEL_SIZE = (5, 5)
GAUSSIAN_SIGMA = 1.0
PROGRESS_REPORT_INTERVAL = 100


@dataclass
class TrainingSample:
    """Container for a single training sample."""
    
    image_patch: np.ndarray      # The I(p_i) - image patch around point
    label: int                   # The O(p_i) - binary classification
    point_index: int             # Original vertex index
    point_coords: np.ndarray     # 3D coordinates of the point


class DentalMeshSampler:
    """
    Generates training samples for dental segmentation from annotated mesh data.
    
    This class handles the conversion of 3D textured dental scans into 2D image
    patches suitable for training a neural network classifier.
    """
    
    def __init__(
        self, 
        mesh_file: str, 
        patch_size: int = DEFAULT_PATCH_SIZE, 
        patch_radius: float = DEFAULT_PATCH_RADIUS
    ):
        """
        Initialize the sampler with a mesh file
-> 67               
        Args:
            mesh_file: Path to mesh file (supports .ply, .obj, etc.)
            patch_size: Size of generated image patches (patch_size x patch_size)
            patch_radius: Radius around each point to include in patch generation
        """
        # Error handling: check mesh file existence
        if not isinstance(mesh_file, str) or not mesh_file:
            raise ValueError("mesh_file must be a non-empty string.")
        if not os.path.isfile(mesh_file):
            raise FileNotFoundError(f"Mesh file not found: {mesh_file}")

        # Validate patch_size and patch_radius
        if not isinstance(patch_size, int) or patch_size <= 0:
            raise ValueError("patch_size must be a positive integer.")
        if not isinstance(patch_radius, (int, float)) or patch_radius <= 0:
            raise ValueError("patch_radius must be a positive number.")

        # Error handling: catch mesh loading exceptions
        try:
            self.mesh = trimesh.load(mesh_file)
        except Exception as e:
            raise RuntimeError(f"Failed to load mesh file '{mesh_file}': {e}")
        self.patch_size = patch_size
        self.patch_radius = patch_radius
        
        # Build KDTree for efficient neighbor queries
        self.kdtree = KDTree(self.mesh.vertices)
        
        # Load vertex labels (0=gingiva, 1=tooth)
        self.vertex_labels = self._load_vertex_labels()
        
        # Print initialization statistics
        self._print_initialization_stats()
    
    def _print_initialization_stats(self):
        """Print statistics about the loaded mesh."""
        total_vertices = len(self.mesh.vertices)
        gingiva_count = np.sum(self.vertex_labels == LABEL_GINGIVA)
        tooth_count = np.sum(self.vertex_labels == LABEL_TOOTH)
        
        print(f"Loaded mesh with {total_vertices} vertices")
        print(f"Label distribution - Gingiva: {gingiva_count}, "
              f"Teeth: {tooth_count}")
    
    def _load_vertex_labels(self) -> np.ndarray:
        """
        Load vertex labels from mesh attributes.
        
        Returns:
            Array of vertex labels (0 for gingiva, 1 for tooth)
        """
        # Try to load from mesh vertex attributes if available
        if hasattr(self.mesh.visual, 'vertex_attributes'):
            if 'labels' in self.mesh.visual.vertex_attributes:
                return np.array(self.mesh.visual.vertex_attributes['labels'])
        
        # Fallback: create dummy labels for demonstration
        # TODO: Replace with actual label loading from annotation tool
        print("Warning: No labels found in mesh, creating dummy labels "
              "for demonstration")
        
        # Create dummy labels with 30% gingiva, 70% teeth
        labels = np.random.choice(
            [LABEL_GINGIVA, LABEL_TOOTH], 
            size=len(self.mesh.vertices), 
            p=[0.3, 0.7]
        )
        return labels
    
    def sample_training_points(
        self, 
        n_samples: int, 
        sampling_strategy: str = 'random',
        balance_classes: bool = True
    ) -> List[int]:
        """
        Sample training points from the mesh.
        
        Args:
            n_samples: Number of samples to generate
            sampling_strategy: One of 'random', 'uniform_surface', 
                             or 'boundary_focused'
            balance_classes: Whether to balance tooth/gingiva samples
            
        Returns:
            List of vertex indices to use for training
            
        Raises:
            ValueError: If unknown sampling strategy is provided
        """
        strategy_map = {
            'random': self._random_sampling,
            'uniform_surface': self._uniform_surface_sampling,
            'boundary_focused': self._boundary_focused_sampling
        }
        
        if sampling_strategy not in strategy_map:
            raise ValueError(f"Unknown sampling strategy: {sampling_strategy}")
        
        return strategy_map[sampling_strategy](n_samples, balance_classes)
    
    def _random_sampling(
        self, 
        n_samples: int, 
        balance_classes: bool
    ) -> List[int]:
        """
        Perform simple random sampling of vertices.
        
        Args:
            n_samples: Number of samples to generate
            balance_classes: Whether to balance classes
            
        Returns:
            List of selected vertex indices
        """
        if not balance_classes:
            return np.random.choice(
                len(self.mesh.vertices), 
                n_samples, 
                replace=False
            ).tolist()
        
        # Balance classes by sampling equally from each
        tooth_indices = np.where(self.vertex_labels == LABEL_TOOTH)[0]
        gingiva_indices = np.where(self.vertex_labels == LABEL_GINGIVA)[0]
        
        n_tooth = n_samples // 2
        n_gingiva = n_samples - n_tooth
        
        selected_tooth = np.random.choice(
            tooth_indices,
            min(n_tooth, len(tooth_indices)),
            replace=False
        )
        
        selected_gingiva = np.random.choice(
            gingiva_indices,
            min(n_gingiva, len(gingiva_indices)),
            replace=False
        )
        
        return np.concatenate([selected_tooth, selected_gingiva]).tolist()
    
    def _uniform_surface_sampling(
        self, 
        n_samples: int, 
        balance_classes: bool
    ) -> List[int]:
        """
        Sample points uniformly across the surface area.
        
        Args:
            n_samples: Number of samples to generate
            balance_classes: Whether to balance classes
            
        Returns:
            List of selected vertex indices
        """
        # Calculate vertex weights based on surrounding face areas
        vertex_weights = self._calculate_vertex_area_weights()
        
        if not balance_classes:
            return np.random.choice(
                len(self.mesh.vertices),
                n_samples,
                replace=False,
                p=vertex_weights
            ).tolist()
        
        # Balance classes with area-weighted sampling
        return self._balanced_area_weighted_sampling(
            n_samples, 
            vertex_weights
        )
    
    def _calculate_vertex_area_weights(self) -> np.ndarray:
        """
        Calculate area-based weights for each vertex.
        
        Returns:
            Normalized weight array for vertices
        """
        face_areas = self.mesh.area_faces
        vertex_weights = np.zeros(len(self.mesh.vertices))
        
        # Accumulate area weights for each vertex
        for i, face in enumerate(self.mesh.faces):
            # Distribute face area equally among vertices
            area_per_vertex = face_areas[i] / 3
            for vertex_idx in face:
                vertex_weights[vertex_idx] += area_per_vertex
        
        # Normalize weights
        vertex_weights /= np.sum(vertex_weights)
        return vertex_weights
    
    def _balanced_area_weighted_sampling(
        self, 
        n_samples: int, 
        vertex_weights: np.ndarray
    ) -> List[int]:
        """
        Perform balanced sampling with area weights.
        
        Args:
            n_samples: Number of samples to generate
            vertex_weights: Pre-calculated vertex weights
            
        Returns:
            List of selected vertex indices
        """
        tooth_indices = np.where(self.vertex_labels == LABEL_TOOTH)[0]
        gingiva_indices = np.where(self.vertex_labels == LABEL_GINGIVA)[0]
        
        # Normalize weights within each class
        tooth_weights = vertex_weights[tooth_indices]
        tooth_weights /= np.sum(tooth_weights)
        
        gingiva_weights = vertex_weights[gingiva_indices]
        gingiva_weights /= np.sum(gingiva_weights)
        
        n_tooth = n_samples // 2
        n_gingiva = n_samples - n_tooth
        
        selected_tooth = np.random.choice(
            tooth_indices,
            min(n_tooth, len(tooth_indices)),
            replace=False,
            p=tooth_weights
        )
        
        selected_gingiva = np.random.choice(
            gingiva_indices,
            min(n_gingiva, len(gingiva_indices)),
            replace=False,
            p=gingiva_weights
        )
        
        return np.concatenate([selected_tooth, selected_gingiva]).tolist()
    
    def _boundary_focused_sampling(
        self, 
        n_samples: int, 
        balance_classes: bool
    ) -> List[int]:
        """
        Focus sampling near tooth-gingiva boundaries.
        
        Args:
            n_samples: Number of samples to generate
            balance_classes: Whether to balance classes (unused here)
            
        Returns:
            List of selected vertex indices
        """
        boundary_vertices = self._find_boundary_vertices()
        
        # Calculate how many boundary vs non-boundary samples
        n_boundary = min(len(boundary_vertices), n_samples // 2)
        n_non_boundary = n_samples - n_boundary
        
        # Sample from boundary vertices
        if n_boundary > 0:
            selected_boundary = np.random.choice(
                boundary_vertices,
                n_boundary,
                replace=False
            )
        else:
            selected_boundary = np.array([], dtype=int)
        
        # Sample from non-boundary vertices
        non_boundary = np.setdiff1d(
            np.arange(len(self.mesh.vertices)),
            boundary_vertices
        )
        
        if n_non_boundary > 0 and len(non_boundary) > 0:
            additional = np.random.choice(
                non_boundary,
                min(n_non_boundary, len(non_boundary)),
                replace=False
            )
        else:
            additional = np.array([], dtype=int)
        
        return np.concatenate([selected_boundary, additional]).tolist()
    
    def _find_boundary_vertices(self) -> np.ndarray:
        """
        Find vertices at tooth-gingiva boundaries.
        
        Returns:
            Array of boundary vertex indices
        """
        boundary_vertices = []
        
        for i, vertex in enumerate(self.mesh.vertices):
            neighbors = self._get_vertex_neighbors(i)
            neighbor_labels = self.vertex_labels[neighbors]
            
            # Check if this vertex has neighbors of different classes
            if len(np.unique(neighbor_labels)) > 1:
                boundary_vertices.append(i)
        
        return np.array(boundary_vertices)
    
    def _get_vertex_neighbors(self, vertex_idx: int) -> np.ndarray:
        """
        Get neighboring vertices within a small radius.
        
        Args:
            vertex_idx: Index of the vertex
            
        Returns:
            Array of neighbor vertex indices
        """
        search_radius = self.patch_radius * BOUNDARY_NEIGHBOR_RADIUS_FACTOR
        neighbors = self.kdtree.query_ball_point(
            self.mesh.vertices[vertex_idx],
            r=search_radius
        )
        return np.array(neighbors)
    
    def generate_image_patch(self, vertex_idx: int) -> np.ndarray:
        """
        Generate image patch around a vertex through texture projection.
        
        This creates a 2D representation of the local mesh surface around
        the given vertex, suitable for CNN input.
        
        Args:
            vertex_idx: Index of the vertex to generate patch for
            
        Returns:
            Image patch as numpy array of shape (patch_size, patch_size, 3)
        """
        center_point = self.mesh.vertices[vertex_idx]
        
        # Find all vertices within radius
        neighbor_indices = self.kdtree.query_ball_point(
            center_point,
            self.patch_radius
        )
        
        # Handle case with insufficient neighbors
        if len(neighbor_indices) < 3:
            return np.zeros(
                (self.patch_size, self.patch_size, 3),
                dtype=np.uint8
            )
        
        # Create local 2D coordinate system
        local_coords = self._create_local_coordinate_system(vertex_idx)
        
        # Project neighbors to 2D and create patch
        patch = self._project_and_rasterize(
            vertex_idx,
            neighbor_indices,
            local_coords
        )
        
        # Apply smoothing
        patch = cv2.GaussianBlur(
            patch,
            GAUSSIAN_KERNEL_SIZE,
            GAUSSIAN_SIGMA
        )
        
        return patch
    
    def _create_local_coordinate_system(
        self, 
        vertex_idx: int
    ) -> Dict[str, np.ndarray]:
        """
        Create orthogonal coordinate system in tangent plane.
        
        Args:
            vertex_idx: Index of the center vertex
            
        Returns:
            Dictionary with 'u' and 'v' basis vectors
        """
        normal = self.mesh.vertex_normals[vertex_idx]
        
        # Choose arbitrary vector not parallel to normal
        if abs(normal[0]) < 0.9:
            temp = np.array([1, 0, 0])
        else:
            temp = np.array([0, 1, 0])
        
        # Create orthogonal basis
        u = np.cross(normal, temp)
        u = u / np.linalg.norm(u)
        
        v = np.cross(normal, u)
        v = v / np.linalg.norm(v)
        
        return {'u': u, 'v': v}
    
    def _project_and_rasterize(
        self,
        vertex_idx: int,
        neighbor_indices: List[int],
        local_coords: Dict[str, np.ndarray]
    ) -> np.ndarray:
        """
        Project neighbors to 2D and create image patch.
        
        Args:
            vertex_idx: Index of center vertex
            neighbor_indices: Indices of neighboring vertices
            local_coords: Local coordinate system
            
        Returns:
            Rasterized image patch
        """
        center_point = self.mesh.vertices[vertex_idx]
        neighbor_points = self.mesh.vertices[neighbor_indices]
        relative_points = neighbor_points - center_point
        
        # Project to 2D local coordinates
        u_coords = np.dot(relative_points, local_coords['u'])
        v_coords = np.dot(relative_points, local_coords['v'])
        
        # Convert to image coordinates
        scale = self.patch_size / (2 * self.patch_radius)
        img_u = ((u_coords + self.patch_radius) * scale).astype(int)
        img_v = ((v_coords + self.patch_radius) * scale).astype(int)
        
        # Create image patch
        patch = np.zeros(
            (self.patch_size, self.patch_size, 3),
            dtype=np.uint8
        )
        
        # Rasterize points with colors
        for i, (iu, iv) in enumerate(zip(img_u, img_v)):
            if 0 <= iu < self.patch_size and 0 <= iv < self.patch_size:
                color = self._get_vertex_color(neighbor_indices[i])
                patch[iv, iu] = color
        
        return patch
    
    def _get_vertex_color(self, vertex_idx: int) -> np.ndarray:
        """
        Get color for a vertex from texture or generate procedurally.
        
        Args:
            vertex_idx: Index of the vertex
            
        Returns:
            RGB color as uint8 array
        """
        if hasattr(self.mesh.visual, 'vertex_colors'):
            return self.mesh.visual.vertex_colors[vertex_idx][:3]
        
        # Generate color based on vertex normal for visualization
        vertex_normal = self.mesh.vertex_normals[vertex_idx]
        color = ((vertex_normal + 1) * 127.5).astype(np.uint8)
        return color
    
    def generate_training_samples(
        self,
        sample_indices: List[int],
        save_patches: bool = False,
        patch_save_dir: str = "patches"
    ) -> List[TrainingSample]:
        """
        Generate training samples for given vertex indices.
        
        Args:
            sample_indices: List of vertex indices to generate samples for
            save_patches: Whether to save image patches to disk
            patch_save_dir: Directory to save patches if save_patches=True
            
        Returns:
            List of TrainingSample objects
        """
        if save_patches:
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
                self._save_patch_to_disk(
                    patch,
                    label,
                    vertex_idx,
                    i,
                    patch_save_dir
                )
            
            # Report progress
            if (i + 1) % PROGRESS_REPORT_INTERVAL == 0:
                print(f"Generated {i + 1}/{len(sample_indices)} samples")
        
        return training_samples
    
    def _save_patch_to_disk(
        self,
        patch: np.ndarray,
        label: int,
        vertex_idx: int,
        sample_idx: int,
        save_dir: str
    ):
        """
        Save a patch image to disk.
        
        Args:
            patch: Image patch to save
            label: Class label
            vertex_idx: Vertex index
            sample_idx: Sample index in batch
            save_dir: Directory to save to
        """
        class_name = "tooth" if label == LABEL_TOOTH else "gingiva"
        filename = (f"{save_dir}/patch_{sample_idx:05d}_"
                   f"{class_name}_v{vertex_idx}.png")
        cv2.imwrite(filename, cv2.cvtColor(patch, cv2.COLOR_RGB2BGR))
    
    def save_training_data(
        self,
        training_samples: List[TrainingSample],
        output_file: str
    ):
        """
        Save training samples to disk in format suitable for ML training.
        
        Args:
            training_samples: List of training samples
            output_file: Path to output pickle file
        """
        # Convert to numpy arrays for efficient storage
        data = {
            'images': np.array([s.image_patch for s in training_samples]),
            'labels': np.array([s.label for s in training_samples]),
            'point_indices': np.array([s.point_index for s in training_samples]),
            'point_coords': np.array([s.point_coords for s in training_samples])
        }
        
        with open(output_file, 'wb') as f:
            pickle.dump(data, f)
        
        print(f"Saved {len(training_samples)} training samples to "
              f"{output_file}")
    
    def visualize_samples(
        self,
        training_samples: List[TrainingSample],
        n_show: int = 16
    ):
        """
        Visualize a subset of training samples.
        
        Args:
            training_samples: List of training samples
            n_show: Number of samples to display (max 16)
        """
        n_show = min(n_show, len(training_samples), 16)
        samples_to_show = random.sample(training_samples, n_show)
        
        # Create grid of subplots
        grid_size = 4
        fig, axes = plt.subplots(grid_size, grid_size, figsize=(12, 12))
        axes = axes.flatten()
        
        for i, sample in enumerate(samples_to_show):
            axes[i].imshow(sample.image_patch)
            
            class_name = "Tooth" if sample.label == LABEL_TOOTH else "Gingiva"
            title = f"{class_name} (v{sample.point_index})"
            axes[i].set_title(title)
            axes[i].axis('off')
        
        # Hide unused subplots
        for i in range(n_show, grid_size * grid_size):
            axes[i].axis('off')
        
        plt.tight_layout()
        plt.show()


def main():
    """Main function demonstrating usage of the DentalMeshSampler."""
    
    # Configuration
    mesh_file = "path/to/your/mesh.ply"
    n_samples = 1000
    
    # Initialize sampler
    print("Initializing dental mesh sampler...")
    sampler = DentalMeshSampler(
        mesh_file,
        patch_size=DEFAULT_PATCH_SIZE,
        patch_radius=DEFAULT_PATCH_RADIUS
    )
    
    # Sample training points
    print("\nSampling training points...")
    sample_indices = sampler.sample_training_points(
        n_samples=n_samples,
        sampling_strategy='boundary_focused',
        balance_classes=True
    )
    
    # Generate training samples
    print("\nGenerating training samples...")
    training_samples = sampler.generate_training_samples(
        sample_indices,
        save_patches=True,
        patch_save_dir="training_patches"
    )
    
    # Save training data
    print("\nSaving training data...")
    sampler.save_training_data(
        training_samples,
        "dental_training_data.pkl"
    )
    
    # Visualize samples
    print("\nVisualizing samples...")
    sampler.visualize_samples(training_samples)
    
    # Print summary statistics
    labels = [s.label for s in training_samples]
    class_distribution = np.bincount(labels)
    
    print(f"\nTraining data summary:")
    print(f"Total samples: {len(training_samples)}")
    print(f"Gingiva samples: {class_distribution[LABEL_GINGIVA]}")
    print(f"Tooth samples: {class_distribution[LABEL_TOOTH]}")


if __name__ == "__main__":
    main()