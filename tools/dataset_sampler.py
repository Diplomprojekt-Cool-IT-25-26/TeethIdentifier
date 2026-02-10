"""
Dataset sampler for 3DTeethSeg MICCAI Challenge Dataset.

This module generates training samples from 3D dental scans for neural network training.
Each sample consists of a 2D image patch (I(p_i)) and a binary label (O(p_i)).

Dataset: https://github.com/abenhamadou/3DTeethSeg_MICCAI_Challenges
"""

import os
import json
import pickle
import numpy as np
import trimesh
import cv2
from typing import List, Dict, Any
from dataclasses import dataclass
from scipy.spatial import KDTree


# Constants
GINGIVA_LABEL = 0
DEFAULT_PATCH_SIZE = 100
DEFAULT_PATCH_RADIUS = 5.0


@dataclass
class TrainingSample:
    """Container for a single training sample."""
    
    image_patch: np.ndarray      # I(p_i) - image patch around point
    label: int                   # O(p_i) - binary: 0=gingiva, 1=tooth
    point_index: int             # Original vertex index
    point_coords: np.ndarray     # 3D coordinates
    patient_id: str              # Patient identifier
    jaw: str                     # 'upper' or 'lower'


class DatasetSampler:
    """
    Generates training samples from 3D dental scans.
    
    Usage:
        sampler = DatasetSampler(dataset_root='./data_part_1')
        sampler.load_scan(0)
        sample_indices = sampler.sample_points(1000)
        samples = sampler.generate_samples(sample_indices)
        sampler.save_training_data(samples, 'training_data.pkl')
    """
    
    def __init__(self, dataset_root: str, patch_size: int = DEFAULT_PATCH_SIZE, 
                 patch_radius: float = DEFAULT_PATCH_RADIUS):
        """
        Initialize the dataset sampler.
        
        Args:
            dataset_root: Path to dataset root containing data_part folders
            patch_size: Size of generated image patches (patch_size x patch_size)
            patch_radius: Radius around each point for patch generation
        """
        if not os.path.exists(dataset_root):
            raise FileNotFoundError(f"Dataset root not found: {dataset_root}")
            
        self.dataset_root = dataset_root
        self.patch_size = patch_size
        self.patch_radius = patch_radius
        
        # Find all available scans
        self.scan_files = self._find_scan_files()
        print(f"Found {len(self.scan_files)} scan files")
        
        # Currently loaded scan data
        self.current_mesh = None
        self.current_annotations = None
        self.current_scan_info = None
        self.kdtree = None
    
    def _find_scan_files(self) -> List[Dict[str, str]]:
        """
        Find all OBJ and JSON files in the dataset.
        
        Expected structure:
            dataset_root/
            ├── data_part_X/
            │   ├── upper/PATIENT_ID/
            │   │   ├── PATIENT_ID_upper.obj
            │   │   └── PATIENT_ID_upper.json
            │   └── lower/PATIENT_ID/
            │       ├── PATIENT_ID_lower.obj
            │       └── PATIENT_ID_lower.json
        """
        scan_files = []
        
        for root, dirs, files in os.walk(self.dataset_root):
            obj_files = [f for f in files if f.endswith('.obj')]
            
            for obj_file in obj_files:
                obj_path = os.path.join(root, obj_file)
                base_name = obj_file.replace('.obj', '')
                json_file = base_name + '.json'
                json_path = os.path.join(root, json_file)
                
                if os.path.exists(json_path):
                    # Extract patient ID and jaw from filename
                    parts = base_name.split('_')
                    jaw = parts[-1].lower() if len(parts) > 1 else 'unknown'
                    patient_id = '_'.join(parts[:-1]) if len(parts) > 1 else base_name
                    
                    scan_files.append({
                        'obj_file': obj_path,
                        'json_file': json_path,
                        'patient_id': patient_id,
                        'jaw': jaw,
                        'base_name': base_name
                    })
        
        return scan_files
    
    def load_scan(self, scan_index: int) -> bool:
        """
        Load a specific scan by index.
        
        Args:
            scan_index: Index of scan to load (0 to len(scan_files)-1)
            
        Returns:
            True if successful, False otherwise
        """
        if scan_index >= len(self.scan_files):
            print(f"Scan index {scan_index} out of range")
            return False
            
        scan_info = self.scan_files[scan_index]
        
        try:
            self.current_mesh = trimesh.load(scan_info['obj_file'])
            
            # Load annotations
            with open(scan_info['json_file'], 'r') as f:
                self.current_annotations = json.load(f)
            
            # Build spatial index for efficient neighbor queries
            self.kdtree = KDTree(self.current_mesh.vertices)
            self.current_scan_info = scan_info
            return True
            
        except Exception as e:
            print(f"Error loading scan: {e}")
            return False
    
    def identify_boundary_vertices(self, boundary_radius: float = 5.0) -> np.ndarray:
        """
        Identify vertices that are near class boundaries (teeth/gingiva transitions).

        Args:
            boundary_radius: Distance threshold to consider a vertex near boundary

        Returns:
            Boolean array where True indicates a boundary vertex
        """
        if self.current_annotations is None or self.kdtree is None:
            raise RuntimeError("No scan loaded. Call load_scan() first.")

        labels = np.array(self.current_annotations['labels'])
        vertices = self.current_mesh.vertices
        n_vertices = len(vertices)

        is_boundary = np.zeros(n_vertices, dtype=bool)

        # For each vertex, check if any neighbors have different labels
        for i in range(n_vertices):
            vertex_label = labels[i]

            # Find neighbors within radius
            neighbor_indices = self.kdtree.query_ball_point(vertices[i], boundary_radius)

            # Check if any neighbor has different label
            for neighbor_idx in neighbor_indices:
                if labels[neighbor_idx] != vertex_label:
                    is_boundary[i] = True
                    break

        return is_boundary

    def sample_points(self, n_samples: int, balance_classes: bool = True,
                     boundary_weight: float = 0.0, boundary_radius: float = 5.0) -> List[int]:
        """
        Sample training points from the currently loaded mesh.

        Args:
            n_samples: Number of points to sample
            balance_classes: If True, sample equal numbers of tooth and gingiva points
            boundary_weight: Fraction of samples to take from boundary regions (0.0-1.0)
                           E.g., 0.6 means 60% from boundaries, 40% from clear regions
            boundary_radius: Distance threshold to consider a vertex near boundary

        Returns:
            List of vertex indices to use for training
        """
        if self.current_annotations is None:
            raise RuntimeError("No scan loaded. Call load_scan() first.")

        labels = np.array(self.current_annotations['labels'])

        if balance_classes:
            # Sample equal numbers from each class
            gingiva_indices = np.where(labels == GINGIVA_LABEL)[0]
            teeth_indices = np.where(labels != GINGIVA_LABEL)[0]

            # If boundary weighting is enabled
            if boundary_weight > 0.0:
                # Identify boundary vertices
                is_boundary = self.identify_boundary_vertices(boundary_radius)

                # Split into boundary and clear regions for each class
                gingiva_boundary = gingiva_indices[is_boundary[gingiva_indices]]
                gingiva_clear = gingiva_indices[~is_boundary[gingiva_indices]]
                teeth_boundary = teeth_indices[is_boundary[teeth_indices]]
                teeth_clear = teeth_indices[~is_boundary[teeth_indices]]

                # Calculate how many samples for each category
                n_gingiva_total = n_samples // 2
                n_teeth_total = n_samples - n_gingiva_total

                n_gingiva_boundary = int(n_gingiva_total * boundary_weight)
                n_gingiva_clear = n_gingiva_total - n_gingiva_boundary
                n_teeth_boundary = int(n_teeth_total * boundary_weight)
                n_teeth_clear = n_teeth_total - n_teeth_boundary

                # Sample from each category (with fallback if not enough boundary vertices)
                selected = []

                # Gingiva boundary
                if len(gingiva_boundary) >= n_gingiva_boundary:
                    selected.append(np.random.choice(gingiva_boundary, n_gingiva_boundary, replace=False))
                else:
                    # Not enough boundary vertices, sample what we can
                    selected.append(gingiva_boundary)
                    # Make up the difference from clear vertices
                    n_gingiva_clear += (n_gingiva_boundary - len(gingiva_boundary))

                # Gingiva clear
                if len(gingiva_clear) >= n_gingiva_clear:
                    selected.append(np.random.choice(gingiva_clear, n_gingiva_clear, replace=False))
                else:
                    selected.append(gingiva_clear)

                # Teeth boundary
                if len(teeth_boundary) >= n_teeth_boundary:
                    selected.append(np.random.choice(teeth_boundary, n_teeth_boundary, replace=False))
                else:
                    selected.append(teeth_boundary)
                    n_teeth_clear += (n_teeth_boundary - len(teeth_boundary))

                # Teeth clear
                if len(teeth_clear) >= n_teeth_clear:
                    selected.append(np.random.choice(teeth_clear, n_teeth_clear, replace=False))
                else:
                    selected.append(teeth_clear)

                return np.concatenate(selected).tolist()
            else:
                # Original behavior - no boundary weighting
                n_gingiva = min(n_samples // 2, len(gingiva_indices))
                n_teeth = min(n_samples - n_gingiva, len(teeth_indices))

                selected_gingiva = np.random.choice(gingiva_indices, n_gingiva, replace=False)
                selected_teeth = np.random.choice(teeth_indices, n_teeth, replace=False)

                return np.concatenate([selected_gingiva, selected_teeth]).tolist()
        else:
            # Random sampling
            return np.random.choice(len(labels), min(n_samples, len(labels)),
                                  replace=False).tolist()
    
    def generate_samples(self, sample_indices: List[int]) -> List[TrainingSample]:
        """
        Generate training samples for given vertex indices.
        
        Args:
            sample_indices: List of vertex indices to generate samples for
            
        Returns:
            List of TrainingSample objects
        """
        if self.current_annotations is None or self.current_mesh is None:
            raise RuntimeError("No scan loaded. Call load_scan() first.")
            
        labels = np.array(self.current_annotations['labels'])
        patient_id = self.current_annotations.get('id_patient', 'unknown')
        jaw = self.current_annotations.get('jaw', 'unknown')
        
        samples = []
        
        for i, vertex_idx in enumerate(sample_indices):
            # Generate image patch
            patch = self._generate_image_patch(vertex_idx)
            
            # Get label (convert FDI to binary: 0=gingiva, 1=tooth)
            fdi_label = labels[vertex_idx]
            binary_label = 0 if fdi_label == GINGIVA_LABEL else 1
            
            # Create training sample
            sample = TrainingSample(
                image_patch=patch,
                label=binary_label,
                point_index=vertex_idx,
                point_coords=self.current_mesh.vertices[vertex_idx].copy(),
                patient_id=patient_id,
                jaw=jaw
            )
            
            samples.append(sample)
            
        
        return samples
    
    def _generate_image_patch(self, vertex_idx: int) -> np.ndarray:
        """
        Generate 2D image patch around a vertex.
        
        This creates a local 2D representation of the mesh surface around
        the given vertex by projecting nearby vertices onto a tangent plane.
        
        Args:
            vertex_idx: Index of the center vertex
            
        Returns:
            Image patch as numpy array (patch_size x patch_size x 3)
        """
        center_point = self.current_mesh.vertices[vertex_idx]
        
        # Find all vertices within radius
        neighbor_indices = self.kdtree.query_ball_point(center_point, self.patch_radius)
        
        if len(neighbor_indices) < 3:
            # Not enough neighbors - try larger radius
            neighbor_indices = self.kdtree.query_ball_point(
                center_point, self.patch_radius * 2
            )
        
        if len(neighbor_indices) < 3:
            # Still not enough - return empty patch
            return np.zeros((self.patch_size, self.patch_size, 3), dtype=np.uint8)
        
        # Create local coordinate system using vertex normal
        normal = self.current_mesh.vertex_normals[vertex_idx]
        
        # Create orthogonal basis vectors in tangent plane
        if abs(normal[0]) < 0.9:
            temp = np.array([1, 0, 0])
        else:
            temp = np.array([0, 1, 0])
        
        u = np.cross(normal, temp)
        u = u / np.linalg.norm(u)
        v = np.cross(normal, u)
        v = v / np.linalg.norm(v)
        
        # Project neighbor vertices onto 2D local coordinates
        neighbor_points = self.current_mesh.vertices[neighbor_indices]
        relative_points = neighbor_points - center_point
        
        u_coords = np.dot(relative_points, u)
        v_coords = np.dot(relative_points, v)
        
        # Convert to image coordinates
        scale = self.patch_size / (2 * self.patch_radius)
        img_u = ((u_coords + self.patch_radius) * scale).astype(int)
        img_v = ((v_coords + self.patch_radius) * scale).astype(int)
        
        # Create image patch
        patch = np.zeros((self.patch_size, self.patch_size, 3), dtype=np.uint8)
        
        # Get vertex colors or use labels for coloring
        labels = np.array(self.current_annotations['labels'])
        
        for i, (iu, iv) in enumerate(zip(img_u, img_v)):
            if 0 <= iu < self.patch_size and 0 <= iv < self.patch_size:
                neighbor_idx = neighbor_indices[i]
                
                # Use vertex color if available, otherwise color by label
                if hasattr(self.current_mesh.visual, 'vertex_colors'):
                    color = self.current_mesh.visual.vertex_colors[neighbor_idx][:3]
                else:
                    # Color based on label for visualization
                    if labels[neighbor_idx] == GINGIVA_LABEL:
                        color = np.array([255, 180, 180])  # Pink for gingiva
                    else:
                        color = np.array([200, 200, 255])  # Light blue for teeth
                
                patch[iv, iu] = color
        
        # Apply Gaussian blur for smoothing
        patch = cv2.GaussianBlur(patch, (3, 3), 0.5)
        
        return patch
    
    def save_training_data(self, samples: List[TrainingSample], output_file: str):
        """
        Save training samples to disk in format suitable for ML training.
        
        The data is saved as a pickle file containing:
        - images: numpy array of shape (n_samples, patch_size, patch_size, 3)
        - labels: numpy array of shape (n_samples,) with binary labels
        - metadata: dictionary with patient IDs, jaw info, etc.
        
        Args:
            samples: List of training samples
            output_file: Path to output pickle file
        """
        data = {
            'images': np.array([s.image_patch for s in samples]),
            'labels': np.array([s.label for s in samples]),
            'point_indices': np.array([s.point_index for s in samples]),
            'point_coords': np.array([s.point_coords for s in samples]),
            'patient_ids': [s.patient_id for s in samples],
            'jaws': [s.jaw for s in samples]
        }
        
        with open(output_file, 'wb') as f:
            pickle.dump(data, f)
        


def main():
    """Example usage of the DatasetSampler."""
    
    # Initialize sampler
    dataset_path = "."  # Current directory or path to data_part_X folders
    sampler = DatasetSampler(dataset_path)
    
    if len(sampler.scan_files) == 0:
        print("No scans found! Make sure dataset is in current directory.")
        return
    
    # Load first scan
    if not sampler.load_scan(0):
        print("Failed to load scan")
        return
    
    # Sample points
    print("\nSampling points...")
    sample_indices = sampler.sample_points(n_samples=100, balance_classes=True)
    
    # Generate training samples
    print("\nGenerating training samples...")
    training_samples = sampler.generate_samples(sample_indices)
    
    # Save training data
    sampler.save_training_data(training_samples, "training_data.pkl")
    
    print("Complete")


if __name__ == "__main__":
    main()