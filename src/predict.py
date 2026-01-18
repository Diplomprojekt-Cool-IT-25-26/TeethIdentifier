"""
Inference Pipeline for TeethIdentifier.

This script provides end-to-end inference from OBJ files to labeled predictions:
1. Load OBJ file (3D mesh)
2. Generate 2D patches for all vertices
3. Run predictions with trained model
4. Output JSON side-car file with vertex labels

Usage:
    python src/predict.py --obj data/scan.obj --output exports/predictions/

Output format (JSON side-car matching input format):
    {
        "name": "PATIENT_001_upper.obj",
        "labels": [0, 1, 1, 0, ...],  # One label per vertex
        "instances": [],
        "jaw": "upper",
        "prediction_metadata": {
            "model_version": "v1.0",
            "timestamp": "2026-01-05T...",
            "threshold": 0.5
        }
    }
"""

import os
import sys
import json
import yaml
import argparse
import numpy as np
import trimesh
from tensorflow import keras
from pathlib import Path
from datetime import datetime
from tqdm import tqdm
from typing import Tuple, List

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dataset_sampler import DatasetSampler

# GPU acceleration (optional import)
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


class TeethPredictor:
    """
    Inference pipeline for teeth vs gingiva classification.

    Takes an OBJ file and produces a JSON side-car with predicted labels.
    """

    def __init__(self, model_path: str, config_path: str = 'config.yaml'):
        """
        Initialize the predictor.

        Args:
            model_path: Path to trained model (.keras file)
            config_path: Path to configuration YAML
        """
        self.model_path = model_path
        self.config = self.load_config(config_path)
        self.model = None
        self.sampler = None

    def load_config(self, config_path: str) -> dict:
        """Load configuration from YAML."""
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)

    def load_model(self) -> keras.Model:
        """Load the trained model."""
        print(f"Loading model from: {self.model_path}")

        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"Model not found: {self.model_path}")

        self.model = keras.models.load_model(self.model_path)
        print(f"✓ Model loaded successfully")

        return self.model

    def load_obj_file(self, obj_path: str) -> Tuple[trimesh.Trimesh, str, str]:
        """
        Load OBJ file and extract metadata.

        Args:
            obj_path: Path to OBJ file

        Returns:
            Tuple of (mesh, patient_id, jaw)
        """
        if not os.path.exists(obj_path):
            raise FileNotFoundError(f"OBJ file not found: {obj_path}")

        print(f"\nLoading OBJ file: {obj_path}")

        # Load mesh
        mesh = trimesh.load(obj_path)

        # Extract metadata from filename
        # Expected format: PATIENT_ID_upper.obj or PATIENT_ID_lower.obj
        base_name = Path(obj_path).stem
        parts = base_name.split('_')

        if len(parts) >= 2 and parts[-1] in ['upper', 'lower']:
            patient_id = '_'.join(parts[:-1])
            jaw = parts[-1]
        else:
            patient_id = base_name
            jaw = 'unknown'

        print(f"  Patient ID: {patient_id}")
        print(f"  Jaw: {jaw}")
        print(f"  Vertices: {len(mesh.vertices):,}")

        return mesh, patient_id, jaw

    def generate_patches_for_mesh(self, mesh: trimesh.Trimesh,
                                  obj_path: str) -> np.ndarray:
        """
        Generate 2D patches for all vertices in the mesh.
        Uses GPU acceleration if available, falls back to CPU.

        Args:
            mesh: Trimesh object
            obj_path: Path to OBJ file (for sampler initialization)

        Returns:
            Numpy array of patches (n_vertices, 100, 100, 3)
        """
        print("\nGenerating patches for all vertices...")

        # Check if GPU acceleration is available and enabled
        use_gpu = (
            TORCH_AVAILABLE and
            torch.cuda.is_available() and
            self.config.get('gpu_acceleration', {}).get('enabled', True)
        )

        if use_gpu:
            try:
                # Try V2 (math approach) first, then V1 (rendering)
                try:
                    from gpu_patch_generator_v2 import GPUPatchGeneratorV2
                    print("[GPU] Using GPU Math approach (V2) - 12× speedup")
                    GPUGenerator = GPUPatchGeneratorV2
                except ImportError:
                    from gpu_patch_generator import GPUPatchGenerator
                    print("[GPU] Using GPU Rendering approach (V1)")
                    GPUGenerator = GPUPatchGenerator

                # Load annotations (needed for GPU generator)
                json_path = obj_path.replace('.obj', '.json')
                with open(json_path, 'r') as f:
                    annotations = json.load(f)

                # Initialize GPU generator
                generator = GPUGenerator(
                    mesh=mesh,
                    annotations=annotations,
                    patch_size=self.config['data_generation']['patch_size'],
                    patch_radius=self.config['data_generation']['patch_radius'],
                    device='cuda'
                )

                # Generate all patches in batches
                vertex_indices = np.arange(len(mesh.vertices))
                batch_size = self.config.get('gpu_acceleration', {}).get('batch_size', 1024)

                patches = generator.generate_patches_batch(vertex_indices, batch_size=batch_size)

                print(f"[OK] Generated {len(patches):,} patches (GPU)")
                return patches

            except Exception as e:
                print(f"[WARNING] GPU acceleration failed: {e}")
                print("[CPU] Falling back to CPU implementation")
                use_gpu = False

        # CPU fallback (existing code)
        if not use_gpu:
            print("[CPU] Using CPU patch generation")

            if self.sampler is None:
                self.sampler = DatasetSampler(
                    dataset_root=os.path.dirname(obj_path),
                    patch_size=self.config['data_generation']['patch_size'],
                    patch_radius=self.config['data_generation']['patch_radius']
                )

            # Set current mesh and build KDTree
            self.sampler.current_mesh = mesh
            from scipy.spatial import KDTree
            self.sampler.kdtree = KDTree(mesh.vertices)

            # Generate patches for all vertices
            n_vertices = len(mesh.vertices)
            patches = []

            # Use progress bar
            for vertex_idx in tqdm(range(n_vertices), desc="Generating patches"):
                try:
                    patch = self.sampler._generate_image_patch(vertex_idx)
                    patches.append(patch)
                except Exception as e:
                    # If patch generation fails, create zero patch
                    print(f"\n[WARNING] Failed to generate patch for vertex {vertex_idx}: {e}")
                    zero_patch = np.zeros((self.config['data_generation']['patch_size'],
                                          self.config['data_generation']['patch_size'], 3),
                                         dtype=np.uint8)
                    patches.append(zero_patch)

            patches = np.array(patches)
            print(f"[OK] Generated {len(patches):,} patches (CPU)")

            return patches

    def predict_vertices(self, patches: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Run predictions on patches.

        Args:
            patches: Array of image patches (n_vertices, 100, 100, 3)

        Returns:
            Tuple of (predicted_labels, predicted_probabilities)
        """
        print("\nRunning predictions...")

        # Predict in batches
        batch_size = self.config['inference'].get('batch_size', 128)
        y_pred_proba = self.model.predict(patches, batch_size=batch_size, verbose=1)

        # Convert to binary labels
        threshold = self.config['inference'].get('threshold', 0.5)
        y_pred = (y_pred_proba > threshold).astype(int).flatten()
        y_pred_proba = y_pred_proba.flatten()

        # Print statistics
        n_gingiva = np.sum(y_pred == 0)
        n_teeth = np.sum(y_pred == 1)

        print(f"\n✓ Predictions complete")
        print(f"  Gingiva vertices: {n_gingiva:,} ({n_gingiva/len(y_pred)*100:.1f}%)")
        print(f"  Tooth vertices: {n_teeth:,} ({n_teeth/len(y_pred)*100:.1f}%)")

        return y_pred, y_pred_proba

    def create_json_sidecar(self, predictions: np.ndarray,
                           obj_path: str, patient_id: str, jaw: str,
                           output_path: str) -> None:
        """
        Create JSON side-car file with predictions.

        Args:
            predictions: Predicted labels (n_vertices,)
            obj_path: Original OBJ file path
            patient_id: Patient identifier
            jaw: 'upper' or 'lower'
            output_path: Path to save JSON file
        """
        # Create output dictionary matching input format
        output_data = {
            "name": os.path.basename(obj_path),
            "labels": predictions.tolist(),
            "instances": [],  # Empty for now (could add tooth instance segmentation later)
            "jaw": jaw,
            "prediction_metadata": {
                "model_version": self.config.get('version', 'v1.0'),
                "timestamp": datetime.now().isoformat(),
                "model_path": str(self.model_path),
                "threshold": float(self.config['inference'].get('threshold', 0.5)),
                "num_vertices": len(predictions),
                "num_gingiva": int(np.sum(predictions == 0)),
                "num_teeth": int(np.sum(predictions == 1))
            }
        }

        # Save to JSON
        with open(output_path, 'w') as f:
            json.dump(output_data, f, indent=2)

        print(f"\n✓ Predictions saved to: {output_path}")

    def predict_single_scan(self, obj_path: str, output_dir: str) -> str:
        """
        Run complete inference pipeline on a single OBJ file.

        Args:
            obj_path: Path to OBJ file
            output_dir: Directory to save output JSON

        Returns:
            Path to output JSON file
        """
        print("\n" + "=" * 70)
        print("TeethIdentifier Inference Pipeline")
        print("=" * 70)

        # Load model if not already loaded
        if self.model is None:
            self.load_model()

        # Load OBJ file
        mesh, patient_id, jaw = self.load_obj_file(obj_path)

        # Generate patches
        patches = self.generate_patches_for_mesh(mesh, obj_path)

        # Predict
        predictions, probabilities = self.predict_vertices(patches)

        # Create output directory
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Generate output filename
        base_name = Path(obj_path).stem
        output_path = output_dir / f"{base_name}_predictions.json"

        # Save JSON
        self.create_json_sidecar(predictions, obj_path, patient_id, jaw,
                                str(output_path))

        print("=" * 70)
        print("✓ Inference complete!")
        print("=" * 70 + "\n")

        return str(output_path)

    def batch_predict(self, obj_files: List[str], output_dir: str) -> List[str]:
        """
        Run inference on multiple OBJ files.

        Args:
            obj_files: List of OBJ file paths
            output_dir: Directory to save output JSON files

        Returns:
            List of output JSON file paths
        """
        print(f"\nProcessing {len(obj_files)} OBJ files...")

        # Load model once
        if self.model is None:
            self.load_model()

        output_paths = []

        for i, obj_path in enumerate(obj_files, 1):
            print(f"\n[{i}/{len(obj_files)}] Processing: {obj_path}")

            try:
                output_path = self.predict_single_scan(obj_path, output_dir)
                output_paths.append(output_path)

            except Exception as e:
                print(f"✗ ERROR processing {obj_path}: {e}")
                continue

        print("\n" + "=" * 70)
        print(f"Batch Inference Complete: {len(output_paths)}/{len(obj_files)} successful")
        print("=" * 70)

        return output_paths


def main():
    """Main entry point with CLI arguments."""
    parser = argparse.ArgumentParser(
        description="TeethIdentifier Inference Pipeline - Predict teeth vs gingiva from OBJ files"
    )

    parser.add_argument(
        '--obj', type=str, required=True,
        help="Path to OBJ file (or directory for batch processing)"
    )

    parser.add_argument(
        '--output', type=str, default='exports/predictions',
        help="Output directory for JSON predictions (default: exports/predictions)"
    )

    parser.add_argument(
        '--model', type=str, default='models/teeth_classifier.keras',
        help="Path to trained model (default: models/teeth_classifier.keras)"
    )

    parser.add_argument(
        '--config', type=str, default='config.yaml',
        help="Path to config file (default: config.yaml)"
    )

    args = parser.parse_args()

    # Create predictor
    predictor = TeethPredictor(model_path=args.model, config_path=args.config)

    # Check if input is file or directory
    if os.path.isfile(args.obj):
        # Single file
        predictor.predict_single_scan(args.obj, args.output)

    elif os.path.isdir(args.obj):
        # Batch processing - find all OBJ files
        import glob
        obj_files = glob.glob(os.path.join(args.obj, '**/*.obj'), recursive=True)

        if not obj_files:
            print(f"✗ No OBJ files found in: {args.obj}")
            return 1

        predictor.batch_predict(obj_files, args.output)

    else:
        print(f"✗ ERROR: Invalid path: {args.obj}")
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
