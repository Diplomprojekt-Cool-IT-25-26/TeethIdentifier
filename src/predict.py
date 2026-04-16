"""Inference Pipeline for TeethIdentifier."""

import os
import sys
import json
import yaml
import argparse
import numpy as np
import trimesh
from pathlib import Path
from datetime import datetime
from tqdm import tqdm
from typing import Tuple, List

# torch must be imported before TensorFlow to avoid a Windows DLL conflict
# where TF's CUDA DLLs prevent torch/lib/shm.dll from loading
try:
    import torch
    TORCH_AVAILABLE = True
except (ImportError, OSError):
    TORCH_AVAILABLE = False

# Enable TF memory growth so PyTorch can use the rest of VRAM for patch
# generation. Without this, TF reserves ~70% of VRAM on init and the GPU
# patch generator thrashes — progressively slower per batch.
import tensorflow as tf
for gpu in tf.config.experimental.list_physical_devices('GPU'):
    tf.config.experimental.set_memory_growth(gpu, True)
from tensorflow import keras

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from tools.dataset_sampler import DatasetSampler


class TeethPredictor:
    """Inference pipeline for teeth vs gingiva classification."""

    def __init__(self, model_path: str, config_path: str = 'config.yaml'):
        self.model_path = model_path
        self.config = self.load_config(config_path)
        self.model = None
        self.sampler = None

    def load_config(self, config_path: str) -> dict:
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)

    def load_model(self) -> keras.Model:
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"Model not found: {self.model_path}")
        self.model = keras.models.load_model(self.model_path)
        print(f"Model loaded: {self.model_path}")
        return self.model

    def load_obj_file(self, obj_path: str) -> Tuple[trimesh.Trimesh, str, str]:
        if not os.path.exists(obj_path):
            raise FileNotFoundError(f"OBJ file not found: {obj_path}")

        mesh = trimesh.load(obj_path, process=False)
        base_name = Path(obj_path).stem
        parts = base_name.split('_')

        if len(parts) >= 2 and parts[-1] in ['upper', 'lower']:
            patient_id = '_'.join(parts[:-1])
            jaw = parts[-1]
        else:
            patient_id = base_name
            jaw = 'unknown'

        print(f"Loaded {Path(obj_path).name}: {len(mesh.vertices):,} vertices ({jaw})")
        return mesh, patient_id, jaw

    def generate_patches_for_mesh(self, mesh: trimesh.Trimesh, obj_path: str) -> np.ndarray:
        use_gpu = (TORCH_AVAILABLE and torch.cuda.is_available() and
                   self.config.get('gpu_acceleration', {}).get('enabled', True))

        if use_gpu:
            try:
                from tools.gpu_patch_generator_v3 import GPUPatchGeneratorV3

                generator = GPUPatchGeneratorV3(
                    mesh=mesh,
                    patch_size=self.config['data_generation']['patch_size'],
                    patch_radius=self.config['data_generation']['patch_radius'],
                    device='cuda'
                )

                vertex_indices = np.arange(len(mesh.vertices))
                batch_size = self.config.get('gpu_acceleration', {}).get('batch_size', 1024)
                patches = generator.generate_patches_batch(vertex_indices, batch_size=batch_size)
                print(f"Generated {len(patches):,} patches (GPU)")
                return patches

            except Exception as e:
                print(f"GPU failed ({e}), using CPU")
                use_gpu = False

        if not use_gpu:
            if self.sampler is None:
                self.sampler = DatasetSampler(
                    dataset_root=os.path.dirname(obj_path),
                    patch_size=self.config['data_generation']['patch_size'],
                    patch_radius=self.config['data_generation']['patch_radius']
                )

            self.sampler.current_mesh = mesh
            from scipy.spatial import KDTree
            self.sampler.kdtree = KDTree(mesh.vertices)

            patches = []
            for vertex_idx in tqdm(range(len(mesh.vertices)), desc="Generating patches"):
                try:
                    patch = self.sampler._generate_image_patch(vertex_idx)
                    patches.append(patch)
                except Exception:
                    zero_patch = np.zeros((self.config['data_generation']['patch_size'],
                                          self.config['data_generation']['patch_size'], 3), dtype=np.uint8)
                    patches.append(zero_patch)

            patches = np.array(patches)
            print(f"Generated {len(patches):,} patches (CPU)")
            return patches

    def predict_vertices(self, patches: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        batch_size = self.config['inference'].get('batch_size', 128)
        y_pred_proba = self.model.predict(patches, batch_size=batch_size, verbose=1)
        threshold = self.config['inference'].get('threshold', 0.5)
        y_pred = (y_pred_proba > threshold).astype(int).flatten()
        y_pred_proba = y_pred_proba.flatten()

        n_gingiva = np.sum(y_pred == 0)
        n_teeth = np.sum(y_pred == 1)
        print(f"Predictions: {n_gingiva:,} gingiva, {n_teeth:,} teeth")
        return y_pred, y_pred_proba

    def create_json_sidecar(self, predictions: np.ndarray, obj_path: str,
                           patient_id: str, jaw: str, output_path: str) -> None:
        output_data = {
            "name": os.path.basename(obj_path),
            "labels": predictions.tolist(),
            "instances": [],
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

        with open(output_path, 'w') as f:
            json.dump(output_data, f, indent=2)
        print(f"Saved: {output_path}")

    def predict_single_scan(self, obj_path: str, output_dir: str) -> str:
        if self.model is None:
            self.load_model()

        mesh, patient_id, jaw = self.load_obj_file(obj_path)
        patches = self.generate_patches_for_mesh(mesh, obj_path)
        predictions, probabilities = self.predict_vertices(patches)

        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        base_name = Path(obj_path).stem
        output_path = output_dir / f"{base_name}_predictions.json"

        self.create_json_sidecar(predictions, obj_path, patient_id, jaw, str(output_path))
        return str(output_path)

    def batch_predict(self, obj_files: List[str], output_dir: str) -> List[str]:
        print(f"Processing {len(obj_files)} files")
        if self.model is None:
            self.load_model()

        output_paths = []
        for i, obj_path in enumerate(obj_files, 1):
            print(f"[{i}/{len(obj_files)}] {Path(obj_path).name}")
            try:
                output_path = self.predict_single_scan(obj_path, output_dir)
                output_paths.append(output_path)
            except Exception as e:
                print(f"[ERROR] {obj_path}: {e}")

        print(f"Complete: {len(output_paths)}/{len(obj_files)} successful")
        return output_paths


def main():
    parser = argparse.ArgumentParser(description="TeethIdentifier Inference")
    parser.add_argument('--obj', type=str, required=True, help="OBJ file or directory")
    parser.add_argument('--output', type=str, default='exports/predictions', help="Output directory")
    parser.add_argument('--model', type=str, default='ml_outputs/models/teeth_classifier.keras', help="Model path")
    parser.add_argument('--config', type=str, default='config.yaml', help="Config file")
    args = parser.parse_args()

    predictor = TeethPredictor(model_path=args.model, config_path=args.config)

    if os.path.isfile(args.obj):
        predictor.predict_single_scan(args.obj, args.output)
    elif os.path.isdir(args.obj):
        import glob
        obj_files = glob.glob(os.path.join(args.obj, '**/*.obj'), recursive=True)
        if not obj_files:
            print(f"[ERROR] No OBJ files in: {args.obj}")
            return 1
        predictor.batch_predict(obj_files, args.output)
    else:
        print(f"[ERROR] Invalid path: {args.obj}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
