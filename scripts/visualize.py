"""Visualize 3D dental scan with model predictions."""

import os
import sys
import json
import yaml
import time
import argparse
import numpy as np
import trimesh
import matplotlib.pyplot as plt
from pathlib import Path
from tensorflow import keras
from scipy.spatial import KDTree

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from tools.dataset_sampler import DatasetSampler

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


def _propagate_predictions(mesh, sampled_indices, sampled_predictions, sampled_confidences):
    """Propagate predictions from sampled vertices to all using nearest neighbor."""
    kdtree = KDTree(mesh.vertices[sampled_indices])
    _, nearest_indices = kdtree.query(mesh.vertices, k=1)
    return sampled_predictions[nearest_indices].tolist(), sampled_confidences[nearest_indices].tolist()


def predict_scan_3d(obj_path: str, model_path: str, config: dict, sample_rate: int = 1):
    """Run model predictions on all vertices of a 3D scan."""
    start = time.time()

    # Load mesh
    mesh = trimesh.load(obj_path, process=False)
    n_verts = len(mesh.vertices)
    print(f"Loaded {n_verts:,} vertices" + (f" (sampling 1/{sample_rate})" if sample_rate > 1 else ""))

    # Load annotations
    with open(obj_path.replace('.obj', '.json'), 'r') as f:
        annotations = json.load(f)

    # Load model
    model = keras.models.load_model(model_path)
    print(f"Model loaded")

    # Check GPU availability
    use_gpu = TORCH_AVAILABLE and torch.cuda.is_available() and config.get('gpu_acceleration', {}).get('enabled', True)

    if sample_rate > 1:
        np.random.seed(42)
        vertex_indices = np.sort(np.random.choice(n_verts, size=n_verts // sample_rate, replace=False))
    else:
        vertex_indices = np.arange(n_verts)

    if use_gpu:
        try:
            try:
                from tools.gpu_patch_generator_v2 import GPUPatchGeneratorV2 as GPUGenerator
            except ImportError:
                from tools.gpu_patch_generator import GPUPatchGenerator as GPUGenerator

            generator = GPUGenerator(
                mesh=mesh, annotations=annotations,
                patch_size=config['data_generation']['patch_size'],
                patch_radius=config['data_generation']['patch_radius'],
                precompute_neighbors=(sample_rate == 1)
            )

            patches = generator.generate_patches_batch(
                vertex_indices,
                batch_size=config.get('gpu_acceleration', {}).get('batch_size', 1024)
            )
            print(f"Generated {len(patches):,} patches (GPU)")

        except Exception as e:
            print(f"GPU failed ({e}), using CPU")
            use_gpu = False

    if not use_gpu:
        sampler = DatasetSampler(
            dataset_root=config['paths']['data_root'],
            patch_size=config['data_generation']['patch_size'],
            patch_radius=config['data_generation']['patch_radius']
        )
        sampler.current_mesh = mesh
        sampler.current_annotations = annotations
        sampler.kdtree = KDTree(mesh.vertices)

        patches = np.array([sampler._generate_image_patch(i) for i in vertex_indices])
        print(f"Generated {len(patches):,} patches (CPU)")

    # Run inference
    predictions_proba = model.predict(patches, verbose=1, batch_size=128)
    sampled_predictions = (predictions_proba >= 0.5).astype(int).flatten()
    sampled_confidences = predictions_proba.flatten()

    # Propagate if sampled
    if sample_rate > 1:
        predictions, confidences = _propagate_predictions(mesh, vertex_indices, sampled_predictions, sampled_confidences)
    else:
        predictions, confidences = sampled_predictions.tolist(), sampled_confidences.tolist()

    predictions = np.array(predictions)
    confidences = np.array(confidences)

    gingiva = np.sum(predictions == 0)
    teeth = np.sum(predictions == 1)
    print(f"Predictions: {gingiva:,} gingiva, {teeth:,} teeth ({time.time()-start:.1f}s)")

    # Color mesh
    colors = np.zeros((n_verts, 4), dtype=np.uint8)
    for i, (pred, conf) in enumerate(zip(predictions, confidences)):
        if pred == 1:
            colors[i] = [100, 100, int(100 + 155 * conf), 255]
        else:
            colors[i] = [int(100 + 155 * (1 - conf)), 100, 150, 255]
    mesh.visual.vertex_colors = colors

    return mesh, predictions, confidences


def main():
    parser = argparse.ArgumentParser(description='Visualize 3D dental scan predictions')
    parser.add_argument('--fast', action='store_true', help='Sample 1/25th of vertices')
    parser.add_argument('--sample-rate', type=int, default=1, help='Sample every Nth vertex')
    parser.add_argument('--scan', type=str, default=None, help='Pattern to match scan filename')
    args = parser.parse_args()

    sample_rate = 25 if args.fast else args.sample_rate

    config_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'config.yaml')
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    model_path = Path(config['paths']['model_dir']) / 'teeth_classifier.keras'
    if not model_path.exists():
        print(f"[ERROR] Model not found: {model_path}")
        return 1

    data_path = Path(config['paths']['data_root'])
    pattern = f"**/*{args.scan}*.obj" if args.scan else "**/*.obj"
    obj_files = list(data_path.glob(pattern))
    if not obj_files:
        print(f"[ERROR] No .obj files found")
        return 1

    obj_path = str(obj_files[0])
    print(f"Scan: {Path(obj_path).name}")

    mesh, predictions, confidences = predict_scan_3d(obj_path, str(model_path), config, sample_rate)

    # Save outputs
    output_dir = Path(config['paths']['results_dir'])
    output_dir.mkdir(parents=True, exist_ok=True)
    suffix = f"_{Path(obj_path).stem}" + ('_fast' if sample_rate > 1 else '')

    mesh_path = output_dir / f'prediction_mesh{suffix}.obj'
    mesh.export(str(mesh_path))
    print(f"Saved mesh: {mesh_path}")

    json_path = output_dir / f'prediction_data{suffix}.json'
    with open(json_path, 'w') as f:
        json.dump({
            'scan': Path(obj_path).name,
            'num_vertices': len(predictions),
            'sample_rate': sample_rate,
            'predictions': predictions.tolist(),
            'confidences': confidences.tolist(),
            'summary': {'gingiva': int(np.sum(predictions == 0)), 'teeth': int(np.sum(predictions == 1))}
        }, f, indent=2)
    print(f"Saved data: {json_path}")

    # Save screenshot
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')
    v = mesh.vertices
    ax.scatter(v[predictions==1, 0], v[predictions==1, 1], v[predictions==1, 2], c='blue', s=1, alpha=0.6, label='Tooth')
    ax.scatter(v[predictions==0, 0], v[predictions==0, 1], v[predictions==0, 2], c='pink', s=1, alpha=0.6, label='Gingiva')
    ax.set_xlabel('X'); ax.set_ylabel('Y'); ax.set_zlabel('Z')
    ax.legend()
    plt.tight_layout()
    img_path = output_dir / f'prediction_viz{suffix}.png'
    plt.savefig(img_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved image: {img_path}")

    # Try interactive viewer
    try:
        mesh.show()
    except ImportError:
        print("Interactive viewer not available (install pyglet<2)")

    return 0


if __name__ == "__main__":
    sys.exit(main())
