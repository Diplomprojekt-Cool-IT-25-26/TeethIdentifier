"""
Visualize 3D dental scan with model predictions.

Colors each vertex based on model predictions:
- Blue = Predicted as Tooth
- Pink = Predicted as Gingiva
"""

import yaml
import numpy as np
import trimesh
import matplotlib.pyplot as plt
from pathlib import Path
from tensorflow import keras
from scipy.spatial import KDTree
from dataset_sampler import DatasetSampler
import time

# GPU acceleration (optional import)
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


def _smooth_propagate_predictions(mesh, sampled_indices, sampled_predictions, sampled_confidences):
    """
    Propagate predictions from sampled vertices to all vertices using K-nearest neighbors.

    Simple and fast: just uses nearest neighbor lookup without slow graph smoothing.

    Args:
        mesh: Trimesh object
        sampled_indices: Indices of vertices that were predicted
        sampled_predictions: Predictions for sampled vertices
        sampled_confidences: Confidences for sampled vertices

    Returns:
        predictions: Array of predictions for all vertices
        confidences: Array of confidences for all vertices
    """
    # Simple nearest neighbor (fast and effective)
    kdtree = KDTree(mesh.vertices[sampled_indices])
    distances, nearest_indices = kdtree.query(mesh.vertices, k=1)

    predictions = sampled_predictions[nearest_indices]
    confidences = sampled_confidences[nearest_indices]

    return predictions.tolist(), confidences.tolist()


def predict_scan_3d(obj_path: str, model_path: str, config: dict, sample_rate: int = 1):
    """
    Run model predictions on all vertices of a 3D scan.

    Args:
        obj_path: Path to .obj file
        model_path: Path to trained model
        config: Configuration dictionary
        sample_rate: Process every Nth vertex (1=all, 25=1/25th for fast mode)

    Returns:
        mesh: Trimesh object with predictions
        predictions: Array of predictions (0=gingiva, 1=tooth)
        confidences: Array of prediction confidences
    """
    total_start = time.time()

    step_start = time.time()
    print(f"\n[1/5] Loading mesh and annotations...")
    mesh = trimesh.load(obj_path, process=False)
    print(f"  Vertices: {len(mesh.vertices):,}")

    if sample_rate > 1:
        # Set random seed for consistent sampling across runs
        np.random.seed(42)
        n_samples = len(mesh.vertices) // sample_rate
        print(f"  [FAST MODE] Random sampling ({n_samples:,} predictions, 1/{sample_rate}th of mesh)")

    # Load annotations (needed for patch generation)
    json_path = obj_path.replace('.obj', '.json')
    import json
    with open(json_path, 'r') as f:
        annotations = json.load(f)
    print(f"  Labels loaded from: {Path(json_path).name}")
    print(f"  [TIME]  Time: {time.time() - step_start:.2f}s")

    # Load model
    step_start = time.time()
    print(f"\n[2/5] Loading model...")
    model = keras.models.load_model(model_path)
    print(f"  [TIME]  Time: {time.time() - step_start:.2f}s")

    # Check if GPU acceleration is available
    use_gpu = (
        TORCH_AVAILABLE and
        torch.cuda.is_available() and
        config.get('gpu_acceleration', {}).get('enabled', True)
    )

    predictions = []
    confidences = []

    if use_gpu:
        try:
            # Try V2 (math approach) first, then V1 (rendering)
            try:
                from gpu_patch_generator_v2 import GPUPatchGeneratorV2
                print("\n[3/5] Initializing GPU patch generator...")
                GPUGenerator = GPUPatchGeneratorV2
            except ImportError:
                from gpu_patch_generator import GPUPatchGenerator
                print("\n[3/5] Initializing GPU patch generator...")
                GPUGenerator = GPUPatchGenerator

            # Initialize GPU generator
            step_start = time.time()
            # Skip neighbor precomputation in fast mode (not worth it for small samples)
            precompute = (sample_rate == 1)  # Only precompute for full scans
            generator = GPUGenerator(
                mesh=mesh,
                annotations=annotations,
                patch_size=config['data_generation']['patch_size'],
                patch_radius=config['data_generation']['patch_radius'],
                precompute_neighbors=precompute
            )
            print(f"  [TIME]  Time: {time.time() - step_start:.2f}s")

            # Generate patches (sampled or all vertices)
            print(f"\n[4/5] Generating patches...")
            step_start = time.time()
            if sample_rate > 1:
                # Random sampling (uniform distribution across mesh)
                n_samples = len(mesh.vertices) // sample_rate
                vertex_indices = np.random.choice(
                    len(mesh.vertices),
                    size=n_samples,
                    replace=False
                )
                vertex_indices.sort()  # Sort for better cache locality
            else:
                # All vertices
                vertex_indices = np.arange(len(mesh.vertices))

            all_patches = generator.generate_patches_batch(
                vertex_indices,
                batch_size=config.get('gpu_acceleration', {}).get('batch_size', 1024)
            )
            patch_time = time.time() - step_start
            print(f"  [TIME]  Patch generation time: {patch_time:.2f}s")

            # Run predictions on sampled patches
            print(f"\n[5/5] Running model inference on {len(vertex_indices):,} vertices...")
            step_start = time.time()
            predictions_proba = model.predict(all_patches, verbose=1, batch_size=128)
            sampled_predictions = (predictions_proba >= 0.5).astype(int).flatten()
            sampled_confidences = predictions_proba.flatten()
            inference_time = time.time() - step_start
            print(f"  [TIME]  Inference time: {inference_time:.2f}s")

            # If we sampled, propagate predictions to all vertices
            if sample_rate > 1:
                print(f"\n[Bonus] Propagating predictions to all vertices...")
                step_start = time.time()
                predictions, confidences = _smooth_propagate_predictions(
                    mesh, vertex_indices, sampled_predictions, sampled_confidences
                )
                print(f"  [TIME]  Propagation time: {time.time() - step_start:.2f}s")
            else:
                predictions = sampled_predictions.tolist()
                confidences = sampled_confidences.tolist()

            print("[OK] GPU prediction complete")

        except Exception as e:
            print(f"[WARNING] GPU failed: {e}")
            print("[CPU] Falling back to CPU implementation")
            use_gpu = False

    if not use_gpu:
        # CPU fallback (original implementation)
        print("[CPU] Using CPU patch generation")

        sampler = DatasetSampler(
            dataset_root=config['paths']['data_root'],
            patch_size=config['data_generation']['patch_size'],
            patch_radius=config['data_generation']['patch_radius']
        )

        sampler.current_mesh = mesh
        sampler.current_annotations = annotations
        sampler.kdtree = KDTree(mesh.vertices)

        # Determine which vertices to process
        if sample_rate > 1:
            # Random sampling (uniform distribution across mesh)
            n_samples = len(mesh.vertices) // sample_rate
            vertex_indices = np.random.choice(
                len(mesh.vertices),
                size=n_samples,
                replace=False
            )
            vertex_indices.sort()  # Sort for better cache locality
            num_to_process = len(vertex_indices)
        else:
            vertex_indices = np.arange(len(mesh.vertices))
            num_to_process = len(mesh.vertices)

        # Process in batches for efficiency
        batch_size = 500
        num_batches = (num_to_process + batch_size - 1) // batch_size

        sampled_predictions = []
        sampled_confidences = []

        for batch_idx in range(num_batches):
            start_idx = batch_idx * batch_size
            end_idx = min(start_idx + batch_size, num_to_process)

            # Generate image patches for this batch
            batch_patches = []
            for i in range(start_idx, end_idx):
                vertex_idx = vertex_indices[i]
                patch = sampler._generate_image_patch(vertex_idx)
                batch_patches.append(patch)

            # Run predictions
            batch_patches = np.array(batch_patches)
            batch_probs = model.predict(batch_patches, verbose=0, batch_size=128)

            # Store results
            for prob in batch_probs:
                sampled_confidences.append(float(prob[0]))
                sampled_predictions.append(1 if prob[0] >= 0.5 else 0)

            if (batch_idx + 1) % 10 == 0:
                print(f"  Processed {end_idx:,}/{num_to_process:,} vertices...")

        # If we sampled, propagate predictions to all vertices
        if sample_rate > 1:
            print(f"[CPU] Propagating predictions to all vertices...")
            sampled_predictions = np.array(sampled_predictions)
            sampled_confidences = np.array(sampled_confidences)

            predictions, confidences = _smooth_propagate_predictions(
                mesh, vertex_indices, sampled_predictions, sampled_confidences
            )
            print(f"[OK] Predictions propagated")
        else:
            predictions = sampled_predictions
            confidences = sampled_confidences

    predictions = np.array(predictions)
    confidences = np.array(confidences)

    total_time = time.time() - total_start

    print(f"\n" + "=" * 70)
    print(f"[TIME]  TOTAL TIME: {total_time:.2f}s ({total_time/60:.2f} minutes)")
    print(f"=" * 70)

    print(f"\nPrediction Summary:")
    print(f"  Gingiva: {np.sum(predictions == 0):,} vertices ({np.sum(predictions == 0)/len(predictions)*100:.1f}%)")
    print(f"  Teeth: {np.sum(predictions == 1):,} vertices ({np.sum(predictions == 1)/len(predictions)*100:.1f}%)")

    # Color the mesh based on predictions
    colors = np.zeros((len(mesh.vertices), 4), dtype=np.uint8)

    for i, (pred, conf) in enumerate(zip(predictions, confidences)):
        if pred == 1:  # Tooth
            # Blue, intensity based on confidence
            intensity = int(100 + 155 * conf)
            colors[i] = [100, 100, intensity, 255]
        else:  # Gingiva
            # Pink, intensity based on confidence
            intensity = int(100 + 155 * (1 - conf))
            colors[i] = [intensity, 100, 150, 255]

    mesh.visual.vertex_colors = colors

    return mesh, predictions, confidences


def main():
    """Main visualization function."""
    import argparse

    parser = argparse.ArgumentParser(description='Visualize 3D dental scan predictions')
    parser.add_argument('--fast', action='store_true',
                        help='Fast mode: sample every 25th vertex (~15 sec instead of 5 min)')
    parser.add_argument('--sample-rate', type=int, default=1,
                        help='Sample every Nth vertex (1=all, 25=fast mode)')
    args = parser.parse_args()

    # Determine sample rate
    if args.fast:
        sample_rate = 25
    else:
        sample_rate = args.sample_rate

    print("=" * 70)
    print("3D Dental Scan Prediction Visualization")
    if sample_rate > 1:
        print(f"[FAST MODE] Sampling 1/{sample_rate}th of vertices")
    print("=" * 70)

    # Load config
    with open('config.yaml', 'r') as f:
        config = yaml.safe_load(f)

    model_path = Path(config['paths']['model_dir']) / 'teeth_classifier.keras'

    if not model_path.exists():
        print(f"\n[ERROR] Model not found: {model_path}")
        print("Train the model first with: train_gpu.bat")
        return 1

    # Find a test scan to visualize
    data_path = Path(config['paths']['data_root'])
    obj_files = list(data_path.glob("**/*.obj"))

    if not obj_files:
        print(f"\n[ERROR] No .obj files found in {data_path}")
        return 1

    # Use the first scan
    obj_path = str(obj_files[0])

    print(f"\nUsing scan: {Path(obj_path).name}")
    print("\nColors:")
    print("  Blue = Tooth (predicted)")
    print("  Pink = Gingiva (predicted)")
    print()

    # Run predictions
    mesh, predictions, confidences = predict_scan_3d(obj_path, str(model_path), config, sample_rate=sample_rate)

    # Save results BEFORE attempting interactive viewer
    output_dir = Path(config['paths']['results_dir'])
    output_dir.mkdir(parents=True, exist_ok=True)

    # Add suffix for fast mode
    suffix = '_fast' if sample_rate > 1 else ''

    # Save colored mesh to OBJ file
    mesh_output_path = output_dir / f'3d_prediction_mesh{suffix}.obj'
    print(f"\nSaving colored mesh to: {mesh_output_path}")
    mesh.export(str(mesh_output_path))
    print(f"[OK] Mesh saved! (open with MeshLab, Blender, etc.)")

    # Save predictions to JSON
    json_output_path = output_dir / f'3d_prediction_data{suffix}.json'
    print(f"\nSaving predictions to: {json_output_path}")
    import json
    prediction_data = {
        'scan': Path(obj_path).name,
        'num_vertices': len(predictions),
        'sample_rate': sample_rate,
        'sampled': sample_rate > 1,
        'predictions': predictions.tolist(),
        'confidences': confidences.tolist(),
        'summary': {
            'gingiva': int(np.sum(predictions == 0)),
            'teeth': int(np.sum(predictions == 1))
        }
    }
    with open(json_output_path, 'w') as f:
        json.dump(prediction_data, f, indent=2)
    print(f"[OK] Predictions saved!")

    # Create matplotlib screenshot
    output_path = output_dir / f'3d_prediction_visualization{suffix}.png'
    print(f"\nSaving screenshot to: {output_path}")

    # Create a screenshot using matplotlib
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')

    # Plot vertices colored by prediction
    vertices = mesh.vertices
    tooth_mask = predictions == 1
    gingiva_mask = predictions == 0

    ax.scatter(vertices[tooth_mask, 0], vertices[tooth_mask, 1], vertices[tooth_mask, 2],
              c='blue', s=1, alpha=0.6, label='Tooth')
    ax.scatter(vertices[gingiva_mask, 0], vertices[gingiva_mask, 1], vertices[gingiva_mask, 2],
              c='pink', s=1, alpha=0.6, label='Gingiva')

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    title = f'3D Prediction Visualization\n{Path(obj_path).name}'
    if sample_rate > 1:
        title += f'\n[Fast Mode: 1/{sample_rate}th sampling]'
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.legend()

    # Equal aspect ratio
    max_range = np.array([vertices[:, 0].max()-vertices[:, 0].min(),
                         vertices[:, 1].max()-vertices[:, 1].min(),
                         vertices[:, 2].max()-vertices[:, 2].min()]).max() / 2.0

    mid_x = (vertices[:, 0].max()+vertices[:, 0].min()) * 0.5
    mid_y = (vertices[:, 1].max()+vertices[:, 1].min()) * 0.5
    mid_z = (vertices[:, 2].max()+vertices[:, 2].min()) * 0.5
    ax.set_xlim(mid_x - max_range, mid_x + max_range)
    ax.set_ylim(mid_y - max_range, mid_y + max_range)
    ax.set_zlim(mid_z - max_range, mid_z + max_range)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"[OK] Screenshot saved!")

    # Close matplotlib figure without displaying
    plt.close()

    # Try to open interactive 3D viewer (optional)
    print("\n" + "=" * 70)
    print("Attempting to open interactive 3D viewer...")
    print("=" * 70)
    print("  - Rotate: Left mouse drag")
    print("  - Pan: Right mouse drag")
    print("  - Zoom: Scroll wheel")
    print("  - Close window to exit")
    print()

    try:
        mesh.show()
    except ImportError as e:
        print(f"[INFO] Interactive viewer not available: {e}")
        print(f"[INFO] To enable viewer, run: pip install \"pyglet<2\"")
        print(f"[INFO] You can still view the saved files:")
        print(f"       - OBJ mesh: {mesh_output_path}")
        print(f"       - Screenshot: {output_path}")

    print("\n" + "=" * 70)
    print("Visualization Complete!")
    print("=" * 70)
    print(f"\nSaved files:")
    print(f"  - Colored mesh (OBJ): {mesh_output_path}")
    print(f"  - Screenshot (PNG): {output_path}")
    print(f"  - Predictions (JSON): {json_output_path}")

    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
