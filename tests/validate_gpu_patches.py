"""
Validation Suite for GPU Patch Generation

This script validates that GPU-generated patches match CPU-generated patches
and benchmarks performance improvements.

Usage:
    python tests/validate_gpu_patches.py

Tests:
1. Patch accuracy: GPU patches match CPU (MSE < threshold)
2. Performance benchmark: Measure speedup
3. Full scan estimate: Extrapolate to 93k vertices
"""

import sys
import os
import time
import numpy as np
import trimesh
from scipy.spatial import KDTree
import json
import yaml
import matplotlib.pyplot as plt
from pathlib import Path

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dataset_sampler import DatasetSampler

# Try to import GPU generators (V2 first, then V1)
GPU_V2_AVAILABLE = False
GPU_V1_AVAILABLE = False

try:
    from gpu_patch_generator_v2 import GPUPatchGeneratorV2
    GPU_V2_AVAILABLE = True
    print("[INFO] GPU Math generator (V2) available")
except ImportError as e:
    print(f"[INFO] GPU Math generator (V2) not available: {e}")

try:
    from gpu_patch_generator import GPUPatchGenerator
    GPU_V1_AVAILABLE = True
    print("[INFO] GPU Rendering generator (V1) available")
except ImportError as e:
    print(f"[INFO] GPU Rendering generator (V1) not available: {e}")

GPU_AVAILABLE = GPU_V2_AVAILABLE or GPU_V1_AVAILABLE


def test_patch_accuracy(num_samples=100):
    """
    Verify GPU patches match CPU patches pixel-by-pixel.

    Args:
        num_samples: Number of random vertices to test

    Returns:
        bool: True if validation passed, False otherwise
    """
    print("=" * 70)
    print("GPU vs CPU Patch Validation")
    print("=" * 70)
    print()

    if not GPU_AVAILABLE:
        print("[ERROR] GPU patch generator not available")
        return False

    # Load config
    config_path = Path(__file__).parent.parent / 'config.yaml'
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    # Load a test scan
    data_path = config['paths']['data_root']
    obj_path = f"{data_path}/data_part_1/lower/O52P1SZT/O52P1SZT_lower.obj"
    json_path = obj_path.replace('.obj', '.json')

    if not os.path.exists(obj_path):
        print(f"[ERROR] Test scan not found: {obj_path}")
        return False

    mesh = trimesh.load(obj_path, process=False)
    with open(json_path, 'r') as f:
        annotations = json.load(f)

    print(f"Loaded mesh: {len(mesh.vertices):,} vertices")

    # Select random test vertices
    np.random.seed(42)
    test_vertices = np.random.choice(len(mesh.vertices), num_samples, replace=False)
    print(f"Testing {num_samples} random vertices\n")

    # Generate CPU patches
    print("[CPU] Generating patches...")
    cpu_start = time.time()

    cpu_sampler = DatasetSampler(
        dataset_root=data_path,
        patch_size=config['data_generation']['patch_size'],
        patch_radius=config['data_generation']['patch_radius']
    )
    cpu_sampler.current_mesh = mesh
    cpu_sampler.current_annotations = annotations
    cpu_sampler.kdtree = KDTree(mesh.vertices)

    cpu_patches = []
    for idx in test_vertices:
        cpu_patches.append(cpu_sampler._generate_image_patch(idx))
    cpu_patches = np.array(cpu_patches)

    cpu_time = time.time() - cpu_start
    print(f"[CPU] Generated {len(cpu_patches)} patches in {cpu_time:.2f}s")

    # Generate GPU patches
    print("[GPU] Generating patches...")
    gpu_start = time.time()

    # Use V2 (math approach) if available, otherwise V1 (rendering)
    if GPU_V2_AVAILABLE:
        print("[GPU] Using V2 (math approach)")
        gpu_generator = GPUPatchGeneratorV2(
            mesh=mesh,
            annotations=annotations,
            patch_size=config['data_generation']['patch_size'],
            patch_radius=config['data_generation']['patch_radius']
        )
    else:
        print("[GPU] Using V1 (rendering approach)")
        gpu_generator = GPUPatchGenerator(
            mesh=mesh,
            annotations=annotations,
            patch_size=config['data_generation']['patch_size'],
            patch_radius=config['data_generation']['patch_radius']
        )

    gpu_patches = gpu_generator.generate_patches_batch(test_vertices, batch_size=32)

    gpu_time = time.time() - gpu_start
    print(f"[GPU] Generated {len(gpu_patches)} patches in {gpu_time:.2f}s")
    print(f"[GPU] Speedup: {cpu_time/gpu_time:.1f}×\n")

    # Compare patches
    print("=" * 70)
    print("Comparison Results")
    print("=" * 70)

    mse_values = []

    for i, (cpu_patch, gpu_patch) in enumerate(zip(cpu_patches, gpu_patches)):
        mse = np.mean((cpu_patch.astype(float) - gpu_patch.astype(float)) ** 2)
        mse_values.append(mse)

        if i < 3:  # Show first 3
            print(f"Patch {i}: MSE = {mse:.4f}")

    avg_mse = np.mean(mse_values)
    max_mse = np.max(mse_values)
    min_mse = np.min(mse_values)

    print()
    print(f"Average MSE: {avg_mse:.4f}")
    print(f"Min MSE:     {min_mse:.4f}")
    print(f"Max MSE:     {max_mse:.4f}")

    # Save comparison images
    results_dir = Path(__file__).parent.parent / 'ml_outputs' / 'results'
    results_dir.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(3, 3, figsize=(12, 12))
    fig.suptitle('GPU vs CPU Patch Comparison', fontsize=16)

    for i in range(min(3, len(cpu_patches))):
        # CPU patch
        axes[i, 0].imshow(cpu_patches[i])
        axes[i, 0].set_title(f"CPU Patch {i}")
        axes[i, 0].axis('off')

        # GPU patch
        axes[i, 1].imshow(gpu_patches[i])
        axes[i, 1].set_title(f"GPU Patch {i}")
        axes[i, 1].axis('off')

        # Difference (amplified for visibility)
        diff = np.abs(cpu_patches[i].astype(float) - gpu_patches[i].astype(float))
        axes[i, 2].imshow(diff.astype(np.uint8) * 5)  # Amplify by 5×
        axes[i, 2].set_title(f"Difference (MSE={mse_values[i]:.2f})")
        axes[i, 2].axis('off')

    plt.tight_layout()
    comparison_path = results_dir / 'gpu_cpu_comparison.png'
    plt.savefig(comparison_path, dpi=150, bbox_inches='tight')
    print(f"\nComparison saved to: {comparison_path}")

    # Validation threshold
    THRESHOLD = 20.0  # Allow minor rounding differences

    print()
    print("=" * 70)
    if avg_mse < THRESHOLD:
        print(f"[PASS] GPU patches match CPU (avg MSE {avg_mse:.2f} < {THRESHOLD})")
        print("=" * 70)
        return True
    else:
        print(f"[FAIL] GPU patches differ from CPU (avg MSE {avg_mse:.2f} >= {THRESHOLD})")
        print("=" * 70)
        return False


def benchmark_performance(num_vertices=5000):
    """
    Benchmark GPU vs CPU speed and extrapolate to full scan.

    Args:
        num_vertices: Number of vertices to benchmark (default: 5000)

    Returns:
        dict: Benchmark results
    """
    print()
    print("=" * 70)
    print("Performance Benchmark")
    print("=" * 70)
    print()

    if not GPU_AVAILABLE:
        print("[ERROR] GPU patch generator not available")
        return None

    # Load config and mesh
    config_path = Path(__file__).parent.parent / 'config.yaml'
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    data_path = config['paths']['data_root']
    obj_path = f"{data_path}/data_part_1/lower/O52P1SZT/O52P1SZT_lower.obj"
    json_path = obj_path.replace('.obj', '.json')

    mesh = trimesh.load(obj_path, process=False)
    with open(json_path, 'r') as f:
        annotations = json.load(f)

    test_vertices = np.arange(min(num_vertices, len(mesh.vertices)))
    print(f"Benchmarking with {len(test_vertices):,} vertices\n")

    # CPU benchmark
    print(f"[CPU] Processing {len(test_vertices):,} vertices...")
    cpu_sampler = DatasetSampler(
        dataset_root=data_path,
        patch_size=config['data_generation']['patch_size'],
        patch_radius=config['data_generation']['patch_radius']
    )
    cpu_sampler.current_mesh = mesh
    cpu_sampler.current_annotations = annotations
    cpu_sampler.kdtree = KDTree(mesh.vertices)

    cpu_start = time.time()
    cpu_patches = [cpu_sampler._generate_image_patch(i) for i in test_vertices]
    cpu_time = time.time() - cpu_start

    # GPU benchmark
    print(f"[GPU] Processing {len(test_vertices):,} vertices...")

    # Use V2 (math approach) if available, otherwise V1 (rendering)
    if GPU_V2_AVAILABLE:
        print("[GPU] Using V2 (math approach)")
        gpu_generator = GPUPatchGeneratorV2(
            mesh,
            annotations,
            patch_size=config['data_generation']['patch_size'],
            patch_radius=config['data_generation']['patch_radius']
        )
    else:
        print("[GPU] Using V1 (rendering approach)")
        gpu_generator = GPUPatchGenerator(
            mesh,
            annotations,
            patch_size=config['data_generation']['patch_size'],
            patch_radius=config['data_generation']['patch_radius']
        )

    gpu_start = time.time()
    gpu_patches = gpu_generator.generate_patches_batch(test_vertices, batch_size=1024)
    gpu_time = time.time() - gpu_start

    # Results
    print()
    print("=" * 70)
    print("Results")
    print("=" * 70)
    print(f"CPU: {cpu_time:.2f}s ({cpu_time/len(test_vertices)*1000:.2f}ms per vertex)")
    print(f"GPU: {gpu_time:.2f}s ({gpu_time/len(test_vertices)*1000:.2f}ms per vertex)")
    print(f"Speedup: {cpu_time/gpu_time:.1f}×")

    # Extrapolate to full scan
    full_scan_vertices = 93288
    cpu_full = (cpu_time / len(test_vertices)) * full_scan_vertices
    gpu_full = (gpu_time / len(test_vertices)) * full_scan_vertices

    print()
    print(f"Full Scan Estimate ({full_scan_vertices:,} vertices):")
    print(f"  CPU: {cpu_full/60:.1f} minutes ({cpu_full:.0f} seconds)")
    print(f"  GPU: {gpu_full:.1f} seconds")
    print(f"  Time saved: {(cpu_full - gpu_full)/60:.1f} minutes")

    print()
    if gpu_full < 30:
        print(f"[OK] GPU meets 30-second target ({gpu_full:.1f}s < 30s)")
    else:
        print(f"[WARNING] GPU exceeds 30-second target ({gpu_full:.1f}s >= 30s)")

    print("=" * 70)

    return {
        'cpu_time': cpu_time,
        'gpu_time': gpu_time,
        'speedup': cpu_time / gpu_time,
        'cpu_full_estimate': cpu_full,
        'gpu_full_estimate': gpu_full
    }


def main():
    """Run all validation tests."""
    print()
    print("=" * 70)
    print("TeethIdentifier GPU Patch Generation Validation Suite")
    print("=" * 70)
    print()

    if not GPU_AVAILABLE:
        print("[ERROR] GPU patch generator not available.")
        print("Make sure PyTorch3D is installed:")
        print("  pip install torch==1.12.0+cu113 --extra-index-url https://download.pytorch.org/whl/cu113")
        print("  pip install fvcore iopath")
        print("  pip install pytorch3d -f https://dl.fbaipublicfiles.com/pytorch3d/packaging/wheels/py310_cu113_pyt112/download.html")
        print("  pip install kornia==0.6.8")
        return

    # Test 1: Patch accuracy
    passed = test_patch_accuracy(num_samples=100)

    if not passed:
        print()
        print("[WARNING] Patch accuracy test failed.")
        print("GPU patches differ significantly from CPU patches.")
        print("Consider disabling GPU acceleration in config.yaml")
        return

    # Test 2: Performance benchmark
    benchmark_performance(num_vertices=5000)

    print()
    print("=" * 70)
    print("Validation Complete")
    print("=" * 70)
    print()
    print("Next steps:")
    print("1. GPU acceleration is validated and ready to use")
    print("2. Run: venv\\Scripts\\python.exe visualize_predictions.py")
    print("3. Expected: ~10-15 seconds for 93k vertex scan (vs 30 minutes CPU)")
    print()


if __name__ == "__main__":
    main()
