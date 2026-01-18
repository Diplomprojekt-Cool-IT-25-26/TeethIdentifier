"""
Quick viewer for saved prediction results.

Loads previously saved predictions and displays them without re-processing.
Much faster than running visualize_predictions.py again.

Usage:
    python view_saved_results.py
"""

import json
import numpy as np
import trimesh
import matplotlib.pyplot as plt
from pathlib import Path
import yaml


def main():
    """Load and display saved prediction results."""

    print("=" * 70)
    print("View Saved Prediction Results")
    print("=" * 70)
    print()

    # Load config
    with open('config.yaml', 'r') as f:
        config = yaml.safe_load(f)

    results_dir = Path(config['paths']['results_dir'])

    # Check if saved results exist
    mesh_path = results_dir / '3d_prediction_mesh.obj'
    json_path = results_dir / '3d_prediction_data.json'
    screenshot_path = results_dir / '3d_prediction_visualization.png'

    if not mesh_path.exists():
        print("[ERROR] No saved results found!")
        print(f"Expected to find: {mesh_path}")
        print()
        print("Run visualize_predictions.py first to generate results.")
        return 1

    # Load prediction data
    print(f"Loading predictions from: {json_path}")
    with open(json_path, 'r') as f:
        data = json.load(f)

    print(f"  Scan: {data['scan']}")
    print(f"  Vertices: {data['num_vertices']:,}")
    print(f"  Gingiva: {data['summary']['gingiva']:,} vertices ({data['summary']['gingiva']/data['num_vertices']*100:.1f}%)")
    print(f"  Teeth: {data['summary']['teeth']:,} vertices ({data['summary']['teeth']/data['num_vertices']*100:.1f}%)")

    # Load colored mesh
    print(f"\nLoading colored mesh from: {mesh_path}")
    mesh = trimesh.load(str(mesh_path))
    print(f"  Loaded {len(mesh.vertices):,} vertices")

    # Display screenshot
    if screenshot_path.exists():
        print(f"\nDisplaying screenshot: {screenshot_path}")
        img = plt.imread(str(screenshot_path))
        plt.figure(figsize=(12, 10))
        plt.imshow(img)
        plt.axis('off')
        plt.title('3D Prediction Visualization (Saved)', fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.show()

    # Try to open interactive 3D viewer
    print("\n" + "=" * 70)
    print("Attempting to open interactive 3D viewer...")
    print("=" * 70)

    try:
        mesh.show()
        print("[OK] Viewer opened successfully!")
    except ImportError as e:
        print(f"[INFO] Interactive viewer not available: {e}")
        print(f"[INFO] To enable viewer, run: pip install \"pyglet<2\"")
        print()
        print("Alternative viewers:")
        print(f"  - Open {mesh_path} in MeshLab")
        print(f"  - Open {mesh_path} in Blender")
        print(f"  - View {screenshot_path} as screenshot")

    print("\n" + "=" * 70)
    print("Done!")
    print("=" * 70)

    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
