"""Quick viewer for saved prediction results."""

import os
import sys
import json
import yaml
import trimesh
import matplotlib.pyplot as plt
from pathlib import Path


def main():
    """Load and display saved prediction results."""
    config_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'config.yaml')
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    results_dir = Path(config['paths']['results_dir'])
    mesh_path = results_dir / '3d_prediction_mesh.obj'
    json_path = results_dir / '3d_prediction_data.json'
    screenshot_path = results_dir / '3d_prediction_visualization.png'

    if not mesh_path.exists():
        print(f"[ERROR] No results found at {mesh_path}")
        print("Run scripts/visualize.py first")
        return 1

    with open(json_path, 'r') as f:
        data = json.load(f)

    n = data['num_vertices']
    g, t = data['summary']['gingiva'], data['summary']['teeth']
    print(f"{data['scan']}: {n:,} vertices ({g:,} gingiva, {t:,} teeth)")

    mesh = trimesh.load(str(mesh_path))
    print(f"Loaded mesh: {len(mesh.vertices):,} vertices")

    if screenshot_path.exists():
        img = plt.imread(str(screenshot_path))
        plt.figure(figsize=(12, 10))
        plt.imshow(img)
        plt.axis('off')
        plt.tight_layout()
        plt.show()

    try:
        mesh.show()
    except ImportError:
        print("Interactive viewer not available (install pyglet<2)")
        print(f"View files: {mesh_path}, {screenshot_path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
