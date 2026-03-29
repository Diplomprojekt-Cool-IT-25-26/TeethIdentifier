"""Show sample V3 patches so you can see what the model actually sees.

Picks a few vertices: some clearly on teeth, some clearly on gingiva,
some on the boundary, and saves a grid image to ml_outputs/results/sample_patches.png
"""

import os
import sys
import json
import numpy as np
import trimesh
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import yaml

config_path = Path(__file__).parent.parent / 'config.yaml'
with open(config_path) as f:
    config = yaml.safe_load(f)

data_root = Path(config['paths']['data_root'])
candidates = [f for f in data_root.glob('**/*.obj') if f.with_suffix('.json').exists()]
if not candidates:
    print("No annotated .obj files found.")
    sys.exit(1)

obj_path = candidates[0]
print(f"Loading {obj_path.name} ...")

mesh = trimesh.load(str(obj_path), process=False)
with open(str(obj_path.with_suffix('.json'))) as f:
    ann = json.load(f)

labels = np.array(ann['labels'])
binary = np.where(labels == 0, 0, 1)  # 0=gingiva, 1=tooth

# Find boundary vertices (at least one neighbour has a different label)
from scipy.spatial import KDTree
kdtree = KDTree(mesh.vertices)
BOUNDARY_RADIUS = config['data_generation']['boundary_radius']

print("Finding boundary vertices (sampling 2000 to check)...")
sample_pool = np.random.choice(len(mesh.vertices), 2000, replace=False)
boundary_pool = []
tooth_pool = []
gingiva_pool = []

for i in sample_pool:
    nb = kdtree.query_ball_point(mesh.vertices[i], BOUNDARY_RADIUS)
    nb_labels = binary[nb]
    is_boundary = (nb_labels.min() != nb_labels.max())
    if is_boundary:
        boundary_pool.append(i)
    elif binary[i] == 1:
        tooth_pool.append(i)
    else:
        gingiva_pool.append(i)

N_EACH = 5
rng = np.random.default_rng(42)
pick_tooth    = rng.choice(tooth_pool,    min(N_EACH, len(tooth_pool)),    replace=False)
pick_gingiva  = rng.choice(gingiva_pool,  min(N_EACH, len(gingiva_pool)),  replace=False)
pick_boundary = rng.choice(boundary_pool, min(N_EACH, len(boundary_pool)), replace=False)

all_indices = np.concatenate([pick_tooth, pick_gingiva, pick_boundary]).astype(int)
print(f"Generating {len(all_indices)} patches (5 tooth / 5 gingiva / 5 boundary)...")

from tools.gpu_patch_generator_v3 import GPUPatchGeneratorV3
gen = GPUPatchGeneratorV3(
    mesh=mesh,
    patch_size=config['data_generation']['patch_size'],
    patch_radius=config['data_generation']['patch_radius'],
)
patches = gen.generate_patches_batch(np.array(all_indices, dtype=np.int64))

# patches shape: (N, 100, 100, 3)  — R=depth, G=shading, B=curvature
rows = 3
cols = N_EACH
fig, axes = plt.subplots(rows, cols, figsize=(cols * 2.5, rows * 2.8))
fig.suptitle('V3 patches — what the model sees\nR=depth, G=shading, B=curvature (blue tint = curved = tooth cusp)',
             fontsize=10)

row_labels = ['Tooth (clear)', 'Gingiva (clear)', 'Boundary']
for row, (label_text, group_patches) in enumerate(zip(
        row_labels, [patches[:N_EACH], patches[N_EACH:2*N_EACH], patches[2*N_EACH:]])):
    for col in range(len(group_patches)):
        ax = axes[row][col]
        ax.imshow(group_patches[col])
        ax.axis('off')
        if col == 0:
            ax.set_ylabel(label_text, fontsize=9, rotation=90, labelpad=4)
            ax.yaxis.label.set_visible(True)
            ax.axis('on')
            ax.set_xticks([])
            ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_visible(False)

plt.tight_layout()
output_path = Path(config['paths']['results_dir']) / 'sample_patches.png'
output_path.parent.mkdir(parents=True, exist_ok=True)
plt.savefig(str(output_path), dpi=150, bbox_inches='tight')
print(f"Saved: {output_path}")
plt.show()
