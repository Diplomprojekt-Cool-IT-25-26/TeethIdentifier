"""
Generate individual patch PNG images for the DentAI demo frontend v2.

Run from the TeethIdentifier root directory:
    python demo_frontend_v2/generate_patches.py

Output: demo_frontend_v2/assets/patches/
    tooth_0.png ... tooth_4.png
    gingiva_0.png ... gingiva_4.png
    boundary_0.png ... boundary_4.png
"""

import os
import sys
import json
import numpy as np
import trimesh
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import yaml
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

config_path = PROJECT_ROOT / 'config.yaml'
with open(config_path) as f:
    config = yaml.safe_load(f)

data_root = Path(config['paths']['data_root'])
output_dir = Path(__file__).parent / 'assets' / 'patches'
output_dir.mkdir(parents=True, exist_ok=True)

# Find annotated OBJ files — pick one from data_part_5 or 6 for variety
candidates = sorted([f for f in data_root.glob('**/*.obj') if f.with_suffix('.json').exists()])
if not candidates:
    print("ERROR: No annotated .obj files found.")
    sys.exit(1)

# Use a consistent scan (seed 42 ensures same result every run)
rng = np.random.default_rng(42)
obj_path = candidates[rng.integers(len(candidates))]
print(f"Loading {obj_path.name} ...")

mesh = trimesh.load(str(obj_path), process=False)
with open(str(obj_path.with_suffix('.json'))) as f:
    ann = json.load(f)

labels = np.array(ann['labels'])
binary = np.where(labels == 0, 0, 1)  # 0=gingiva, 1=tooth

from scipy.spatial import KDTree
kdtree = KDTree(mesh.vertices)
BOUNDARY_RADIUS = config['data_generation']['boundary_radius']

print("Sampling vertex pools (2000 candidates)...")
sample_pool = rng.choice(len(mesh.vertices), min(2000, len(mesh.vertices)), replace=False)
boundary_pool, tooth_pool, gingiva_pool = [], [], []

for i in sample_pool:
    nb = kdtree.query_ball_point(mesh.vertices[i], BOUNDARY_RADIUS)
    nb_labels = binary[np.array(nb)]
    is_boundary = nb_labels.min() != nb_labels.max()
    if is_boundary:
        boundary_pool.append(i)
    elif binary[i] == 1:
        tooth_pool.append(i)
    else:
        gingiva_pool.append(i)

N_EACH = 5
pick_tooth    = rng.choice(tooth_pool,    min(N_EACH, len(tooth_pool)),    replace=False)
pick_gingiva  = rng.choice(gingiva_pool,  min(N_EACH, len(gingiva_pool)),  replace=False)
pick_boundary = rng.choice(boundary_pool, min(N_EACH, len(boundary_pool)), replace=False)
all_indices   = np.concatenate([pick_tooth, pick_gingiva, pick_boundary]).astype(np.int64)

print(f"Generating {len(all_indices)} patches (5 tooth / 5 gingiva / 5 boundary)...")

try:
    import torch
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    from tools.gpu_patch_generator_v3 import GPUPatchGeneratorV3
    gen = GPUPatchGeneratorV3(
        mesh=mesh,
        patch_size=config['data_generation']['patch_size'],
        patch_radius=config['data_generation']['patch_radius'],
        device=device
    )
except Exception as e:
    print(f"GPU patch generator failed ({e}), falling back to CPU DatasetSampler")
    from tools.dataset_sampler import DatasetSampler
    sampler = DatasetSampler(
        dataset_root=str(obj_path.parent),
        patch_size=config['data_generation']['patch_size'],
        patch_radius=config['data_generation']['patch_radius']
    )
    sampler.current_mesh = mesh
    sampler.kdtree = kdtree
    patches = np.array([sampler._generate_image_patch(int(i)) for i in all_indices])
    gen = None

if gen is not None:
    patches = gen.generate_patches_batch(all_indices)

print(f"Patches shape: {patches.shape}")

# Save individual PNG files
categories = ['tooth'] * N_EACH + ['gingiva'] * N_EACH + ['boundary'] * N_EACH
counts = {'tooth': 0, 'gingiva': 0, 'boundary': 0}

for patch, cat in zip(patches, categories):
    idx = counts[cat]
    out_path = output_dir / f'{cat}_{idx}.png'
    patch_u8 = (np.clip(patch, 0, 1) * 255).astype(np.uint8)
    plt.imsave(str(out_path), patch_u8)
    print(f"  Saved: {out_path.name}")
    counts[cat] += 1

print(f"\nDone — {len(patches)} patch images saved to {output_dir}")
