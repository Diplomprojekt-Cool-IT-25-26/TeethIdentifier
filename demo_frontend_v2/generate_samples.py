"""
Generate classified OBJ files for both demo scans:
  Scan A (original_scan.obj):
    classified_raw.obj             — raw model output, no post-processing
    classified_postprocessed.obj   — same run + post-processing
  Scan B (original_scan_2.obj):
    classified_raw_scan2.obj       — raw model output
    classified_postprocessed_scan2.obj — same run + post-processing

Colors (v x y z r g b):
  Teeth:   R=100/255  G=100/255  B=254/255
  Gingiva: R=254/255  G=100/255  B=150/255

Run from TeethIdentifier root:
    venv\\Scripts\\python.exe demo_frontend_v2/generate_samples.py
"""

import os, sys, time, yaml
import numpy as np
import trimesh
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# torch MUST be imported before TensorFlow on Windows to avoid DLL conflicts
import torch

config_path = PROJECT_ROOT / 'config.yaml'
with open(config_path) as f:
    config = yaml.safe_load(f)

MODEL_PATH  = PROJECT_ROOT / config['paths']['model_dir'] / 'teeth_classifier.keras'
OUT_DIR     = PROJECT_ROOT / 'demo_frontend' / 'samples'

TEETH_RGB   = np.array([100/255, 100/255, 254/255])
GINGIVA_RGB = np.array([254/255, 100/255, 150/255])

def save_colored_obj(mesh, predictions, out_path):
    """Write OBJ with embedded vertex colors: v x y z r g b"""
    print(f"  Saving {out_path.name} ...")
    with open(out_path, 'w') as f:
        for i, v in enumerate(mesh.vertices):
            c = TEETH_RGB if predictions[i] == 1 else GINGIVA_RGB
            f.write(f"v {v[0]:.8f} {v[1]:.8f} {v[2]:.8f} {c[0]:.8f} {c[1]:.8f} {c[2]:.8f}\n")
        for face in mesh.faces:
            f.write(f"f {face[0]+1} {face[1]+1} {face[2]+1}\n")
    teeth   = int(np.sum(predictions == 1))
    gingiva = int(np.sum(predictions == 0))
    total   = teeth + gingiva
    print(f"  Teeth: {teeth:,} ({teeth/total*100:.1f}%)  Gingiva: {gingiva:,} ({gingiva/total*100:.1f}%)")

def process_scan(obj_path, raw_out, pp_out, model, device, config, label):
    print(f"\n{'='*60}")
    print(f"Processing {label}: {obj_path.name}")
    print(f"{'='*60}")

    mesh = trimesh.load(str(obj_path), process=False)
    print(f"  {len(mesh.vertices):,} vertices")

    from tools.gpu_patch_generator_v3 import GPUPatchGeneratorV3
    print(f"Generating patches on {device} ...")
    t0 = time.time()
    gen = GPUPatchGeneratorV3(
        mesh=mesh,
        patch_size=config['data_generation']['patch_size'],
        patch_radius=config['data_generation']['patch_radius'],
        device=device
    )
    vertex_indices = np.arange(len(mesh.vertices), dtype=np.int64)
    patches = gen.generate_patches_batch(
        vertex_indices,
        batch_size=config['gpu_acceleration']['batch_size']
    )
    print(f"  {len(patches):,} patches in {time.time()-t0:.1f}s")

    print("Running inference ...")
    t1 = time.time()
    proba = model.predict(patches, batch_size=config['inference']['batch_size'], verbose=1)
    raw_predictions = (proba.flatten() >= config['inference']['threshold']).astype(int)
    print(f"  Done in {time.time()-t1:.1f}s")

    print(f"\n[1/2] Raw output")
    save_colored_obj(mesh, raw_predictions, raw_out)

    print(f"\n[2/2] Post-processing ...")
    from tools.mesh_morphology import postprocess
    pp_predictions = postprocess(
        raw_predictions, mesh,
        open_iters=2,
        close_iters=1,
        erode_gingiva_iters=1,
        majority_vote_iters=2,
        open_gingiva_iters=0,   # disabled — too aggressive on gingiva regions
        reconstruct_iters=0
    )
    changed = int(np.sum(pp_predictions != raw_predictions))
    print(f"  {changed:,} vertices changed ({changed/len(raw_predictions)*100:.1f}%)")
    save_colored_obj(mesh, pp_predictions, pp_out)

# ── Load model once ───────────────────────────────────────────────────────
import tensorflow as tf
for gpu in tf.config.experimental.list_physical_devices('GPU'):
    tf.config.experimental.set_memory_growth(gpu, True)
from tensorflow import keras
print(f"Loading model: {MODEL_PATH.name} ...")
model = keras.models.load_model(str(MODEL_PATH))
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# ── Scan A ────────────────────────────────────────────────────────────────
process_scan(
    obj_path = PROJECT_ROOT / 'demo_frontend' / 'samples' / 'original_scan.obj',
    raw_out  = OUT_DIR / 'classified_raw.obj',
    pp_out   = OUT_DIR / 'classified_postprocessed.obj',
    model=model, device=device, config=config,
    label='Scan A'
)

# ── Scan B ────────────────────────────────────────────────────────────────
process_scan(
    obj_path = PROJECT_ROOT / 'demo_frontend' / 'samples' / 'original_scan_2.obj',
    raw_out  = OUT_DIR / 'classified_raw_scan2.obj',
    pp_out   = OUT_DIR / 'classified_postprocessed_scan2.obj',
    model=model, device=device, config=config,
    label='Scan B'
)

print(f"\nDone. Files written to {OUT_DIR}")
print("  classified_raw.obj / classified_postprocessed.obj")
print("  classified_raw_scan2.obj / classified_postprocessed_scan2.obj")
