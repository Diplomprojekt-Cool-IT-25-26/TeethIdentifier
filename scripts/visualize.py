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
from matplotlib.widgets import Button
from pathlib import Path
from scipy.spatial import KDTree

# torch must be imported before TensorFlow to avoid a Windows DLL conflict
# where TF's CUDA DLLs prevent torch/lib/shm.dll from loading
try:
    import torch
    TORCH_AVAILABLE = True
except (ImportError, OSError):
    TORCH_AVAILABLE = False

# Limit TF GPU memory so PyTorch can use the rest for patch generation
import tensorflow as tf
for gpu in tf.config.experimental.list_physical_devices('GPU'):
    tf.config.experimental.set_memory_growth(gpu, True)
from tensorflow import keras

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from tools.dataset_sampler import DatasetSampler
from tools.mesh_morphology import postprocess


def _propagate_predictions(mesh, sampled_indices, sampled_predictions, sampled_confidences):
    """Propagate predictions from sampled vertices to all using nearest neighbor."""
    kdtree = KDTree(mesh.vertices[sampled_indices])
    _, nearest_indices = kdtree.query(mesh.vertices, k=1)
    return sampled_predictions[nearest_indices].tolist(), sampled_confidences[nearest_indices].tolist()


def predict_scan_3d(obj_path: str, model_path: str, config: dict, sample_rate: int = 1, postprocess_enabled: bool = True, majority_vote_iters: int = 0, open_gingiva_iters: int = 0, reconstruct_iters: int = 0):
    """Run model predictions on all vertices of a 3D scan."""
    start = time.time()

    # Load mesh
    mesh = trimesh.load(obj_path, process=False)
    n_verts = len(mesh.vertices)
    print(f"Loaded {n_verts:,} vertices" + (f" (sampling 1/{sample_rate})" if sample_rate > 1 else ""))

    # Load annotations (required for CPU fallback; GPU path doesn't use them)
    json_path = obj_path.replace('.obj', '.json')
    if os.path.exists(json_path):
        with open(json_path, 'r') as f:
            annotations = json.load(f)
    else:
        annotations = None
        print(f"Warning: no annotation JSON found, CPU fallback will not be available")

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
            from tools.gpu_patch_generator_v3 import GPUPatchGeneratorV3 as GPUGenerator

            generator = GPUGenerator(
                mesh=mesh, annotations=annotations,
                patch_size=config['data_generation']['patch_size'],
                patch_radius=config['data_generation']['patch_radius']
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

    if postprocess_enabled:
        predictions = postprocess(predictions, mesh, majority_vote_iters=majority_vote_iters, open_gingiva_iters=open_gingiva_iters, reconstruct_iters=reconstruct_iters)

    gingiva = np.sum(predictions == 0)
    teeth = np.sum(predictions == 1)
    print(f"Predictions: {gingiva:,} gingiva, {teeth:,} teeth ({time.time()-start:.1f}s)")

    # Color mesh — flat realistic colors, no confidence modulation
    # Teeth: ivory/off-white  Gingiva: natural pink
    TOOTH_COLOR   = [245, 240, 225, 255]
    GINGIVA_COLOR = [220, 115, 130, 255]
    colors = np.where(
        (predictions == 1)[:, None],
        TOOTH_COLOR,
        GINGIVA_COLOR
    ).astype(np.uint8)
    mesh.visual.vertex_colors = colors

    return mesh, predictions, confidences


def _show_toggle_viewer(mesh, predictions):
    """3D mesh viewer with T to toggle between labeled (predictions) and unlabeled (raw mesh)."""
    TOOTH_COLOR   = [245, 240, 225, 255]
    GINGIVA_COLOR = [220, 115, 130, 255]
    PLAIN_COLOR   = [200, 195, 185, 255]

    n = len(mesh.vertices)
    labeled_colors = np.where(
        (predictions == 1)[:, None], TOOTH_COLOR, GINGIVA_COLOR
    ).astype(np.uint8)
    unlabeled_colors = np.full((n, 4), PLAIN_COLOR, dtype=np.uint8)

    try:
        import pyglet
        from trimesh.viewer.windowed import SceneViewer

        mesh_labeled = trimesh.Trimesh(
            vertices=mesh.vertices, faces=mesh.faces,
            vertex_colors=labeled_colors, process=False
        )
        mesh_unlabeled = trimesh.Trimesh(
            vertices=mesh.vertices, faces=mesh.faces,
            vertex_colors=unlabeled_colors, process=False
        )

        scene = trimesh.Scene()
        scene.add_geometry(mesh_labeled,   node_name='labeled',   geom_name='labeled')
        scene.add_geometry(mesh_unlabeled, node_name='unlabeled', geom_name='unlabeled')

        state = {'labeled': True}

        class ToggleViewer(SceneViewer):
            def on_key_press(self, symbol, modifiers):
                if symbol == pyglet.window.key.T:
                    state['labeled'] = not state['labeled']
                    if state['labeled']:
                        self.unhide_geometry('labeled')
                        self.hide_geometry('unlabeled')
                        self.set_caption('TeethIdentifier — Labeled (predictions)  |  T = toggle')
                    else:
                        self.hide_geometry('labeled')
                        self.unhide_geometry('unlabeled')
                        self.set_caption('TeethIdentifier — Unlabeled (raw mesh)  |  T = toggle')
                else:
                    super().on_key_press(symbol, modifiers)

        viewer = ToggleViewer(
            scene=scene,
            start_loop=False,
            caption='TeethIdentifier — Labeled (predictions)  |  T = toggle',
        )
        viewer.hide_geometry('unlabeled')
        pyglet.app.run()

    except Exception as e:
        print(f"3D viewer failed ({e}), falling back to matplotlib")
        v = mesh.vertices
        step = max(1, n // 15000)
        idx = np.arange(0, n, step)
        v_sub, pred_sub = v[idx], predictions[idx]

        fig = plt.figure(figsize=(14, 10))
        ax = fig.add_subplot(111, projection='3d')
        plt.subplots_adjust(bottom=0.12)
        state2 = {'labeled': True, 'artists': []}

        def redraw(labeled):
            for a in state2['artists']:
                a.remove()
            state2['artists'].clear()
            if labeled:
                s1 = ax.scatter(v_sub[pred_sub==1,0], v_sub[pred_sub==1,1], v_sub[pred_sub==1,2], c='#F5F0E1', s=1, alpha=0.9, label='Tooth')
                s2 = ax.scatter(v_sub[pred_sub==0,0], v_sub[pred_sub==0,1], v_sub[pred_sub==0,2], c='#DC7382', s=1, alpha=0.9, label='Gingiva')
                state2['artists'] = [s1, s2]
                ax.set_title('Labeled — Model Predictions  [T to toggle]', fontsize=13)
            else:
                s1 = ax.scatter(v_sub[:,0], v_sub[:,1], v_sub[:,2], c='#C0BAB0', s=1, alpha=0.7, label='Raw mesh')
                state2['artists'] = [s1]
                ax.set_title('Unlabeled — Raw Mesh  [T to toggle]', fontsize=13)
            ax.legend(loc='upper right', markerscale=6)
            fig.canvas.draw_idle()

        redraw(True)
        ax.set_xlabel('X'); ax.set_ylabel('Y'); ax.set_zlabel('Z')
        btn_ax = plt.axes([0.38, 0.02, 0.24, 0.06])
        btn = Button(btn_ax, 'Toggle Labeled / Unlabeled')

        def on_toggle(_event):
            state2['labeled'] = not state2['labeled']
            redraw(state2['labeled'])

        btn.on_clicked(on_toggle)
        fig.canvas.mpl_connect('key_press_event', lambda e: on_toggle(None) if e.key == 't' else None)
        plt.show()


def main():
    parser = argparse.ArgumentParser(description='Visualize 3D dental scan predictions')
    parser.add_argument('--fast', action='store_true', help='Sample 1/25th of vertices')
    parser.add_argument('--sample-rate', type=int, default=1, help='Sample every Nth vertex')
    parser.add_argument('--scan', type=str, default=None, help='Pattern to match scan filename')
    parser.add_argument('--no-postprocess', action='store_true', help='Skip post-processing (for comparison)')
    parser.add_argument('--majority-vote', type=int, default=0, metavar='N', help='Majority vote iterations to trim gingiva peninsulas (0 = off)')
    parser.add_argument('--open-gingiva', type=int, default=0, metavar='N', help='Opening iterations on gingiva to remove fingers (0 = off)')
    parser.add_argument('--reconstruct', type=int, default=0, metavar='N', help='Morphological reconstruction erosion depth to remove fingers (0 = off)')
    args = parser.parse_args()

    sample_rate = 25 if args.fast else args.sample_rate
    postprocess_enabled = not args.no_postprocess

    config_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'config.yaml')
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    model_path = Path(config['paths']['model_dir']) / 'teeth_classifier.keras'
    if not model_path.exists():
        print(f"[ERROR] Model not found: {model_path}")
        return 1

    data_path = Path(config['paths']['data_root'])
    pattern = f"**/*{args.scan}*.obj" if args.scan else "**/*.obj"
    obj_files = sorted(data_path.glob(pattern))
    if not obj_files:
        print(f"[ERROR] No .obj files found")
        return 1

    obj_path = str(obj_files[0])
    print(f"Scan: {Path(obj_path).name}")

    mesh, predictions, confidences = predict_scan_3d(obj_path, str(model_path), config, sample_rate, postprocess_enabled, args.majority_vote, args.open_gingiva, args.reconstruct)

    # Save outputs
    output_dir = Path(config['paths']['results_dir'])
    output_dir.mkdir(parents=True, exist_ok=True)
    suffix = (f"_{Path(obj_path).stem}"
              + ('_fast' if sample_rate > 1 else '')
              + ('' if postprocess_enabled else '_raw')
              + (f'_mv{args.majority_vote}' if args.majority_vote > 0 else '')
              + (f'_og{args.open_gingiva}' if args.open_gingiva > 0 else '')
              + (f'_rc{args.reconstruct}' if args.reconstruct > 0 else ''))

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
    ax.scatter(v[predictions==1, 0], v[predictions==1, 1], v[predictions==1, 2], c='#F5F0E1', s=1, alpha=0.8, label='Tooth')
    ax.scatter(v[predictions==0, 0], v[predictions==0, 1], v[predictions==0, 2], c='#DC7382', s=1, alpha=0.8, label='Gingiva')
    ax.set_xlabel('X'); ax.set_ylabel('Y'); ax.set_zlabel('Z')
    ax.legend()
    plt.tight_layout()
    img_path = output_dir / f'prediction_viz{suffix}.png'
    plt.savefig(img_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved image: {img_path}")

    # Interactive 3D viewer with T to toggle labeled/unlabeled
    print("Opening 3D viewer — press T to toggle labeled/unlabeled")
    _show_toggle_viewer(mesh, predictions)

    return 0


if __name__ == "__main__":
    sys.exit(main())
