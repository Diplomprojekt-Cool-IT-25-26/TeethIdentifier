"""Grid search over post-processing parameters to find best F1."""
import json
import sys
import numpy as np
import trimesh

sys.path.insert(0, '.')
from tools.mesh_morphology import postprocess

with open('data/data_part_1/lower/O52P1SZT/O52P1SZT_lower.json') as f:
    gt = (np.array(json.load(f)['labels']) != 0).astype(int)
with open('ml_outputs/results/prediction_data_O52P1SZT_lower_raw.json') as f:
    raw = np.array(json.load(f)['predictions'])

print("Loading mesh...")
mesh = trimesh.load('data/data_part_1/lower/O52P1SZT/O52P1SZT_lower.obj', process=False)
print("Starting grid search...")

def score(pred):
    tp = np.sum((pred == 1) & (gt == 1))
    fp = np.sum((pred == 1) & (gt == 0))
    fn = np.sum((pred == 0) & (gt == 1))
    p = tp / (tp + fp) * 100 if tp + fp else 0
    r = tp / (tp + fn) * 100 if tp + fn else 0
    return 2 * p * r / (p + r) if p + r else 0

results = []
total = 4 * 4 * 6 * 5
done = 0
for oi in range(1, 5):
    for ci in range(1, 5):
        for eg in range(1, 7):
            for og in range(0, 5):
                pred = postprocess(raw, mesh, open_iters=oi, close_iters=ci,
                                   erode_gingiva_iters=eg, open_gingiva_iters=og)
                results.append((score(pred), oi, ci, eg, og))
                done += 1
                if done % 20 == 0:
                    print(f"  {done}/{total}...")

results.sort(reverse=True)
print()
print(f"{'F1':>7}  {'open_t':>6}  {'close_t':>7}  {'erode_g':>7}  {'open_g':>6}")
print('-' * 42)
for s, oi, ci, eg, og in results[:20]:
    print(f"{s:6.2f}%  {oi:6d}  {ci:7d}  {eg:7d}  {og:6d}")
