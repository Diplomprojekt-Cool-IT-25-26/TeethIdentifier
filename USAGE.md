# TeethIdentifier — Usage Guide

Binary classification of teeth vs gingiva on 3D dental scans.
All commands are run from the **project root** (`D:\All projects\TeethIdentifier`) inside the venv.

```
venv\Scripts\activate
```

---

## Workflow overview

```
1. generate_data  →  2. train  →  3. evaluate  →  4. visualize
                          ↑
                   continue_training  (if more epochs needed)
```

---

## 1. Generate training data

Samples patches from raw `.obj` scans and saves them as `.pkl` files.
Uses **three-pool balanced sampling**: ⅓ boundary, ⅓ clear gingiva, ⅓ clear tooth.

```
venv\Scripts\python scripts\generate_data.py
```

Key settings in `config.yaml`:
| Key | Default | Meaning |
|---|---|---|
| `data_generation.scans_for_training` | 400 | How many of the 1,850 scans to use |
| `data_generation.samples_per_scan` | 500 | Patches per scan |
| `data_generation.boundary_weight` | 0.333 | Fraction drawn from boundary region |
| `data_generation.boundary_radius` | 5.0 | mm radius to define "near boundary" |

Output: `ml_outputs/training_data/{train,val,test}.pkl`

---

## 2. Train from scratch

```
scripts\train_gpu.bat
```

Or directly:

```
venv\Scripts\python src\train.py
```

Saves the best checkpoint to `ml_outputs/models/teeth_classifier.keras`
and per-epoch checkpoints to `ml_outputs/models/checkpoints/`.

Key settings in `config.yaml`:
| Key | Default | Meaning |
|---|---|---|
| `training.epochs` | 100 | Max epochs (early stopping at patience=20) |
| `training.batch_size` | 64 | Training batch size |
| `training.learning_rate` | 0.001 | Initial learning rate |

---

## 3. Continue training from a checkpoint

Use this when training has plateaued and you want to resume with a lower LR.

```
venv\Scripts\python scripts\continue_training.py
venv\Scripts\python scripts\continue_training.py --model ml_outputs\models\checkpoints\<name>.keras
venv\Scripts\python scripts\continue_training.py --epochs 30 --lr 0.00005
```

| Flag | Default | Meaning |
|---|---|---|
| `--model` | `ml_outputs/models/teeth_classifier.keras` | Checkpoint to resume from |
| `--epochs` | 50 | Additional epochs to train |
| `--lr` | 0.0001 | Learning rate for this run |

Saves updated model back to `ml_outputs/models/teeth_classifier.keras`.

---

## 4. Evaluate on the test set

Runs the model against the held-out test split and produces accuracy, precision,
recall, F1, ROC-AUC, confusion matrix, and per-patient breakdown.

```
venv\Scripts\python src\evaluate.py
```

Output saved to `ml_outputs/results/`.

---

## 5. Visualise predictions vs ground truth

Generates patches with GPU V3, runs inference, applies post-processing, then
saves a **side-by-side comparison PNG** (ground truth left, prediction right)
and the coloured `.obj` mesh.

```
venv\Scripts\python scripts\visualize.py
venv\Scripts\python scripts\visualize.py --scan O52P1SZT
venv\Scripts\python scripts\visualize.py --no-postprocess
venv\Scripts\python scripts\visualize.py --model ml_outputs\models\checkpoints\<name>.keras
```

| Flag | Default | Meaning |
|---|---|---|
| `--scan` | first found | Substring to match scan filename |
| `--model` | `teeth_classifier.keras` | Model to use for inference |
| `--no-postprocess` | off | Skip morphological smoothing step |

Output: `ml_outputs/results/<scan>_comparison.png` and `<scan>_prediction.obj`

Stats printed: tooth recall, gingiva recall, spill counts, overall accuracy.

---

## 6. Benchmark patch generators (V2 vs V3)

Measures init time + generation time for both generators and checks against
the 30-second target.

```
venv\Scripts\python tests\benchmark.py
venv\Scripts\python tests\benchmark.py --vertices 5000
```

| Flag | Default | Meaning |
|---|---|---|
| `--vertices` | full mesh | Vertices to time (use 5000 for a quick run) |

---

## File layout

```
TeethIdentifier/
├── config.yaml                  ← all tunable settings
├── USAGE.md                     ← this file
│
├── scripts/
│   ├── generate_data.py         ← step 1: build .pkl training datasets
│   ├── train_gpu.bat            ← step 2: launch training (Windows)
│   ├── continue_training.py     ← step 3: resume from checkpoint
│   └── visualize.py             ← step 5: predict + compare vs ground truth
│
├── src/
│   ├── model.py                 ← TeethNet CNN architecture
│   ├── data_loader.py           ← loads .pkl files into TF datasets
│   ├── train.py                 ← training loop (called by train_gpu.bat)
│   ├── evaluate.py              ← test-set evaluation + plots
│   └── predict.py               ← batch inference on new .obj files
│
├── tools/
│   ├── dataset_sampler.py       ← patch sampling from raw scans
│   ├── gpu_patch_generator_v2.py ← GPU patches (math projection)
│   ├── gpu_patch_generator_v3.py ← GPU patches (custom rasterizer, fastest)
│   ├── gpu_patch_generator.py   ← GPU patches (PyTorch3D, legacy)
│   ├── mesh_morphology.py       ← post-processing (open/close/smooth)
│   └── verify_dataset.py        ← sanity-check raw data files
│
├── tests/
│   └── benchmark.py             ← step 6: time V2 vs V3 generators
│
└── ml_outputs/
    ├── training_data/           ← generated .pkl files
    ├── models/
    │   ├── teeth_classifier.keras       ← current best model (V3 patches)
    │   ├── teeth_classifier_old.keras   ← legacy model (V2 patches)
    │   └── checkpoints/                 ← per-epoch saves
    ├── logs/                    ← TensorBoard + CSV training logs
    └── results/                 ← evaluation plots, prediction meshes
```

---

## Benchmarks to reach

| Metric | Target | Current |
|---|---|---|
| Val accuracy | 99% | ~95.5% |
| Full-scan inference time | < 30s | ~24s patch gen + ~10s inference |
