# TeethIdentifier - Usage Guide

Complete guide for training and using the TeethIdentifier neural network system.

---

## 📋 Table of Contents

1. [Prerequisites](#prerequisites)
2. [Installation](#installation)
3. [Quick Start](#quick-start)
4. [Step-by-Step Workflow](#step-by-step-workflow)
5. [Common Tasks](#common-tasks)
6. [Troubleshooting](#troubleshooting)
7. [Advanced Usage](#advanced-usage)

---

## Prerequisites

### Hardware Requirements
- **GPU**: NVIDIA GPU with 8GB+ VRAM (e.g., RTX 3070, RTX 3080)
- **RAM**: 16GB+ recommended
- **Storage**: 50GB+ free space for training data

### Software Requirements
- **Python**: 3.9 or higher
- **NVIDIA Drivers**: Latest version
- **CUDA**: 11.8 or 12.x
- **cuDNN**: 8.9+

---

## Installation

### Step 1: Navigate to Project Directory
```bash
cd "D:\All projects\TeethIdentifier"
```

### Step 2: Install Dependencies
```bash
pip install -r requirements.txt
```

**Expected installation time**: 5-10 minutes

**Verify installation**:
```bash
python -c "import tensorflow as tf; print(f'TensorFlow version: {tf.__version__}')"
```

### Step 3: Verify GPU Setup
```bash
python scripts/check_gpu.py
```

**Expected output**:
```
======================================================================
TensorFlow GPU Configuration Check
======================================================================

✓ TensorFlow version: 2.15.0

======================================================================
GPUs detected: 1
======================================================================

  GPU 0: /physical_device:GPU:0
    Device name: NVIDIA GeForce RTX 3070

✓ Test computation successful!
  Matrix multiplication result shape: (1000, 1000)

======================================================================
✓ GPU check PASSED - Ready for training!
======================================================================
```

---

## Quick Start

For those who want to get started immediately:

```bash
# 1. Generate training data (~2-3 hours)
python scripts/generate_training_data.py

# 2. Train the model (~8-12 hours on RTX 3070)
python src/train.py

# 3. Evaluate performance
python src/evaluate.py

# 4. Run inference on a scan
python src/predict.py --obj data/data_part_1/upper/PATIENT_001/PATIENT_001_upper.obj --output exports/predictions/

# 5. Launch dashboard for visualization
streamlit run app.py
```

---

## Step-by-Step Workflow

### Phase 1: Data Preparation (2-3 hours)

#### Generate Training Data

This step converts 3D dental scans into 2D image patches for training.

```bash
python scripts/generate_training_data.py
```

**What it does**:
- Loads all OBJ/JSON scan pairs from `data/` directory
- Splits patients into train/val/test (70/15/15)
- Generates 500 balanced samples per scan
- Saves to `training_data/train.pkl`, `val.pkl`, `test.pkl`

**Expected output**:
```
======================================================================
TeethIdentifier Training Data Generation
======================================================================

Initializing dataset sampler...
Found 1850 total scans

Patient-level split:
  Total patients: 1850
  Train: 1295 patients (70.0%)
  Val: 277 patients (15.0%)
  Test: 278 patients (15.0%)

Scan-level split:
  Train: 1295 scans
  Val: 277 scans
  Test: 278 scans

======================================================================
Generating TRAIN Data
======================================================================
  Scans: 1295
  Samples per scan: 500
  Expected total: 647,500 samples

Processing train scans: 100%|████████████████████| 1295/1295
...
✓ Saved successfully!
```

**Files created**:
- `training_data/train.pkl` (~5-10 GB)
- `training_data/val.pkl` (~1-2 GB)
- `training_data/test.pkl` (~1-2 GB)

**Configuration**:
Edit `config.yaml` to adjust:
- `scans_for_training`: Limit number of scans (for testing)
- `samples_per_scan`: Samples per scan (default: 500)
- `split_ratios`: Train/val/test ratios

---

### Phase 2: Model Training (8-12 hours)

#### Train the Neural Network

```bash
python src/train.py
```

**What it does**:
- Loads train/val datasets
- Builds CNN model (TeethNet)
- Trains with callbacks (early stopping, LR scheduling)
- Saves best model and training history

**Console output example**:
```
======================================================================
TeethIdentifier Training Pipeline
======================================================================
Started: 2026-01-05 10:00:00
======================================================================

======================================================================
GPU Configuration
======================================================================
GPUs detected: 1
✓ Memory growth enabled
  GPU 0: /physical_device:GPU:0
✓ Mixed precision enabled: mixed_float16
  (Expected 2× speedup on RTX 3070)
✓ Random seed set: 42
======================================================================

======================================================================
Data Loading
======================================================================

Loading data from: training_data/train.pkl
  Total samples: 647,500
  Gingiva: 323,750 (50.0%)
  Teeth: 323,750 (50.0%)

Loading data from: training_data/val.pkl
  Total samples: 138,500
  Gingiva: 69,250 (50.0%)
  Teeth: 69,250 (50.0%)

✓ Data loading complete
======================================================================

======================================================================
Model Building
======================================================================

Model Summary:
----------------------------------------------------------------------
Model: "TeethNet"
_________________________________________________________________
 Layer (type)                Output Shape              Param #
=================================================================
 rescaling (Rescaling)       (None, 100, 100, 3)       0
 conv2d_1 (Conv2D)           (None, 100, 100, 32)      896
 batch_norm_1 (BatchNorm)    (None, 100, 100, 32)      128
 max_pool_1 (MaxPooling2D)   (None, 50, 50, 32)        0
 dropout_1 (Dropout)         (None, 50, 50, 32)        0
 conv2d_2 (Conv2D)           (None, 50, 50, 64)        18,496
 batch_norm_2 (BatchNorm)    (None, 50, 50, 64)        256
 max_pool_2 (MaxPooling2D)   (None, 25, 25, 64)        0
 dropout_2 (Dropout)         (None, 25, 25, 64)        0
 conv2d_3 (Conv2D)           (None, 25, 25, 128)       73,856
 batch_norm_3 (BatchNorm)    (None, 25, 25, 128)       512
 max_pool_3 (MaxPooling2D)   (None, 12, 12, 128)       0
 dropout_3 (Dropout)         (None, 12, 12, 128)       0
 flatten (Flatten)           (None, 18432)             0
 dense_1 (Dense)             (None, 256)               4,718,848
 batch_norm_4 (BatchNorm)    (None, 256)               1,024
 dropout_4 (Dropout)         (None, 256)               0
 dense_2 (Dense)             (None, 64)                16,448
 dropout_5 (Dropout)         (None, 64)                0
 output (Dense)              (None, 1)                 65
=================================================================
Total params: 4,830,529
Trainable params: 4,829,569
Non-trainable params: 960
_________________________________________________________________

Parameter Count:
  Trainable:     4,829,569
  Non-trainable: 960
  Total:         4,830,529

✓ Model built successfully
======================================================================

======================================================================
Training
======================================================================

Training Configuration:
  Epochs: 50
  Batch size: 32
  Steps per epoch: 20234
  Validation steps: 4328
  Total training steps: 1,011,700

Callbacks:
  - ModelCheckpoint
  - EarlyStopping
  - ReduceLROnPlateau
  - TensorBoard
  - CSVLogger

======================================================================
Starting training...
======================================================================

Epoch 1/50
20234/20234 [==============================] - 1015s 50ms/step - loss: 0.3254 - accuracy: 0.8532 - precision: 0.8621 - recall: 0.8425 - auc: 0.9234 - val_loss: 0.2145 - val_accuracy: 0.9134 - val_precision: 0.9245 - val_recall: 0.9012 - val_auc: 0.9645

Epoch 2/50
20234/20234 [==============================] - 987s 49ms/step - loss: 0.2034 - accuracy: 0.9187 - precision: 0.9256 - recall: 0.9112 - auc: 0.9712 - val_loss: 0.1823 - val_accuracy: 0.9287 - val_precision: 0.9312 - val_recall: 0.9254 - val_auc: 0.9787
...

Epoch 00012: ReduceLROnPlateau reducing learning rate to 0.0005000000237487257.
...

Epoch 00023: early stopping

======================================================================
Training Complete!
======================================================================
  Training time: 345.2 minutes
  Final train accuracy: 0.9523
  Final val accuracy: 0.9387
======================================================================
```

**Files created**:
- `models/teeth_classifier.keras` - Final trained model
- `models/training_history.json` - Training metrics
- `models/training_log.csv` - Epoch-by-epoch log
- `models/checkpoints/model_epoch_XX_val_acc_X.XXXX.keras` - Best checkpoints
- `logs/YYYYMMDD-HHMMSS/` - TensorBoard logs
- `results/training_history.png` - Training curves

**Monitor training with TensorBoard**:
```bash
# In a separate terminal
tensorboard --logdir logs
```
Then open http://localhost:6006 in your browser

**Interrupt training**:
Press `Ctrl+C` - you'll be asked if you want to save the current model

**Resume from checkpoint**:
If training was interrupted, the best model is automatically saved. You can manually load checkpoints:
```python
from tensorflow import keras
model = keras.models.load_model('models/checkpoints/model_epoch_12_val_acc_0.9387.keras')
```

---

### Phase 3: Model Evaluation (5 minutes)

#### Evaluate Performance

```bash
python src/evaluate.py
```

**What it does**:
- Loads trained model
- Runs predictions on test set
- Computes metrics (accuracy, precision, recall, F1, AUC)
- Generates confusion matrix and ROC curve
- Analyzes per-patient performance
- Saves sample predictions

**Expected output**:
```
======================================================================
TeethIdentifier Model Evaluation
======================================================================

Loading model from: models/teeth_classifier.keras
✓ Model loaded successfully

Loading test data...
  Total samples: 139,000
  Gingiva: 69,500 (50.0%)
  Teeth: 69,500 (50.0%)
✓ Loaded 139,000 test samples

Running predictions...
1088/1088 [==============================] - 23s 21ms/step

✓ Predictions complete
  Gingiva vertices: 68,234 (49.1%)
  Tooth vertices: 70,766 (50.9%)

Calculating metrics...
✓ Metrics calculated

======================================================================
Evaluation Metrics
======================================================================
  Accuracy:  0.9287 (92.87%)
  Precision: 0.9312
  Recall:    0.9254
  F1-Score:  0.9283
  ROC-AUC:   0.9787

Confusion Matrix:
                Predicted
              Gingiva  Tooth
  Actual Gingiva  64234   5266
         Tooth     4645  64855

Detailed Breakdown:
  True Negatives:  64,234
  False Positives: 5,266
  False Negatives: 4,645
  True Positives:  64,855
======================================================================

✓ Classification report saved to: results/classification_report.txt
✓ Confusion matrix saved to: results/confusion_matrix.png
✓ ROC curve saved to: results/roc_curve.png

Analyzing per-patient performance...
✓ Per-patient analysis saved to: results/per_patient_analysis.csv

Per-Patient Summary:
  Total patients: 278
  Mean accuracy: 0.9287 ± 0.0234
  Min accuracy: 0.8523 (Patient: O52P1SZT)
  Max accuracy: 0.9876 (Patient: A12K5MNP)

Saving 20 sample predictions...
✓ Sample predictions saved to: results/sample_predictions/predictions_grid.png

======================================================================
Evaluation Complete!
======================================================================

Results saved to: results

Generated files:
  - classification_report.txt
  - confusion_matrix.png
  - roc_curve.png
  - per_patient_analysis.csv
  - sample_predictions/predictions_grid.png
======================================================================
```

**Files created**:
- `results/classification_report.txt` - Detailed metrics
- `results/confusion_matrix.png` - Confusion matrix heatmap
- `results/roc_curve.png` - ROC curve visualization
- `results/per_patient_analysis.csv` - Per-patient breakdown
- `results/sample_predictions/predictions_grid.png` - Sample predictions

---

### Phase 4: Inference (1-5 minutes per scan)

#### Run Predictions on New Scans

**Single file**:
```bash
python src/predict.py \
  --obj data/data_part_1/upper/PATIENT_001/PATIENT_001_upper.obj \
  --output exports/predictions/
```

**Batch processing** (entire directory):
```bash
python src/predict.py \
  --obj data/data_part_1/upper/ \
  --output exports/predictions/
```

**Custom model**:
```bash
python src/predict.py \
  --obj scan.obj \
  --output exports/predictions/ \
  --model models/checkpoints/model_epoch_12_val_acc_0.9387.keras
```

**Expected output**:
```
======================================================================
TeethIdentifier Inference Pipeline
======================================================================

Loading model from: models/teeth_classifier.keras
✓ Model loaded successfully

Loading OBJ file: data/data_part_1/upper/PATIENT_001/PATIENT_001_upper.obj
  Patient ID: PATIENT_001
  Jaw: upper
  Vertices: 48,523

Generating patches for all vertices...
Generating patches: 100%|████████████████| 48523/48523 [02:15<00:00, 357.89it/s]
✓ Generated 48,523 patches

Running predictions...
379/379 [==============================] - 8s 21ms/step

✓ Predictions complete
  Gingiva vertices: 14,234 (29.3%)
  Tooth vertices: 34,289 (70.7%)

✓ Predictions saved to: exports/predictions/PATIENT_001_upper_predictions.json

======================================================================
✓ Inference complete!
======================================================================
```

**Output JSON format**:
```json
{
  "name": "PATIENT_001_upper.obj",
  "labels": [0, 1, 1, 0, 1, 1, ...],
  "instances": [],
  "jaw": "upper",
  "prediction_metadata": {
    "model_version": "v1.0",
    "timestamp": "2026-01-05T14:23:45.123456",
    "model_path": "models/teeth_classifier.keras",
    "threshold": 0.5,
    "num_vertices": 48523,
    "num_gingiva": 14234,
    "num_teeth": 34289
  }
}
```

---

### Phase 5: Visualization Dashboard

#### Launch Streamlit App

```bash
streamlit run app.py
```

**What it provides**:
- **Step 1: Data Overview** - View dataset statistics and sample images
- **Step 2: Model Configuration** - Explore hyperparameters and architecture
- **Step 3: Training** - View training curves and metrics
- **Step 4: Evaluation** - Interactive confusion matrix and ROC curve
- **Step 5: Inference** - Upload scans and download predictions

**URL**: Automatically opens in browser at http://localhost:8501

**Tips**:
- Use for presentations - looks professional
- Interactive visualizations
- No coding required for viewers
- Can upload custom pickle files to explore

---

## Common Tasks

### Adjust Hyperparameters

Edit `config.yaml`:

```yaml
# Increase batch size for faster training (if you have VRAM)
training:
  batch_size: 64  # Default: 32

# Reduce epochs for quick testing
training:
  epochs: 10  # Default: 50

# Use fewer scans for quick testing
data_generation:
  scans_for_training: 100  # Default: 1500
  samples_per_scan: 100    # Default: 500
```

After editing, regenerate data and retrain.

---

### Test on Small Dataset

For quick iteration during development:

```yaml
# config.yaml
data_generation:
  scans_for_training: 50      # Just 50 scans
  samples_per_scan: 100       # 100 samples per scan

training:
  epochs: 5                    # Just 5 epochs
```

Then:
```bash
python scripts/generate_training_data.py  # ~10 minutes
python src/train.py                        # ~30 minutes
```

---

### Resume Training from Checkpoint

If training was interrupted:

```python
# Create custom training script
from tensorflow import keras
from src.train import TeethTrainer

# Load checkpoint
model = keras.models.load_model('models/checkpoints/model_epoch_12_val_acc_0.9387.keras')

# Continue training
trainer = TeethTrainer()
trainer.model = model
# ... continue training
```

---

### Export Model for Production

```python
from tensorflow import keras

# Load model
model = keras.models.load_model('models/teeth_classifier.keras')

# Export to TensorFlow SavedModel format
model.export('models/saved_model/')

# Or export to TFLite for mobile
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()
with open('models/model.tflite', 'wb') as f:
    f.write(tflite_model)
```

---

### Compare Multiple Models

```bash
# Train with different configs
python src/train.py  # Creates models/teeth_classifier.keras

# Rename to preserve
mv models/teeth_classifier.keras models/teeth_classifier_v1.keras

# Edit config.yaml (e.g., change learning rate)

# Train again
python src/train.py  # Creates new models/teeth_classifier.keras

# Rename
mv models/teeth_classifier.keras models/teeth_classifier_v2.keras

# Evaluate both
python src/evaluate.py  # Will use default model
```

---

## Troubleshooting

### GPU Not Detected

**Problem**: `python scripts/check_gpu.py` shows 0 GPUs

**Solution**:
1. Check NVIDIA drivers:
   ```bash
   nvidia-smi
   ```
   Should show your RTX 3070

2. Reinstall TensorFlow with GPU support:
   ```bash
   pip uninstall tensorflow
   pip install tensorflow[and-cuda]>=2.15.0
   ```

3. Check CUDA compatibility:
   ```bash
   python -c "import tensorflow as tf; print(tf.sysconfig.get_build_info())"
   ```

---

### Out of Memory (OOM) Error

**Problem**: `ResourceExhaustedError: OOM when allocating tensor`

**Solution**:
Reduce batch size in `config.yaml`:
```yaml
training:
  batch_size: 16  # Reduce from 32
```

Or disable mixed precision:
```yaml
training:
  use_mixed_precision: false
```

---

### Training Data Not Found

**Problem**: `FileNotFoundError: Training data not found: training_data/train.pkl`

**Solution**:
Generate training data first:
```bash
python scripts/generate_training_data.py
```

---

### Model Not Found

**Problem**: `FileNotFoundError: Model not found: models/teeth_classifier.keras`

**Solution**:
Train the model first:
```bash
python src/train.py
```

---

### Slow Training

**Problem**: Training is very slow (>30 min/epoch)

**Possible causes**:
1. **GPU not being used** - Check `python scripts/check_gpu.py`
2. **Mixed precision disabled** - Enable in `config.yaml`:
   ```yaml
   training:
     use_mixed_precision: true
   ```
3. **Data not cached** - First epoch is always slower (builds cache)

---

### Poor Model Performance (<85% accuracy)

**Possible solutions**:
1. **Train longer** - Increase epochs:
   ```yaml
   training:
     epochs: 75  # From 50
   ```

2. **Adjust learning rate**:
   ```yaml
   training:
     learning_rate: 0.0005  # From 0.001
   ```

3. **Check data balance** - Run evaluation to see per-class metrics

4. **Try different architecture** - Edit `config.yaml`:
   ```yaml
   model:
     conv_filters: [64, 128, 256]  # Increase filters
   ```

---

## Advanced Usage

### Custom Data Augmentation

Enable in `config.yaml`:
```yaml
augmentation:
  enabled: true
  rotation_range: 15
  horizontal_flip: true
  vertical_flip: true
```

Then modify `src/data_loader.py` to implement augmentation.

---

### Hyperparameter Tuning

Use Keras Tuner:
```python
import keras_tuner as kt
from src.model import TeethClassifier

def build_model(hp):
    config['training']['learning_rate'] = hp.Float('lr', 1e-4, 1e-2, sampling='log')
    config['model']['conv_filters'] = [
        hp.Int('filters_1', 16, 64, step=16),
        hp.Int('filters_2', 32, 128, step=16),
        hp.Int('filters_3', 64, 256, step=16)
    ]
    classifier = TeethClassifier(config)
    return classifier.build_model()

tuner = kt.RandomSearch(build_model, objective='val_accuracy', max_trials=10)
tuner.search(train_dataset, validation_data=val_dataset)
```

---

### Multi-GPU Training

Edit `src/train.py` to use `tf.distribute`:
```python
strategy = tf.distribute.MirroredStrategy()
with strategy.scope():
    model = classifier.build_model()
```

---

### Class Weighting

If data is imbalanced, use class weights:
```yaml
training:
  use_class_weights: true
```

Then in `src/train.py`:
```python
class_weights = {0: 1.5, 1: 1.0}  # Weight gingiva more
model.fit(..., class_weight=class_weights)
```

---

## Tips & Best Practices

1. **Start small** - Test with 50 scans and 5 epochs first
2. **Monitor GPU usage** - Run `nvidia-smi` in separate terminal
3. **Save checkpoints frequently** - Already configured in callbacks
4. **Use TensorBoard** - Great for visualizing training progress
5. **Keep original data** - Never modify `data/` directory
6. **Version control models** - Rename models when experimenting
7. **Document experiments** - Keep notes on what configs you tried

---

## Performance Benchmarks

**Expected times on RTX 3070**:

| Task | Time |
|------|------|
| Data generation (1500 scans) | 2-3 hours |
| Single epoch training | 15-20 min |
| Full training (30-50 epochs) | 8-12 hours |
| Evaluation | 5 min |
| Single scan inference | 2-3 min |
| Batch inference (100 scans) | 3-4 hours |

---

## Getting Help

1. **Check logs**: All scripts print detailed progress
2. **Read error messages**: Usually self-explanatory
3. **Check config.yaml**: Most issues are configuration problems
4. **Verify data**: Run `verify_dataset_structure.py`
5. **Test GPU**: Run `scripts/check_gpu.py`

---

## Summary Workflow

```bash
# Complete workflow in one go:

# 1. Setup
pip install -r requirements.txt
python scripts/check_gpu.py

# 2. Data preparation
python scripts/generate_training_data.py

# 3. Training
python src/train.py

# 4. Evaluation
python src/evaluate.py

# 5. Inference
python src/predict.py --obj data/scan.obj --output exports/

# 6. Visualization
streamlit run app.py
```

That's it! You now have a fully trained teeth vs gingiva classifier. 🦷✨

---

**Last updated**: 2026-01-05
**Version**: 1.0.0
