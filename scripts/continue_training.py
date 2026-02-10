"""Continue training from a saved model checkpoint."""

import os
import sys
import yaml
import argparse
import tensorflow as tf
from tensorflow import keras
from pathlib import Path
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.data_loader import TeethDataLoader


def continue_training(model_path: str, config_path: str = None, additional_epochs: int = 50, learning_rate: float = 0.0001):
    """Continue training from a saved model."""
    if config_path is None:
        config_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'config.yaml')

    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    print(f"Continue training: {model_path}, epochs={additional_epochs}, lr={learning_rate}")

    # GPU setup
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print(f"GPU enabled: {len(gpus)} device(s)")

    if config['training'].get('use_mixed_precision', True):
        keras.mixed_precision.set_global_policy('mixed_float16')

    # Load data
    loader = TeethDataLoader(config)
    train_dataset, train_samples = loader.load_train_data()
    val_dataset, val_samples = loader.load_val_data()
    print(f"Data: {train_samples:,} train, {val_samples:,} val")

    # Load and recompile model
    model = keras.models.load_model(model_path)
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        loss=config['training']['loss'],
        metrics=['accuracy', keras.metrics.Precision(name='precision'),
                 keras.metrics.Recall(name='recall'), keras.metrics.AUC(name='auc')]
    )

    # Callbacks
    checkpoint_dir = Path(config['paths']['model_dir']) / 'checkpoints'
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    log_dir = Path(config['paths']['logs_dir']) / datetime.now().strftime('%Y%m%d-%H%M%S-continue')

    callbacks = [
        keras.callbacks.ModelCheckpoint(
            str(checkpoint_dir / 'model_epoch_{epoch:02d}_val_acc_{val_accuracy:.4f}.keras'),
            monitor='val_accuracy', save_best_only=True, mode='max', verbose=1),
        keras.callbacks.EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True, verbose=1),
        keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=7, min_lr=0.00001, verbose=1),
        keras.callbacks.TensorBoard(log_dir=str(log_dir), histogram_freq=1),
        keras.callbacks.CSVLogger(str(Path(config['paths']['logs_dir']) / 'training_continue.csv'), append=True)
    ]

    batch_size = config['training']['batch_size']
    start_time = datetime.now()

    history = model.fit(
        train_dataset, epochs=additional_epochs,
        steps_per_epoch=train_samples // batch_size,
        validation_data=val_dataset, validation_steps=val_samples // batch_size,
        callbacks=callbacks, verbose=1
    )

    training_time = (datetime.now() - start_time).total_seconds() / 60.0
    print(f"Training complete: {training_time:.1f}min, "
          f"train_acc={history.history['accuracy'][-1]:.4f}, "
          f"val_acc={history.history['val_accuracy'][-1]:.4f}")

    final_model_path = Path(config['paths']['model_dir']) / 'teeth_classifier_v2.keras'
    model.save(str(final_model_path))
    print(f"Saved: {final_model_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Continue training from checkpoint')
    parser.add_argument('--model', type=str, default='ml_outputs/models/teeth_classifier.keras')
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--lr', type=float, default=0.0001)
    args = parser.parse_args()

    continue_training(model_path=args.model, additional_epochs=args.epochs, learning_rate=args.lr)
