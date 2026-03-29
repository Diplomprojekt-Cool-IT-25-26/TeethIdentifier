"""Training Script for TeethIdentifier Neural Network."""

import os
import sys
import json
import yaml
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.model import TeethClassifier
from src.data_loader import TeethDataLoader


class TeethTrainer:
    """Training pipeline for TeethNet CNN."""

    def __init__(self, config_path: str = 'config.yaml'):
        self.config_path = config_path
        self.config = self.load_config()
        self.model = None
        self.history = None

        self.model_dir = Path(self.config['paths']['model_dir'])
        self.logs_dir = Path(self.config['paths']['logs_dir'])
        self.results_dir = Path(self.config['paths'].get('results_dir', 'results'))

        self.model_dir.mkdir(parents=True, exist_ok=True)
        (self.model_dir / 'checkpoints').mkdir(parents=True, exist_ok=True)
        self.logs_dir.mkdir(parents=True, exist_ok=True)
        self.results_dir.mkdir(parents=True, exist_ok=True)

    def load_config(self) -> Dict[str, Any]:
        with open(self.config_path, 'r') as f:
            return yaml.safe_load(f)

    def setup_gpu(self) -> None:
        """Configure TensorFlow GPU settings."""
        gpus = tf.config.list_physical_devices('GPU')
        if gpus:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            print(f"GPU: {len(gpus)} device(s)")
        else:
            print("No GPU detected, using CPU")

        if self.config['training'].get('use_mixed_precision', False):
            tf.keras.mixed_precision.set_global_policy('mixed_float16')

        seed = self.config.get('random_seed', 42)
        np.random.seed(seed)
        tf.random.set_seed(seed)

    def prepare_data(self):
        """Load training and validation datasets."""
        loader = TeethDataLoader(self.config)
        train_dataset, train_samples = loader.load_train_data()
        val_dataset, val_samples = loader.load_val_data()
        print(f"Data: {train_samples:,} train, {val_samples:,} val")
        return train_dataset, train_samples, val_dataset, val_samples

    def build_model(self) -> keras.Model:
        """Build and compile the CNN model."""
        classifier = TeethClassifier(self.config)
        model = classifier.build_model()
        params = classifier.count_parameters()
        print(f"Model: {params['total']:,} parameters")
        self.model = model
        return model

    def get_callbacks(self) -> List[keras.callbacks.Callback]:
        """Create training callbacks."""
        callbacks = []

        checkpoint_path = self.model_dir / 'checkpoints' / 'model_epoch_{epoch:02d}_val_acc_{val_accuracy:.4f}.keras'
        callbacks.append(keras.callbacks.ModelCheckpoint(
            filepath=str(checkpoint_path),
            monitor=self.config['training']['monitor_metric'],
            save_best_only=self.config['training']['save_best_only'],
            mode='max' if 'acc' in self.config['training']['monitor_metric'] else 'min',
            verbose=1
        ))

        callbacks.append(keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=self.config['training']['early_stopping_patience'],
            restore_best_weights=True,
            verbose=1
        ))

        callbacks.append(keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=self.config['training']['reduce_lr_factor'],
            patience=self.config['training']['reduce_lr_patience'],
            min_lr=self.config['training']['min_learning_rate'],
            verbose=1
        ))

        if self.config['logging'].get('tensorboard', True):
            log_dir = self.logs_dir / datetime.now().strftime("%Y%m%d-%H%M%S")
            callbacks.append(keras.callbacks.TensorBoard(
                log_dir=str(log_dir),
                histogram_freq=self.config['logging'].get('histogram_freq', 1)
            ))

        if self.config['logging'].get('csv_logging', True):
            callbacks.append(keras.callbacks.CSVLogger(
                filename=str(self.model_dir / 'training_log.csv')
            ))

        return callbacks

    def train(self, train_dataset, train_samples, val_dataset, val_samples) -> None:
        """Train the model."""
        batch_size = self.config['training']['batch_size']
        epochs = self.config['training']['epochs']
        steps_per_epoch = train_samples // batch_size

        print(f"Training: {epochs} epochs, batch_size={batch_size}")
        callbacks = self.get_callbacks()

        start_time = datetime.now()
        self.history = self.model.fit(
            train_dataset,
            epochs=epochs,
            steps_per_epoch=steps_per_epoch,
            validation_data=val_dataset,
            callbacks=callbacks,
            verbose=1
        )
        training_time = (datetime.now() - start_time).total_seconds()

        print(f"Complete: {training_time/60:.1f}min, "
              f"train_acc={self.history.history['accuracy'][-1]:.4f}, "
              f"val_acc={self.history.history['val_accuracy'][-1]:.4f}")

    def save_model(self) -> None:
        model_path = self.model_dir / 'teeth_classifier.keras'
        self.model.save(str(model_path))
        print(f"Model saved: {model_path}")

    def save_history(self) -> None:
        history_path = self.model_dir / 'training_history.json'
        history_dict = {key: [float(val) for val in values] for key, values in self.history.history.items()}
        with open(history_path, 'w') as f:
            json.dump(history_dict, f, indent=2)

    def plot_training_history(self) -> None:
        if self.history is None:
            return

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        ax1.plot(self.history.history['accuracy'], label='Train')
        ax1.plot(self.history.history['val_accuracy'], label='Val')
        ax1.set_title('Accuracy')
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        ax2.plot(self.history.history['loss'], label='Train')
        ax2.plot(self.history.history['val_loss'], label='Val')
        ax2.set_title('Loss')
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(self.results_dir / 'training_history.png', dpi=150)
        plt.close()

    def run(self) -> None:
        """Run the complete training pipeline."""
        print(f"TeethIdentifier Training - {datetime.now().strftime('%Y-%m-%d %H:%M')}")

        try:
            self.setup_gpu()
            train_dataset, train_samples, val_dataset, val_samples = self.prepare_data()
            self.build_model()
            self.train(train_dataset, train_samples, val_dataset, val_samples)
            self.save_model()
            self.save_history()
            self.plot_training_history()
            print(f"Output: {self.model_dir}")
            return 0

        except KeyboardInterrupt:
            print("Training interrupted")
            if self.model:
                self.save_model()
            return 1

        except Exception as e:
            print(f"[ERROR] {type(e).__name__}: {e}")
            import traceback
            traceback.print_exc()
            return 1


def main():
    trainer = TeethTrainer()
    return trainer.run()


if __name__ == "__main__":
    sys.exit(main())
