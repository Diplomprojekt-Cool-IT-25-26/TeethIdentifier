"""
Training Script for TeethIdentifier Neural Network.

This script handles the complete training pipeline:
- GPU configuration
- Data loading
- Model building
- Training with callbacks (early stopping, LR scheduling, checkpointing)
- Training history visualization and saving

Usage:
    python src/train.py

Expected output:
    models/teeth_classifier.keras  - Final trained model
    models/checkpoints/           - Training checkpoints
    models/training_history.json  - Training metrics
    models/training_log.csv       - Epoch-by-epoch log
    logs/                         - TensorBoard logs
"""

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

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.model import TeethClassifier
from src.data_loader import TeethDataLoader


class TeethTrainer:
    """
    Training pipeline for TeethNet CNN.

    Handles GPU setup, data loading, model training with callbacks,
    and result visualization.
    """

    def __init__(self, config_path: str = 'config.yaml'):
        """
        Initialize the trainer.

        Args:
            config_path: Path to configuration YAML file
        """
        self.config_path = config_path
        self.config = self.load_config()
        self.model = None
        self.history = None

        # Paths
        self.model_dir = Path(self.config['paths']['model_dir'])
        self.logs_dir = Path(self.config['paths']['logs_dir'])
        self.results_dir = Path(self.config['paths'].get('results_dir', 'results'))

        # Create directories
        self.model_dir.mkdir(parents=True, exist_ok=True)
        (self.model_dir / 'checkpoints').mkdir(parents=True, exist_ok=True)
        self.logs_dir.mkdir(parents=True, exist_ok=True)
        self.results_dir.mkdir(parents=True, exist_ok=True)

    def load_config(self) -> Dict[str, Any]:
        """
        Load configuration from YAML file.

        Returns:
            Configuration dictionary
        """
        print(f"Loading configuration from: {self.config_path}")
        with open(self.config_path, 'r') as f:
            config = yaml.safe_load(f)
        return config

    def setup_gpu(self) -> None:
        """
        Configure TensorFlow GPU settings.

        Sets up:
        - Memory growth (allocate as needed)
        - Mixed precision training (2× speedup)
        - Device placement
        """
        print("\n" + "=" * 70)
        print("GPU Configuration")
        print("=" * 70)

        # Check for GPUs
        gpus = tf.config.list_physical_devices('GPU')
        print(f"GPUs detected: {len(gpus)}")

        if gpus:
            try:
                # Enable memory growth
                for gpu in gpus:
                    tf.config.experimental.set_memory_growth(gpu, True)
                print("[OK] Memory growth enabled")

                # Print GPU details
                for i, gpu in enumerate(gpus):
                    print(f"  GPU {i}: {gpu.name}")

            except RuntimeError as e:
                print(f"[WARNING] Could not configure GPU: {e}")

        else:
            print("[WARNING] No GPU detected. Training will run on CPU (slow).")
            if not self.config['gpu'].get('force_gpu', False):
                response = input("Continue anyway? (y/n): ")
                if response.lower() != 'y':
                    print("Exiting...")
                    sys.exit(1)

        # Mixed precision training
        if self.config['training'].get('use_mixed_precision', False):
            policy = tf.keras.mixed_precision.Policy('mixed_float16')
            tf.keras.mixed_precision.set_global_policy(policy)
            print(f"[OK] Mixed precision enabled: {policy.name}")
            print("  (Expected 2× speedup on RTX 3070)")

        # Set random seeds for reproducibility
        seed = self.config.get('random_seed', 42)
        np.random.seed(seed)
        tf.random.set_seed(seed)
        print(f"[OK] Random seed set: {seed}")

        print("=" * 70 + "\n")

    def prepare_data(self):
        """
        Load training and validation datasets.

        Returns:
            Tuple of (train_dataset, train_samples, val_dataset, val_samples)
        """
        print("=" * 70)
        print("Data Loading")
        print("=" * 70 + "\n")

        loader = TeethDataLoader(self.config)

        # Load datasets
        train_dataset, train_samples = loader.load_train_data()
        val_dataset, val_samples = loader.load_val_data()

        print("\n[OK] Data loading complete")
        print("=" * 70 + "\n")

        return train_dataset, train_samples, val_dataset, val_samples

    def build_model(self) -> keras.Model:
        """
        Build and compile the CNN model.

        Returns:
            Compiled Keras model
        """
        print("=" * 70)
        print("Model Building")
        print("=" * 70 + "\n")

        classifier = TeethClassifier(self.config)
        model = classifier.build_model()

        # Print summary
        print("\nModel Summary:")
        print("-" * 70)
        model.summary()

        # Count parameters
        params = classifier.count_parameters()
        print("\nParameter Count:")
        print(f"  Trainable:     {params['trainable']:,}")
        print(f"  Non-trainable: {params['non_trainable']:,}")
        print(f"  Total:         {params['total']:,}")

        print("\n[OK] Model built successfully")
        print("=" * 70 + "\n")

        self.model = model
        return model

    def get_callbacks(self) -> List[keras.callbacks.Callback]:
        """
        Create training callbacks.

        Returns:
            List of Keras callbacks for training
        """
        callbacks = []

        # ModelCheckpoint - save best model
        checkpoint_path = self.model_dir / 'checkpoints' / \
            'model_epoch_{epoch:02d}_val_acc_{val_accuracy:.4f}.keras'

        callbacks.append(keras.callbacks.ModelCheckpoint(
            filepath=str(checkpoint_path),
            monitor=self.config['training']['monitor_metric'],
            save_best_only=self.config['training']['save_best_only'],
            mode='max' if 'acc' in self.config['training']['monitor_metric'] else 'min',
            verbose=1
        ))

        # EarlyStopping
        callbacks.append(keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=self.config['training']['early_stopping_patience'],
            restore_best_weights=True,
            verbose=1
        ))

        # ReduceLROnPlateau
        callbacks.append(keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=self.config['training']['reduce_lr_factor'],
            patience=self.config['training']['reduce_lr_patience'],
            min_lr=self.config['training']['min_learning_rate'],
            verbose=1
        ))

        # TensorBoard
        if self.config['logging'].get('tensorboard', True):
            log_dir = self.logs_dir / datetime.now().strftime("%Y%m%d-%H%M%S")
            callbacks.append(keras.callbacks.TensorBoard(
                log_dir=str(log_dir),
                histogram_freq=self.config['logging'].get('histogram_freq', 1),
                write_graph=True,
                write_images=False
            ))

        # CSVLogger
        if self.config['logging'].get('csv_logging', True):
            csv_path = self.model_dir / 'training_log.csv'
            callbacks.append(keras.callbacks.CSVLogger(
                filename=str(csv_path),
                separator=',',
                append=False
            ))

        return callbacks

    def train(self, train_dataset, train_samples, val_dataset, val_samples) -> None:
        """
        Train the model.

        Args:
            train_dataset: Training tf.data.Dataset
            train_samples: Number of training samples
            val_dataset: Validation tf.data.Dataset
            val_samples: Number of validation samples
        """
        print("=" * 70)
        print("Training")
        print("=" * 70)

        # Calculate steps
        batch_size = self.config['training']['batch_size']
        epochs = self.config['training']['epochs']

        steps_per_epoch = train_samples // batch_size
        validation_steps = val_samples // batch_size

        print(f"\nTraining Configuration:")
        print(f"  Epochs: {epochs}")
        print(f"  Batch size: {batch_size}")
        print(f"  Steps per epoch: {steps_per_epoch}")
        print(f"  Validation steps: {validation_steps}")
        print(f"  Total training steps: {steps_per_epoch * epochs:,}")

        print(f"\nCallbacks:")
        callbacks = self.get_callbacks()
        for callback in callbacks:
            print(f"  - {callback.__class__.__name__}")

        print("\n" + "=" * 70)
        print("Starting training...")
        print("=" * 70 + "\n")

        # Train model
        start_time = datetime.now()

        self.history = self.model.fit(
            train_dataset,
            epochs=epochs,
            steps_per_epoch=steps_per_epoch,
            validation_data=val_dataset,
            validation_steps=validation_steps,
            callbacks=callbacks,
            verbose=1
        )

        end_time = datetime.now()
        training_time = (end_time - start_time).total_seconds()

        print("\n" + "=" * 70)
        print("Training Complete!")
        print("=" * 70)
        print(f"  Training time: {training_time/60:.1f} minutes")
        print(f"  Final train accuracy: {self.history.history['accuracy'][-1]:.4f}")
        print(f"  Final val accuracy: {self.history.history['val_accuracy'][-1]:.4f}")
        print("=" * 70 + "\n")

    def save_model(self) -> None:
        """Save the final trained model."""
        model_path = self.model_dir / 'teeth_classifier.keras'
        self.model.save(str(model_path))
        print(f"[OK] Model saved to: {model_path}")

    def save_history(self) -> None:
        """Save training history to JSON file."""
        history_path = self.model_dir / 'training_history.json'

        # Convert history to serializable format
        history_dict = {
            key: [float(val) for val in values]
            for key, values in self.history.history.items()
        }

        with open(history_path, 'w') as f:
            json.dump(history_dict, f, indent=2)

        print(f"[OK] Training history saved to: {history_path}")

    def plot_training_history(self) -> None:
        """Plot and save training history curves."""
        if self.history is None:
            print("[WARNING] No training history to plot")
            return

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

        # Plot accuracy
        ax1.plot(self.history.history['accuracy'], label='Train Accuracy', linewidth=2)
        ax1.plot(self.history.history['val_accuracy'], label='Val Accuracy', linewidth=2)
        ax1.set_title('Model Accuracy', fontsize=14, fontweight='bold')
        ax1.set_xlabel('Epoch', fontsize=12)
        ax1.set_ylabel('Accuracy', fontsize=12)
        ax1.legend(fontsize=10)
        ax1.grid(True, alpha=0.3)

        # Plot loss
        ax2.plot(self.history.history['loss'], label='Train Loss', linewidth=2)
        ax2.plot(self.history.history['val_loss'], label='Val Loss', linewidth=2)
        ax2.set_title('Model Loss', fontsize=14, fontweight='bold')
        ax2.set_xlabel('Epoch', fontsize=12)
        ax2.set_ylabel('Loss', fontsize=12)
        ax2.legend(fontsize=10)
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()

        # Save figure
        plot_path = self.results_dir / 'training_history.png'
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        print(f"[OK] Training plots saved to: {plot_path}")

        plt.close()

    def run(self) -> None:
        """Run the complete training pipeline."""
        print("\n" + "=" * 70)
        print("TeethIdentifier Training Pipeline")
        print("=" * 70)
        print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("=" * 70 + "\n")

        try:
            # Setup
            self.setup_gpu()

            # Load data
            train_dataset, train_samples, val_dataset, val_samples = self.prepare_data()

            # Build model
            self.build_model()

            # Train
            self.train(train_dataset, train_samples, val_dataset, val_samples)

            # Save results
            self.save_model()
            self.save_history()
            self.plot_training_history()

            # Final summary
            print("\n" + "=" * 70)
            print("Training Pipeline Complete!")
            print("=" * 70)
            print("\nGenerated files:")
            print(f"  Model: {self.model_dir / 'teeth_classifier.keras'}")
            print(f"  History: {self.model_dir / 'training_history.json'}")
            print(f"  Plots: {self.results_dir / 'training_history.png'}")
            print(f"  Logs: {self.logs_dir}")
            print("\nNext steps:")
            print("  1. Evaluate: python src/evaluate.py")
            print("  2. Visualize: tensorboard --logdir logs")
            print("  3. Infer: python src/predict.py --obj <file.obj>")
            print("=" * 70 + "\n")

            return 0

        except KeyboardInterrupt:
            print("\n\n[WARNING] Training interrupted by user")
            if self.model:
                save = input("Save current model? (y/n): ")
                if save.lower() == 'y':
                    self.save_model()
            return 1

        except Exception as e:
            print(f"\n\n[ERROR] Training failed")
            print(f"  {type(e).__name__}: {e}")
            import traceback
            traceback.print_exc()
            return 1


def main():
    """Main entry point."""
    trainer = TeethTrainer()
    return trainer.run()


if __name__ == "__main__":
    sys.exit(main())
