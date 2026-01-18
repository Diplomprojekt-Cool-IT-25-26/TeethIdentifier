"""
Continue training from a saved model checkpoint.

Loads the existing model and continues training with reduced learning rate
for fine-tuning to push accuracy higher.
"""

import yaml
import tensorflow as tf
from tensorflow import keras
from pathlib import Path
from datetime import datetime
from src.data_loader import TeethDataLoader


def continue_training(
    model_path: str,
    config_path: str = 'config.yaml',
    additional_epochs: int = 50,
    learning_rate: float = 0.0001
):
    """
    Continue training from a saved model.

    Args:
        model_path: Path to saved .keras model
        config_path: Path to config.yaml
        additional_epochs: Number of additional epochs to train
        learning_rate: New learning rate (should be lower than initial)
    """
    print("=" * 70)
    print("Continue Training from Checkpoint")
    print("=" * 70)
    print(f"Model: {model_path}")
    print(f"Additional epochs: {additional_epochs}")
    print(f"Learning rate: {learning_rate}")
    print("=" * 70 + "\n")

    # Load config
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    # Setup GPU
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            print(f"[OK] GPU memory growth enabled")
        except RuntimeError as e:
            print(f"[WARNING] {e}")

    # Enable mixed precision
    if config['training'].get('use_mixed_precision', True):
        policy = keras.mixed_precision.Policy('mixed_float16')
        keras.mixed_precision.set_global_policy(policy)
        print(f"[OK] Mixed precision enabled: {policy.name}\n")

    # Load data
    print("Loading data...")
    loader = TeethDataLoader(config)
    train_dataset, train_samples = loader.load_train_data()
    val_dataset, val_samples = loader.load_val_data()
    print(f"  Train: {train_samples:,} samples")
    print(f"  Val: {val_samples:,} samples\n")

    # Load existing model
    print(f"Loading model from: {model_path}")
    model = keras.models.load_model(model_path)
    print(f"[OK] Model loaded\n")

    # Recompile with new learning rate
    print(f"Recompiling with learning rate: {learning_rate}")
    optimizer = keras.optimizers.Adam(learning_rate=learning_rate)

    model.compile(
        optimizer=optimizer,
        loss=config['training']['loss'],
        metrics=[
            'accuracy',
            keras.metrics.Precision(name='precision'),
            keras.metrics.Recall(name='recall'),
            keras.metrics.AUC(name='auc')
        ]
    )
    print(f"[OK] Model recompiled\n")

    # Setup callbacks
    callbacks = []

    # ModelCheckpoint
    checkpoint_dir = Path(config['paths']['model_dir']) / 'checkpoints'
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    callbacks.append(keras.callbacks.ModelCheckpoint(
        filepath=str(checkpoint_dir / 'model_epoch_{epoch:02d}_val_acc_{val_accuracy:.4f}.keras'),
        monitor='val_accuracy',
        save_best_only=True,
        mode='max',
        verbose=1
    ))

    # EarlyStopping (more patient for fine-tuning)
    callbacks.append(keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=15,  # More patient
        restore_best_weights=True,
        verbose=1
    ))

    # ReduceLROnPlateau
    callbacks.append(keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=7,
        min_lr=0.00001,
        verbose=1
    ))

    # TensorBoard
    log_dir = Path(config['paths']['logs_dir']) / datetime.now().strftime('%Y%m%d-%H%M%S-continue')
    callbacks.append(keras.callbacks.TensorBoard(
        log_dir=str(log_dir),
        histogram_freq=1
    ))

    # CSVLogger
    csv_path = Path(config['paths']['logs_dir']) / 'training_continue.csv'
    callbacks.append(keras.callbacks.CSVLogger(
        filename=str(csv_path),
        separator=',',
        append=True  # Append to existing log
    ))

    # Calculate steps
    batch_size = config['training']['batch_size']
    steps_per_epoch = train_samples // batch_size
    validation_steps = val_samples // batch_size

    print("Training Configuration:")
    print(f"  Additional epochs: {additional_epochs}")
    print(f"  Batch size: {batch_size}")
    print(f"  Steps per epoch: {steps_per_epoch}")
    print(f"  Learning rate: {learning_rate}")
    print()

    # Continue training
    print("=" * 70)
    print("Resuming training...")
    print("=" * 70 + "\n")

    start_time = datetime.now()

    history = model.fit(
        train_dataset,
        epochs=additional_epochs,
        steps_per_epoch=steps_per_epoch,
        validation_data=val_dataset,
        validation_steps=validation_steps,
        callbacks=callbacks,
        verbose=1
    )

    end_time = datetime.now()
    training_time = (end_time - start_time).total_seconds() / 60.0

    print("\n" + "=" * 70)
    print("Training Complete!")
    print("=" * 70)
    print(f"  Training time: {training_time:.1f} minutes")
    print(f"  Final train accuracy: {history.history['accuracy'][-1]:.4f}")
    print(f"  Final val accuracy: {history.history['val_accuracy'][-1]:.4f}")
    print(f"  Best val accuracy: {max(history.history['val_accuracy']):.4f}")
    print("=" * 70 + "\n")

    # Save final model (as v2 to preserve original)
    final_model_path = Path(config['paths']['model_dir']) / 'teeth_classifier_v2.keras'
    print(f"Saving final model to: {final_model_path}")
    model.save(str(final_model_path))
    print(f"[OK] Model saved (v2 - original v1 preserved)\n")

    print("=" * 70)
    print("Done!")
    print("=" * 70)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Continue training from checkpoint')
    parser.add_argument('--model', type=str,
                        default='ml_outputs/models/teeth_classifier.keras',
                        help='Path to saved model')
    parser.add_argument('--epochs', type=int, default=50,
                        help='Additional epochs to train')
    parser.add_argument('--lr', type=float, default=0.0001,
                        help='Learning rate (lower than original for fine-tuning)')

    args = parser.parse_args()

    continue_training(
        model_path=args.model,
        additional_epochs=args.epochs,
        learning_rate=args.lr
    )
