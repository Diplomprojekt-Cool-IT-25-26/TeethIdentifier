"""
CNN Model Architecture for Teeth vs Gingiva Classification.

This module defines the TeethClassifier class, which creates a simple
convolutional neural network for binary classification of dental scan patches.

Architecture: TeethNet
- 3 Convolutional blocks (32, 64, 128 filters)
- Batch normalization for stability
- Dropout for regularization
- 2 Dense layers (256, 64 units)
- Sigmoid output for binary classification

Usage:
    from src.model import TeethClassifier

    config = load_config('config.yaml')
    classifier = TeethClassifier(config)
    model = classifier.build_model()
    model.summary()
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from typing import Dict, Any


class TeethClassifier:
    """
    Convolutional Neural Network for binary classification of teeth vs gingiva.

    Attributes:
        config (dict): Configuration dictionary with model hyperparameters
        model (keras.Model): Compiled Keras model (None until build_model() is called)
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the TeethClassifier.

        Args:
            config: Configuration dictionary containing model architecture and
                   training parameters. Expected keys:
                   - model.input_shape: Input image shape [height, width, channels]
                   - model.conv_filters: List of filter counts for Conv2D layers
                   - model.dense_units: List of units for Dense layers
                   - model.dropout_rates: List of dropout rates
                   - model.kernel_size: Kernel size for Conv2D
                   - model.pool_size: Pool size for MaxPooling2D
                   - model.activation: Activation function
                   - model.final_activation: Output activation (sigmoid)
                   - training.learning_rate: Learning rate for optimizer
                   - training.optimizer: Optimizer name
                   - training.loss: Loss function
                   - training.metrics: List of metric names
        """
        self.config = config
        self.model = None

    def build_model(self) -> keras.Model:
        """
        Build and compile the CNN model.

        Architecture:
            Input (100, 100, 3)
            → Rescaling layer (normalize to [0, 1])
            → Conv Block 1: Conv2D(32) → BatchNorm → MaxPool → Dropout(0.25)
            → Conv Block 2: Conv2D(64) → BatchNorm → MaxPool → Dropout(0.25)
            → Conv Block 3: Conv2D(128) → BatchNorm → MaxPool → Dropout(0.3)
            → Flatten
            → Dense(256) → BatchNorm → Dropout(0.5)
            → Dense(64) → Dropout(0.4)
            → Dense(1, sigmoid)

        Returns:
            Compiled Keras model ready for training
        """
        # Extract configuration
        input_shape = self.config['model']['input_shape']
        conv_filters = self.config['model']['conv_filters']
        dense_units = self.config['model']['dense_units']
        dropout_rates = self.config['model']['dropout_rates']
        kernel_size = self.config['model']['kernel_size']
        pool_size = self.config['model']['pool_size']
        activation = self.config['model']['activation']
        use_batch_norm = self.config['model'].get('use_batch_norm', True)

        # Build model using Sequential API
        model = keras.Sequential(name=self.config['model']['name'])

        # Input layer
        model.add(layers.Input(shape=input_shape, name='input'))

        # Rescaling layer (normalize uint8 [0-255] to float32 [0-1])
        model.add(layers.Rescaling(1./255, name='rescaling'))

        # Data augmentation layers (run on GPU during training)
        if self.config.get('augmentation', {}).get('enabled', False):
            aug_config = self.config['augmentation']

            # Random flip
            if aug_config.get('horizontal_flip', False):
                model.add(layers.RandomFlip('horizontal'))
            if aug_config.get('vertical_flip', False):
                model.add(layers.RandomFlip('vertical'))

            # Random rotation
            rotation_range = aug_config.get('rotation_range', 0)
            if rotation_range > 0:
                model.add(layers.RandomRotation(
                    factor=rotation_range / 360.0,
                    fill_mode='nearest'
                ))

            # Random zoom
            zoom_range = aug_config.get('zoom_range', 0)
            if zoom_range > 0:
                model.add(layers.RandomZoom(
                    height_factor=(-zoom_range, zoom_range),
                    fill_mode='nearest'
                ))

            # Random brightness
            brightness_range = aug_config.get('brightness_range', None)
            if brightness_range:
                model.add(layers.RandomBrightness(
                    factor=(brightness_range[0] - 1.0, brightness_range[1] - 1.0)
                ))

        # Convolutional Block 1
        model.add(layers.Conv2D(
            filters=conv_filters[0],
            kernel_size=(kernel_size, kernel_size),
            padding='same',
            activation=activation,
            name='conv2d_1'
        ))
        if use_batch_norm:
            model.add(layers.BatchNormalization(name='batch_norm_1'))
        model.add(layers.MaxPooling2D(
            pool_size=(pool_size, pool_size),
            name='max_pool_1'
        ))
        model.add(layers.Dropout(dropout_rates[0], name='dropout_1'))

        # Convolutional Block 2
        model.add(layers.Conv2D(
            filters=conv_filters[1],
            kernel_size=(kernel_size, kernel_size),
            padding='same',
            activation=activation,
            name='conv2d_2'
        ))
        if use_batch_norm:
            model.add(layers.BatchNormalization(name='batch_norm_2'))
        model.add(layers.MaxPooling2D(
            pool_size=(pool_size, pool_size),
            name='max_pool_2'
        ))
        model.add(layers.Dropout(dropout_rates[1], name='dropout_2'))

        # Convolutional Block 3
        model.add(layers.Conv2D(
            filters=conv_filters[2],
            kernel_size=(kernel_size, kernel_size),
            padding='same',
            activation=activation,
            name='conv2d_3'
        ))
        if use_batch_norm:
            model.add(layers.BatchNormalization(name='batch_norm_3'))
        model.add(layers.MaxPooling2D(
            pool_size=(pool_size, pool_size),
            name='max_pool_3'
        ))
        model.add(layers.Dropout(dropout_rates[2], name='dropout_3'))

        # Flatten
        model.add(layers.Flatten(name='flatten'))

        # Dense Block 1
        model.add(layers.Dense(
            units=dense_units[0],
            activation=activation,
            name='dense_1'
        ))
        if use_batch_norm:
            model.add(layers.BatchNormalization(name='batch_norm_4'))
        model.add(layers.Dropout(dropout_rates[3], name='dropout_4'))

        # Dense Block 2
        model.add(layers.Dense(
            units=dense_units[1],
            activation=activation,
            name='dense_2'
        ))
        model.add(layers.Dropout(dropout_rates[4], name='dropout_5'))

        # Output layer (binary classification)
        model.add(layers.Dense(
            units=1,
            activation=self.config['model']['final_activation'],
            name='output'
        ))

        # Compile model
        self._compile_model(model)

        self.model = model
        return model

    def _compile_model(self, model: keras.Model) -> None:
        """
        Compile the model with optimizer, loss, and metrics.

        Args:
            model: Keras model to compile
        """
        # Get optimizer
        optimizer_name = self.config['training']['optimizer']
        learning_rate = self.config['training']['learning_rate']

        if optimizer_name.lower() == 'adam':
            optimizer = keras.optimizers.Adam(learning_rate=learning_rate)
        elif optimizer_name.lower() == 'sgd':
            optimizer = keras.optimizers.SGD(learning_rate=learning_rate)
        elif optimizer_name.lower() == 'rmsprop':
            optimizer = keras.optimizers.RMSprop(learning_rate=learning_rate)
        else:
            raise ValueError(f"Unknown optimizer: {optimizer_name}")

        # Get loss
        loss = self.config['training']['loss']

        # Get metrics
        metric_names = self.config['training']['metrics']
        metrics = []

        for metric_name in metric_names:
            if metric_name == 'accuracy':
                metrics.append('accuracy')
            elif metric_name == 'precision':
                metrics.append(keras.metrics.Precision(name='precision'))
            elif metric_name == 'recall':
                metrics.append(keras.metrics.Recall(name='recall'))
            elif metric_name == 'auc':
                metrics.append(keras.metrics.AUC(name='auc'))
            else:
                # Try to use metric name as string
                metrics.append(metric_name)

        # Compile
        model.compile(
            optimizer=optimizer,
            loss=loss,
            metrics=metrics
        )

    def get_model_summary(self) -> str:
        """
        Get a string representation of the model summary.

        Returns:
            Model summary as string

        Raises:
            ValueError: If model hasn't been built yet
        """
        if self.model is None:
            raise ValueError("Model not built yet. Call build_model() first.")

        # Capture model summary to string
        summary_lines = []
        self.model.summary(print_fn=lambda x: summary_lines.append(x))
        return '\n'.join(summary_lines)

    def save_model(self, filepath: str) -> None:
        """
        Save the model to disk.

        Args:
            filepath: Path where model should be saved (e.g., 'models/teeth_classifier.keras')

        Raises:
            ValueError: If model hasn't been built yet
        """
        if self.model is None:
            raise ValueError("Model not built yet. Call build_model() first.")

        self.model.save(filepath)
        print(f"Model saved to: {filepath}")

    def load_model(self, filepath: str) -> keras.Model:
        """
        Load a saved model from disk.

        Args:
            filepath: Path to saved model file

        Returns:
            Loaded Keras model
        """
        self.model = keras.models.load_model(filepath)
        print(f"Model loaded from: {filepath}")
        return self.model

    def count_parameters(self) -> Dict[str, int]:
        """
        Count the number of parameters in the model.

        Returns:
            Dictionary with 'trainable' and 'non_trainable' parameter counts

        Raises:
            ValueError: If model hasn't been built yet
        """
        if self.model is None:
            raise ValueError("Model not built yet. Call build_model() first.")

        trainable = sum([tf.size(w).numpy() for w in self.model.trainable_weights])
        non_trainable = sum([tf.size(w).numpy() for w in self.model.non_trainable_weights])

        return {
            'trainable': int(trainable),
            'non_trainable': int(non_trainable),
            'total': int(trainable + non_trainable)
        }


def main():
    """Example usage of TeethClassifier."""
    import yaml

    # Load configuration
    with open('config.yaml', 'r') as f:
        config = yaml.safe_load(f)

    # Create and build model
    print("Creating TeethClassifier...")
    classifier = TeethClassifier(config)
    model = classifier.build_model()

    # Print summary
    print("\n" + "=" * 70)
    print("Model Summary")
    print("=" * 70)
    model.summary()

    # Count parameters
    params = classifier.count_parameters()
    print("\n" + "=" * 70)
    print("Parameter Count")
    print("=" * 70)
    print(f"  Trainable:     {params['trainable']:,}")
    print(f"  Non-trainable: {params['non_trainable']:,}")
    print(f"  Total:         {params['total']:,}")
    print("=" * 70)


if __name__ == "__main__":
    main()
