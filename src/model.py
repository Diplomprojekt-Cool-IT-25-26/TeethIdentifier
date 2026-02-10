"""CNN Model Architecture for Teeth vs Gingiva Classification."""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from typing import Dict, Any


class TeethClassifier:
    """Convolutional Neural Network for binary classification of teeth vs gingiva."""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.model = None

    def build_model(self) -> keras.Model:
        input_shape = self.config['model']['input_shape']
        conv_filters = self.config['model']['conv_filters']
        dense_units = self.config['model']['dense_units']
        dropout_rates = self.config['model']['dropout_rates']
        kernel_size = self.config['model']['kernel_size']
        pool_size = self.config['model']['pool_size']
        activation = self.config['model']['activation']
        use_batch_norm = self.config['model'].get('use_batch_norm', True)

        model = keras.Sequential(name=self.config['model']['name'])
        model.add(layers.Input(shape=input_shape, name='input'))
        model.add(layers.Rescaling(1./255, name='rescaling'))

        # Data augmentation (runs on GPU during training)
        if self.config.get('augmentation', {}).get('enabled', False):
            aug_config = self.config['augmentation']
            if aug_config.get('horizontal_flip', False):
                model.add(layers.RandomFlip('horizontal'))
            if aug_config.get('vertical_flip', False):
                model.add(layers.RandomFlip('vertical'))
            rotation_range = aug_config.get('rotation_range', 0)
            if rotation_range > 0:
                model.add(layers.RandomRotation(factor=rotation_range / 360.0, fill_mode='nearest'))
            zoom_range = aug_config.get('zoom_range', 0)
            if zoom_range > 0:
                model.add(layers.RandomZoom(height_factor=(-zoom_range, zoom_range), fill_mode='nearest'))
            brightness_range = aug_config.get('brightness_range', None)
            if brightness_range:
                model.add(layers.RandomBrightness(factor=(brightness_range[0] - 1.0, brightness_range[1] - 1.0)))

        # Conv Block 1
        model.add(layers.Conv2D(conv_filters[0], (kernel_size, kernel_size), padding='same', activation=activation, name='conv2d_1'))
        if use_batch_norm:
            model.add(layers.BatchNormalization(name='batch_norm_1'))
        model.add(layers.MaxPooling2D((pool_size, pool_size), name='max_pool_1'))
        model.add(layers.Dropout(dropout_rates[0], name='dropout_1'))

        # Conv Block 2
        model.add(layers.Conv2D(conv_filters[1], (kernel_size, kernel_size), padding='same', activation=activation, name='conv2d_2'))
        if use_batch_norm:
            model.add(layers.BatchNormalization(name='batch_norm_2'))
        model.add(layers.MaxPooling2D((pool_size, pool_size), name='max_pool_2'))
        model.add(layers.Dropout(dropout_rates[1], name='dropout_2'))

        # Conv Block 3
        model.add(layers.Conv2D(conv_filters[2], (kernel_size, kernel_size), padding='same', activation=activation, name='conv2d_3'))
        if use_batch_norm:
            model.add(layers.BatchNormalization(name='batch_norm_3'))
        model.add(layers.MaxPooling2D((pool_size, pool_size), name='max_pool_3'))
        model.add(layers.Dropout(dropout_rates[2], name='dropout_3'))

        # Dense layers
        model.add(layers.Flatten(name='flatten'))
        model.add(layers.Dense(dense_units[0], activation=activation, name='dense_1'))
        if use_batch_norm:
            model.add(layers.BatchNormalization(name='batch_norm_4'))
        model.add(layers.Dropout(dropout_rates[3], name='dropout_4'))

        model.add(layers.Dense(dense_units[1], activation=activation, name='dense_2'))
        model.add(layers.Dropout(dropout_rates[4], name='dropout_5'))

        # Output
        model.add(layers.Dense(1, activation=self.config['model']['final_activation'], name='output'))

        self._compile_model(model)
        self.model = model
        return model

    def _compile_model(self, model: keras.Model) -> None:
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

        metrics = []
        for metric_name in self.config['training']['metrics']:
            if metric_name == 'accuracy':
                metrics.append('accuracy')
            elif metric_name == 'precision':
                metrics.append(keras.metrics.Precision(name='precision'))
            elif metric_name == 'recall':
                metrics.append(keras.metrics.Recall(name='recall'))
            elif metric_name == 'auc':
                metrics.append(keras.metrics.AUC(name='auc'))
            else:
                metrics.append(metric_name)

        model.compile(optimizer=optimizer, loss=self.config['training']['loss'], metrics=metrics)

    def get_model_summary(self) -> str:
        if self.model is None:
            raise ValueError("Model not built yet")
        summary_lines = []
        self.model.summary(print_fn=lambda x: summary_lines.append(x))
        return '\n'.join(summary_lines)

    def save_model(self, filepath: str) -> None:
        if self.model is None:
            raise ValueError("Model not built yet")
        self.model.save(filepath)

    def load_model(self, filepath: str) -> keras.Model:
        self.model = keras.models.load_model(filepath)
        return self.model

    def count_parameters(self) -> Dict[str, int]:
        if self.model is None:
            raise ValueError("Model not built yet")
        trainable = sum([tf.size(w).numpy() for w in self.model.trainable_weights])
        non_trainable = sum([tf.size(w).numpy() for w in self.model.non_trainable_weights])
        return {'trainable': int(trainable), 'non_trainable': int(non_trainable), 'total': int(trainable + non_trainable)}


def main():
    import yaml
    with open('config.yaml', 'r') as f:
        config = yaml.safe_load(f)

    classifier = TeethClassifier(config)
    model = classifier.build_model()
    params = classifier.count_parameters()
    print(f"TeethNet: {params['total']:,} parameters ({params['trainable']:,} trainable)")


if __name__ == "__main__":
    main()
