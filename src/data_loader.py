"""Data Loading Module for TeethIdentifier."""

import os
import pickle
import numpy as np
import tensorflow as tf
from typing import Dict, Any, Tuple
from pathlib import Path
from tensorflow.keras import layers


class TeethDataLoader:
    """Data loader for teeth vs gingiva classification."""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.training_data_dir = Path(config['paths']['training_data_dir'])

        if config.get('augmentation', {}).get('enabled', False):
            self.augmentation_model = self._build_augmentation_model()
        else:
            self.augmentation_model = None

    def _build_augmentation_model(self) -> tf.keras.Sequential:
        aug_config = self.config['augmentation']
        aug_layers = []

        if aug_config.get('horizontal_flip', False):
            aug_layers.append(layers.RandomFlip('horizontal'))
        if aug_config.get('vertical_flip', False):
            aug_layers.append(layers.RandomFlip('vertical'))

        rotation_range = aug_config.get('rotation_range', 0)
        if rotation_range > 0:
            aug_layers.append(layers.RandomRotation(factor=rotation_range / 360.0, fill_mode='nearest'))

        zoom_range = aug_config.get('zoom_range', 0)
        if zoom_range > 0:
            aug_layers.append(layers.RandomZoom(height_factor=(-zoom_range, zoom_range), fill_mode='nearest'))

        brightness_range = aug_config.get('brightness_range', None)
        if brightness_range:
            aug_layers.append(layers.RandomBrightness(factor=(brightness_range[0] - 1.0, brightness_range[1] - 1.0)))

        return tf.keras.Sequential(aug_layers, name='data_augmentation')

    def load_pickle(self, pickle_path: str) -> Dict[str, Any]:
        if not os.path.exists(pickle_path):
            raise FileNotFoundError(f"Pickle file not found: {pickle_path}")

        with open(pickle_path, 'rb') as f:
            data = pickle.load(f)

        for key in ['images', 'labels']:
            if key not in data:
                raise ValueError(f"Pickle file missing key: {key}")

        n_samples = len(data['labels'])
        n_gingiva = np.sum(data['labels'] == 0)
        n_teeth = np.sum(data['labels'] == 1)
        print(f"Loaded {Path(pickle_path).name}: {n_samples:,} samples ({n_gingiva:,} gingiva, {n_teeth:,} teeth)")

        return data

    def create_tf_dataset(self, images: np.ndarray, labels: np.ndarray,
                         is_training: bool = False) -> tf.data.Dataset:
        batch_size = self.config['training']['batch_size']
        images = images.astype(np.float32)
        dataset = tf.data.Dataset.from_tensor_slices((images, labels))
        dataset = dataset.cache()

        if is_training:
            buffer_size = min(10000, len(labels))
            dataset = dataset.shuffle(buffer_size=buffer_size, reshuffle_each_iteration=True)

        dataset = dataset.repeat()
        dataset = dataset.batch(batch_size)
        dataset = dataset.prefetch(buffer_size=tf.data.AUTOTUNE)
        return dataset

    def load_train_data(self) -> Tuple[tf.data.Dataset, int]:
        train_path = self.training_data_dir / 'train.pkl'
        if not train_path.exists():
            raise FileNotFoundError(f"Training data not found: {train_path}")

        data = self.load_pickle(str(train_path))
        dataset = self.create_tf_dataset(images=data['images'], labels=data['labels'], is_training=True)
        return dataset, len(data['labels'])

    def load_val_data(self) -> Tuple[tf.data.Dataset, int]:
        val_path = self.training_data_dir / 'val.pkl'
        if not val_path.exists():
            raise FileNotFoundError(f"Validation data not found: {val_path}")

        data = self.load_pickle(str(val_path))
        dataset = self.create_tf_dataset(images=data['images'], labels=data['labels'], is_training=False)
        return dataset, len(data['labels'])

    def load_test_data(self) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
        test_path = self.training_data_dir / 'test.pkl'
        if not test_path.exists():
            raise FileNotFoundError(f"Test data not found: {test_path}")

        data = self.load_pickle(str(test_path))
        metadata = {
            'patient_ids': data.get('patient_ids', []),
            'jaws': data.get('jaws', []),
            'point_indices': data.get('point_indices', []),
            'point_coords': data.get('point_coords', [])
        }
        return data['images'], data['labels'], metadata

    def get_data_stats(self) -> Dict[str, Any]:
        stats = {}
        for split in ['train', 'val', 'test']:
            pkl_path = self.training_data_dir / f'{split}.pkl'
            if pkl_path.exists():
                data = self.load_pickle(str(pkl_path))
                stats[split] = {
                    'num_samples': len(data['labels']),
                    'num_gingiva': int(np.sum(data['labels'] == 0)),
                    'num_teeth': int(np.sum(data['labels'] == 1)),
                    'image_shape': data['images'].shape[1:],
                    'num_patients': len(set(data.get('patient_ids', [])))
                }
            else:
                stats[split] = None
        return stats


def main():
    import yaml

    with open('config.yaml', 'r') as f:
        config = yaml.safe_load(f)

    loader = TeethDataLoader(config)
    stats = loader.get_data_stats()

    for split, split_stats in stats.items():
        if split_stats:
            print(f"{split.upper()}: {split_stats['num_samples']:,} samples "
                  f"({split_stats['num_gingiva']:,} gingiva, {split_stats['num_teeth']:,} teeth)")
        else:
            print(f"{split.upper()}: Not found")


if __name__ == "__main__":
    main()
