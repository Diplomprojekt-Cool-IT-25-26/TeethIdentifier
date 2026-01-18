"""
Data Loading Module for TeethIdentifier.

This module handles loading training data from pickle files and creating
TensorFlow datasets with proper preprocessing and batching.

Expected pickle file format:
    {
        'images': np.ndarray,      # Shape: (n_samples, 100, 100, 3), dtype: uint8
        'labels': np.ndarray,      # Shape: (n_samples,), dtype: int, values: {0, 1}
        'patient_ids': List[str],  # Patient identifiers
        'jaws': List[str],         # 'upper' or 'lower'
        'point_indices': np.ndarray,    # Original vertex indices
        'point_coords': np.ndarray      # 3D coordinates
    }

Usage:
    from src.data_loader import TeethDataLoader

    config = load_config('config.yaml')
    loader = TeethDataLoader(config)
    train_dataset, train_samples = loader.load_train_data()
    val_dataset, val_samples = loader.load_val_data()
"""

import os
import pickle
import numpy as np
import tensorflow as tf
from typing import Dict, Any, Tuple, Optional
from pathlib import Path
from tensorflow.keras import layers


class TeethDataLoader:
    """
    Data loader for teeth vs gingiva classification.

    Handles loading pickle files and creating TensorFlow datasets with
    proper batching, shuffling, and prefetching for optimal performance.
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the data loader.

        Args:
            config: Configuration dictionary with paths and training settings
        """
        self.config = config
        self.training_data_dir = Path(config['paths']['training_data_dir'])

        # Build data augmentation model if enabled
        if config.get('augmentation', {}).get('enabled', False):
            self.augmentation_model = self._build_augmentation_model()
        else:
            self.augmentation_model = None

    def _build_augmentation_model(self) -> tf.keras.Sequential:
        """
        Build data augmentation model using TensorFlow layers.

        Returns:
            Sequential model with augmentation layers
        """
        aug_config = self.config['augmentation']

        aug_layers = []

        # Random flip
        if aug_config.get('horizontal_flip', False):
            aug_layers.append(layers.RandomFlip('horizontal'))
        if aug_config.get('vertical_flip', False):
            aug_layers.append(layers.RandomFlip('vertical'))

        # Random rotation
        rotation_range = aug_config.get('rotation_range', 0)
        if rotation_range > 0:
            aug_layers.append(layers.RandomRotation(
                factor=rotation_range / 360.0,  # Convert degrees to fraction
                fill_mode='nearest'
            ))

        # Random zoom
        zoom_range = aug_config.get('zoom_range', 0)
        if zoom_range > 0:
            aug_layers.append(layers.RandomZoom(
                height_factor=(-zoom_range, zoom_range),
                fill_mode='nearest'
            ))

        # Random brightness
        brightness_range = aug_config.get('brightness_range', None)
        if brightness_range:
            aug_layers.append(layers.RandomBrightness(
                factor=(brightness_range[0] - 1.0, brightness_range[1] - 1.0)
            ))

        return tf.keras.Sequential(aug_layers, name='data_augmentation')

    def _apply_augmentation(self, image, label):
        """
        Apply data augmentation to a single image.

        Args:
            image: Input image tensor
            label: Corresponding label

        Returns:
            Augmented image and unchanged label
        """
        if self.augmentation_model is not None:
            image = self.augmentation_model(image, training=True)
        return image, label

    def load_pickle(self, pickle_path: str) -> Dict[str, Any]:
        """
        Load a pickle file containing training data.

        Args:
            pickle_path: Path to pickle file

        Returns:
            Dictionary with keys: 'images', 'labels', 'patient_ids', 'jaws'

        Raises:
            FileNotFoundError: If pickle file doesn't exist
            ValueError: If pickle file has incorrect format
        """
        if not os.path.exists(pickle_path):
            raise FileNotFoundError(f"Pickle file not found: {pickle_path}")

        print(f"Loading data from: {pickle_path}")
        with open(pickle_path, 'rb') as f:
            data = pickle.load(f)

        # Validate format
        required_keys = ['images', 'labels']
        for key in required_keys:
            if key not in data:
                raise ValueError(f"Pickle file missing required key: {key}")

        # Print statistics
        n_samples = len(data['labels'])
        n_gingiva = np.sum(data['labels'] == 0)
        n_teeth = np.sum(data['labels'] == 1)

        print(f"  Total samples: {n_samples:,}")
        print(f"  Gingiva: {n_gingiva:,} ({n_gingiva/n_samples*100:.1f}%)")
        print(f"  Teeth: {n_teeth:,} ({n_teeth/n_samples*100:.1f}%)")

        return data

    def create_tf_dataset(
        self,
        images: np.ndarray,
        labels: np.ndarray,
        is_training: bool = False
    ) -> tf.data.Dataset:
        """
        Create a TensorFlow dataset from numpy arrays.

        Args:
            images: Image array (n_samples, height, width, channels)
            labels: Label array (n_samples,)
            is_training: If True, shuffle and repeat dataset

        Returns:
            tf.data.Dataset configured for training or evaluation
        """
        batch_size = self.config['training']['batch_size']

        # Keep images as uint8 (0-255) - model expects this range
        # Convert to float32 but don't normalize
        images = images.astype(np.float32)

        # Create dataset
        dataset = tf.data.Dataset.from_tensor_slices((images, labels))

        # Cache before shuffling for better performance
        dataset = dataset.cache()

        # Shuffle if training
        if is_training:
            # Shuffle with buffer size
            buffer_size = min(10000, len(labels))
            dataset = dataset.shuffle(buffer_size=buffer_size, reshuffle_each_iteration=True)

        # Repeat indefinitely for multiple epochs (both train and val need this)
        dataset = dataset.repeat()

        # Batch
        dataset = dataset.batch(batch_size)

        # NOTE: Data augmentation is now in the model (runs on GPU)
        # Not applied here in the data pipeline

        # Prefetch for performance
        dataset = dataset.prefetch(buffer_size=tf.data.AUTOTUNE)

        return dataset

    def load_train_data(self) -> Tuple[tf.data.Dataset, int]:
        """
        Load training dataset.

        Returns:
            Tuple of (dataset, num_samples)

        Raises:
            FileNotFoundError: If train.pkl doesn't exist
        """
        train_path = self.training_data_dir / 'train.pkl'

        if not train_path.exists():
            raise FileNotFoundError(
                f"Training data not found: {train_path}\n"
                f"Run 'python generate_training_data.py' first"
            )

        data = self.load_pickle(str(train_path))
        dataset = self.create_tf_dataset(
            images=data['images'],
            labels=data['labels'],
            is_training=True
        )

        return dataset, len(data['labels'])

    def load_val_data(self) -> Tuple[tf.data.Dataset, int]:
        """
        Load validation dataset.

        Returns:
            Tuple of (dataset, num_samples)

        Raises:
            FileNotFoundError: If val.pkl doesn't exist
        """
        val_path = self.training_data_dir / 'val.pkl'

        if not val_path.exists():
            raise FileNotFoundError(
                f"Validation data not found: {val_path}\n"
                f"Run 'python generate_training_data.py' first"
            )

        data = self.load_pickle(str(val_path))
        dataset = self.create_tf_dataset(
            images=data['images'],
            labels=data['labels'],
            is_training=False
        )

        return dataset, len(data['labels'])

    def load_test_data(self) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
        """
        Load test dataset.

        Returns images and labels as numpy arrays (not tf.data.Dataset)
        for easier evaluation and visualization.

        Returns:
            Tuple of (images, labels, metadata)
            - images: np.ndarray (n_samples, 100, 100, 3)
            - labels: np.ndarray (n_samples,)
            - metadata: dict with patient_ids, jaws, etc.

        Raises:
            FileNotFoundError: If test.pkl doesn't exist
        """
        test_path = self.training_data_dir / 'test.pkl'

        if not test_path.exists():
            raise FileNotFoundError(
                f"Test data not found: {test_path}\n"
                f"Run 'python generate_training_data.py' first"
            )

        data = self.load_pickle(str(test_path))

        # Extract metadata
        metadata = {
            'patient_ids': data.get('patient_ids', []),
            'jaws': data.get('jaws', []),
            'point_indices': data.get('point_indices', []),
            'point_coords': data.get('point_coords', [])
        }

        return data['images'], data['labels'], metadata

    def get_data_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the loaded datasets.

        Returns:
            Dictionary with dataset statistics

        Raises:
            FileNotFoundError: If data files don't exist
        """
        stats = {}

        # Check which datasets exist
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

    def load_all_data(self) -> Tuple[
        Tuple[tf.data.Dataset, int],
        Tuple[tf.data.Dataset, int],
        Tuple[np.ndarray, np.ndarray, Dict]
    ]:
        """
        Load all datasets (train, val, test) at once.

        Returns:
            Tuple of ((train_dataset, train_samples),
                     (val_dataset, val_samples),
                     (test_images, test_labels, test_metadata))

        Raises:
            FileNotFoundError: If any data file doesn't exist
        """
        print("\n" + "=" * 70)
        print("Loading Training Data")
        print("=" * 70 + "\n")

        # Load training data
        print("Loading training dataset...")
        train_dataset, train_samples = self.load_train_data()

        # Load validation data
        print("\nLoading validation dataset...")
        val_dataset, val_samples = self.load_val_data()

        # Load test data
        print("\nLoading test dataset...")
        test_images, test_labels, test_metadata = self.load_test_data()

        print("\n" + "=" * 70)
        print("Data Loading Complete")
        print("=" * 70)
        print(f"  Training:   {train_samples:,} samples")
        print(f"  Validation: {val_samples:,} samples")
        print(f"  Test:       {len(test_labels):,} samples")
        print("=" * 70 + "\n")

        return (
            (train_dataset, train_samples),
            (val_dataset, val_samples),
            (test_images, test_labels, test_metadata)
        )


def main():
    """Example usage of TeethDataLoader."""
    import yaml

    # Load configuration
    with open('config.yaml', 'r') as f:
        config = yaml.safe_load(f)

    # Create data loader
    loader = TeethDataLoader(config)

    # Get statistics
    print("Dataset Statistics:")
    print("=" * 70)
    stats = loader.get_data_stats()

    for split, split_stats in stats.items():
        if split_stats:
            print(f"\n{split.upper()}:")
            print(f"  Samples: {split_stats['num_samples']:,}")
            print(f"  Gingiva: {split_stats['num_gingiva']:,}")
            print(f"  Teeth: {split_stats['num_teeth']:,}")
            print(f"  Patients: {split_stats['num_patients']}")
            print(f"  Image shape: {split_stats['image_shape']}")
        else:
            print(f"\n{split.upper()}: Not found")

    print("\n" + "=" * 70)

    # Try loading datasets
    try:
        print("\nAttempting to load all datasets...")
        (train_dataset, train_samples), (val_dataset, val_samples), \
            (test_images, test_labels, test_metadata) = loader.load_all_data()

        print(f"\n✓ Successfully loaded all datasets!")
        print(f"  Train: {train_samples:,} samples")
        print(f"  Val: {val_samples:,} samples")
        print(f"  Test: {len(test_labels):,} samples")

    except FileNotFoundError as e:
        print(f"\n✗ Data files not found: {e}")
        print("\nGenerate training data first:")
        print("  python generate_training_data.py")


if __name__ == "__main__":
    main()
