"""
TeethIdentifier source code package.

This package contains the neural network training pipeline for
binary classification of teeth vs gingiva from 3D dental scans.
"""

__version__ = "1.0.0"
__author__ = "TeethIdentifier Team"

from .model import TeethClassifier
from .data_loader import TeethDataLoader

__all__ = ['TeethClassifier', 'TeethDataLoader']
