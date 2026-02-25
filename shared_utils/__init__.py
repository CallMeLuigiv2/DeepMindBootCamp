"""
Shared utilities for the Road to ML Expert course.

Install with: pip install -e . (from project root)

Usage:
    from shared_utils.common import set_seed, get_device, count_parameters
    from shared_utils.plotting import plot_training_curves, plot_confusion_matrix
    from shared_utils.data import load_mnist, load_cifar10
    from shared_utils.metrics import accuracy, precision, recall, f1_score
"""

from shared_utils.common import set_seed, get_device, count_parameters
