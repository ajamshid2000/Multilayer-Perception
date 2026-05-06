"""
Multilayer Perceptron - A complete neural network implementation for École 42.
"""

from .neural_network import NeuralNetwork
from .data_loader import DataLoader, load_data_for_training
from .utils import Metrics, Visualization

__version__ = "1.0.0"
__author__ = "ajamshid(Abdul Rashed jamshidi) - École 42"
__all__ = [
    'NeuralNetwork',
    'DataLoader',
    'load_data_for_training',
    'Metrics',
    'Visualization'
]
