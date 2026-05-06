# Multilayer Perceptron for Breast Cancer Classification

A comprehensive implementation of a Multilayer Perceptron (MLP) neural network from scratch using NumPy to classify breast cancer tumors as malignant or benign based on the Wisconsin diagnostic dataset.

## Overview

This project implements a fully-connected Multilayer Perceptron neural network for binary classification of breast cancer data. The implementation demonstrates:

- **Custom neural network** with forward and backward propagation
- **Activation functions** (ReLU, Sigmoid, Tanh)
- **Softmax output layer** for binary classification
- **Mini-batch gradient descent** training
- **Data loading and preprocessing** utilities
- **Model persistence** (save/load)
- **Visualization** tools for training curves

## Features

### Core Components

- **Neural Network Class**: Fully configurable MLP with flexible architecture
- **Forward Propagation**: Efficient computation through network layers
- **Backpropagation**: Gradient computation and weight updates
- **Activation Functions**: ReLU, Sigmoid, and Tanh with derivatives
- **Training Pipeline**: Mini-batch training with validation support

### Additional Features

- **Data Utilities**:
  - Wisconsin diagnostic breast cancer dataset
  - Data standardization
  - One-hot encoding
  - Train-validation splitting

- **Evaluation Metrics**:
  - Binary accuracy
  - Precision, Recall, F1-Score
  - Classification reports
  - Confusion matrices

- **Visualization**:
  - Training history plots (loss and accuracy)
  - Confusion matrices
  - Data distribution plots

## Project Structure

```
Multilayer-Perception/
├── src/
│   ├── __init__.py
│   ├── neural_network.py      # Core MLP implementation
│   ├── data_loader.py         # Data loading and preprocessing
│   ├── train.py              # Training script
│   ├── predict.py            # Prediction script
│   └── utils.py              # Utilities and visualization
├── split_dataset.py          # Dataset splitting utility
├── data/                     # Wisconsin breast cancer dataset
├── models/                   # Saved model directory
├── config.json              # Configuration file
├── requirements.txt         # Project dependencies
└── README.md               # This file
```

## Installation

### Prerequisites

- Python 3.7 or higher
- pip package manager

### Setup

1. **Clone or navigate to the project directory**:
```bash
cd Multilayer-Perception
```

2. **Install dependencies**:
```bash
pip install -r requirements.txt
```

## Quick Start

### 1. Split the Dataset

```bash
python split_dataset.py
```

This creates training and validation CSV files in the `data/` directory.

### 2. Train the Model

```bash
python src/train.py --epochs 100 --batch-size 32 --learning-rate 0.01
```

Available options:
- `--epochs`: Number of training epochs (default: 100)
- `--batch-size`: Batch size for training (default: 32)
- `--learning-rate`: Learning rate (default: 0.01)
- `--layers`: Network layer sizes (default: [30, 16, 16, 2])
- `--activation`: Activation function - relu/sigmoid/tanh (default: relu)
- `--data-dir`: Data directory (default: data)
- `--model-path`: Model save path (default: models/trained_model.npy)

### 3. Make Predictions

```bash
python src/predict.py --model models/trained_model.npy --num-samples 10
```

Available options:
- `--model`: Path to trained model file
- `--data-dir`: Data directory
- `--num-samples`: Number of samples to predict (default: 10)
- `--show-probabilities`: Show prediction probabilities

## Dataset

The project uses the **Wisconsin Diagnostic Breast Cancer (WDBC)** dataset which contains:

- **569 samples** with **30 features** each
- **Binary classification**: Malignant (1) or Benign (0)
- **Features**: Computed from digitized images of fine needle aspirate (FNA) of breast mass
- **Default split**: 80% training, 20% validation

## Usage Examples

Use the available scripts below to split data, train the model, and make predictions.

### Custom Training Script

```python
from src.neural_network import NeuralNetwork
from src.data_loader import load_data_for_training
import numpy as np

# Load data
X_train, y_train, X_valid, y_valid = load_data_for_training('data')

# Create and train model
model = NeuralNetwork(
    layer_sizes=[30, 16, 16, 2],
    learning_rate=0.01,
    activation='relu'
)

history = model.train(
    X_train, y_train,
    epochs=100,
    batch_size=32,
    validation_data=(X_valid, y_valid),
    verbose=True
)

# Save model
model.save('models/my_model.npy')

# Make predictions
predictions = model.predict(X_valid)
probabilities = model.predict_proba(X_valid)
```

## Architecture Details

The default network architecture for breast cancer classification:

- **Input Layer**: 30 features (WDBC dataset)
- **Hidden Layer 1**: 16 neurons with ReLU activation
- **Hidden Layer 2**: 16 neurons with ReLU activation
- **Output Layer**: 2 neurons with Softmax activation (binary classification)

### Training Details

- **Loss Function**: Cross-entropy for multi-class classification
- **Optimizer**: Gradient descent with mini-batches
- **Activation Functions**: Configurable (ReLU, Sigmoid, Tanh)
- **Weight Initialization**: Xavier initialization

## Performance

Target accuracies on validation set:
- Benign classification: > 95%
- Malignant classification: > 95%
- Overall accuracy: > 95%

## File Descriptions

- `neural_network.py`: Core MLP implementation with forward/backward propagation
- `data_loader.py`: Dataset loading and preprocessing for Wisconsin breast cancer data
- `train.py`: Command-line training script with argument parsing
- `predict.py`: Prediction script for making classifications on new data
- `utils.py`: Visualization and evaluation metrics classes

## Key Mathematical Concepts

The implementation covers:

1. **Feedforward propagation**: Computing outputs through network layers
2. **Backpropagation**: Computing gradients using the chain rule
3. **Gradient descent**: Updating weights to minimize loss
4. **Activation functions**: Non-linear transformations (ReLU, Sigmoid, Tanh)
5. **Cross-entropy loss**: Binary classification loss function

## Configuration

Edit `config.json` to customize default training parameters:

```json
{
  "network": {
    "layer_sizes": [30, 16, 16, 2],
    "activation": "relu"
  },
  "training": {
    "epochs": 100,
    "batch_size": 32,
    "learning_rate": 0.01
  },
  "data": {
    "dataset": "breast_cancer",
    "data_dir": "data"
  }
}
```

## Troubleshooting

### Common Issues

1. **Model not found**: Ensure you've trained the model first using `python src/train.py`
2. **Data not found**: Run `python split_dataset.py` to generate train/validation splits
3. **Memory issues**: Reduce batch size or use smaller layer sizes
4. **Poor accuracy**: Try different learning rates, epochs, or network architectures

## Contributing

Contributions are welcome. Please ensure code clarity and proper documentation.

## License

This project is part of the École 42 Machine Learning curriculum.
