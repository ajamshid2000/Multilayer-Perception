"""
Training script for Multilayer Perceptron on Wisconsin breast cancer dataset.
"""

import numpy as np
import argparse
import sys
import os
from pathlib import Path
import json

sys.path.insert(0, str(Path(__file__).parent))

from neural_network import NeuralNetwork
from data_loader import load_data_for_training
from utils import Visualization


def main():
    try:
        f = open("config.json")
        conf = json.load(f)
    except:
        print("config.json does not exist, initializing using default value")
        conf = 0
    
    parser = argparse.ArgumentParser(
        description='Train a Multilayer Perceptron on breast cancer dataset.'
    )
    parser.add_argument(
        '--epochs',
        type=int,
        default=conf["training"]["epochs"] if conf else 100,
        help='Number of training epochs (default: 100)'
    )
    parser.add_argument(
        '--batch-size',
        type=int,
        default=conf["training"]["batch_size"] if conf else 32,
        help='Batch size for training (default: 32)'
    )
    parser.add_argument(
        '--learning-rate',
        type=float,
        default=conf["training"]["learning_rate"] if conf else 0.01,
        help='Learning rate (default: 0.01)'
    )
    parser.add_argument(
        '--layers',
        type=int,
        nargs='+',
        default=conf["network"]["layer_sizes"] if conf else [30, 16, 16, 2],
        help='Layer sizes (default: [30, 16, 16, 2])'
    )
    parser.add_argument(
        '--activation',
        type=str,
        default=conf["network"]["activation"] if conf else 'relu',
        choices=['relu', 'sigmoid', 'tanh'],
        help='Activation function (default: relu)'
    )
    parser.add_argument(
        '--model-path',
        type=str,
        default=conf["output"]["model_path"] if conf else 'models/trained_model.npy',
        help='Path to save the trained model (default: models/trained_model.npy)'
    )
    parser.add_argument(
        '--history-path',
        type=str,
        default=conf["output"]["history_path"] if conf else 'models/training_history.npy',
        help='Path to save the training history (default: models/training_history.npy)'
    )
    parser.add_argument(
        '--plots-dir',
        type=str,
        default=conf["output"]["plots_dir"] if conf else 'models/plots',
        help='Directory to save training plots (default: models/plots)'
    )
    parser.add_argument(
        '--data-dir',
        type=str,
        default=conf["data"]["data_dir"] if conf else 'data',
        help='Directory containing dataset files (default: data)'
    )
    parser.add_argument(
        '--validation-split',
        type=float,
        default=conf["training"]["validation_split"] if conf else 0.2,
        help='Validation split ratio (default: 0.2)'
    )
    parser.add_argument(
        '--random-seed',
        type=int,
        default=42,
        help='Random seed for reproducible splitting (default: 42)'
    )

    if(conf):
        f.close()
    
    args = parser.parse_args()

    print("=" * 60)
    print("Multilayer Perceptron Training")
    print("Wisconsin Breast Cancer Dataset")
    print("=" * 60)
    print(f"Data directory: {args.data_dir}")
    print(f"Model path: {args.model_path}")
    print(f"Layer sizes: {args.layers}")
    print(f"Learning rate: {args.learning_rate}")
    print(f"Epochs: {args.epochs}")
    print("=" * 60)

    os.makedirs(os.path.dirname(args.model_path) or '.', exist_ok=True)
    os.makedirs(os.path.dirname(args.history_path) or '.', exist_ok=True)
    os.makedirs(args.plots_dir, exist_ok=True)
    os.makedirs(args.data_dir, exist_ok=True)

    # Load data
    print("\n[1/4] Loading breast cancer dataset...")
    X_train, y_train, X_test, y_test = load_data_for_training(
        args.data_dir,
        validation_split=args.validation_split,
        random_state=args.random_seed
    )

    print(f"  Training samples: {X_train.shape[0]}")
    print(f"  Validation samples: {X_test.shape[0]}")
    print(f"  Input features: {X_train.shape[1]}")

    # Initialize network
    print("\n[2/4] Initializing network...")
    model = NeuralNetwork(
        layer_sizes=args.layers,
        learning_rate=args.learning_rate,
        activation=args.activation
    )

    # Train model
    print("\n[3/4] Training model...")
    history = model.train(
        X_train,
        y_train,
        epochs=args.epochs,
        batch_size=args.batch_size,
        validation_data=(X_test, y_test),
        verbose=True
    )

    # Evaluate model
    print("\n[4/4] Evaluating model...")
    train_pred, _ = model.forward(X_train)
    val_pred, _ = model.forward(X_test)

    def cross_entropy(targets, predictions):
        return -np.mean(targets * np.log(predictions + 1e-8))

    train_loss = cross_entropy(y_train, train_pred)
    val_loss = cross_entropy(y_test, val_pred)

    train_accuracy = np.mean(np.argmax(train_pred, axis=1) == np.argmax(y_train, axis=1))
    val_accuracy = np.mean(np.argmax(val_pred, axis=1) == np.argmax(y_test, axis=1))

    print(f"\nTraining Results:\n  Loss: {train_loss:.4f}\n  Accuracy: {train_accuracy:.4f}")
    print(f"\nValidation Results:\n  Loss: {val_loss:.4f}\n  Accuracy: {val_accuracy:.4f}")

    # Save model and history
    model.save(args.model_path)
    np.save(args.history_path, history, allow_pickle=True)
    print(f"\nModel saved to: {args.model_path}")
    print(f"History saved to: {args.history_path}")

    # Plot training curves
    history_plot_path = os.path.join(args.plots_dir, 'training_history.png')
    Visualization.plot_training_history(history, save_path=history_plot_path)
    print(f"Training plots saved to: {history_plot_path}")


if __name__ == '__main__':
    main()
