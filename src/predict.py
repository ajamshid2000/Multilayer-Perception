"""
Prediction script for Multilayer Perceptron on breast cancer dataset.
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


def binary_cross_entropy(y_true: np.ndarray, probabilities: np.ndarray) -> float:
    positive_prob = probabilities[:, 1]
    y_target = y_true[:, 1]
    return -np.mean(y_target * np.log(positive_prob + 1e-8) + (1 - y_target) * np.log(1 - positive_prob + 1e-8))


def main():
    try:
        with open("../config.json") as f:
            conf = json.load(f)
    except:
        print("config.json does not exist, initializing using default value")
        conf = 0
    
    parser = argparse.ArgumentParser(
        description='Make predictions using a trained Multilayer Perceptron model on breast cancer dataset.'
    )
    parser.add_argument(
        '--model',
        type=str,
        default=conf["output"]["model_path"]
        if conf 
        and "output" in conf 
        and "model_path" in conf["output"] 
        else 'models/trained_model.npy',
        help='Path to save the trained model (default: models/trained_model.npy)'
    )
    parser.add_argument(
        '--data-dir',
        type=str,
        default=conf["data"]["data_dir"] 
        if conf 
        and "data" in conf 
        and "data_dir" in conf["data"] 
        else 'data',
        help='Directory containing dataset files (default: data)'
    )
    parser.add_argument(
        '--num-samples',
        type=int,
        default=10,
        help='Number of samples to predict on (default: 10)'
    )
    parser.add_argument(
        '--show-probabilities',
        action='store_true',
        help='Show prediction probabilities'
    )
        
    args = parser.parse_args()

    print("=" * 60)
    print("Multilayer Perceptron Prediction")
    print("Wisconsin Breast Cancer Dataset")
    print("=" * 60)
    print(f"Model: {args.model}")
    print(f"Data directory: {args.data_dir}")
    print("=" * 60)

    if not os.path.exists(args.model):
        raise FileNotFoundError(f"Trained model not found: {args.model}")

    # Load model
    print("\n[1/3] Loading model...")
    model_data = np.load(args.model, allow_pickle=True).item()
    model = NeuralNetwork(
        layer_sizes=model_data['layer_sizes'],
        learning_rate=model_data['learning_rate'],
        activation=model_data['activation']
    )
    model.weights = model_data['weights']
    model.biases = model_data['biases']

    print(f"  Model loaded from: {args.model}")
    print(f"  Layer sizes: {model_data['layer_sizes']}")

    # Load test data
    print("\n[2/3] Loading breast cancer dataset...")
    X_train, y_train, X_test, y_test = load_data_for_training(args.data_dir)

    print(f"  Test samples: {X_test.shape[0]}")

    # Make predictions
    print("\n[3/3] Making predictions...")
    probabilities = model.predict_proba(X_test)
    predictions = np.argmax(probabilities, axis=1)
    true_labels = np.argmax(y_test, axis=1)

    sample_count = min(args.num_samples, X_test.shape[0])
    correct = np.sum(predictions[:sample_count] == true_labels[:sample_count])
    accuracy = correct / sample_count * 100

    bce = binary_cross_entropy(y_test[:sample_count], probabilities[:sample_count])
    print(f"Binary cross-entropy on first {sample_count} samples: {bce:.4f}")
    print(f"Accuracy on first {sample_count} samples: {accuracy:.2f}%")

    if args.show_probabilities:
        for i in range(sample_count):
            label_names = {0: 'Benign', 1: 'Malignant'}
            print(f"\nSample {i+1}")
            print(f"  True: {label_names[true_labels[i]]}")
            print(f"  Pred: {label_names[predictions[i]]}")
            print(f"  Probabilities: B={probabilities[i][0]:.4f}, M={probabilities[i][1]:.4f}")

    print("\n" + "=" * 60)


if __name__ == '__main__':
    main()
