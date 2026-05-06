"""
Neural Network implementation for Multilayer Perceptron.
Author: École 42
"""

import numpy as np
from typing import List, Tuple, Callable


class NeuralNetwork:
    """
    A fully connected Multilayer Perceptron implementation.
    
    Attributes:
        layers: List of layer sizes (including input and output layers)
        weights: List of weight matrices for each layer
        biases: List of bias vectors for each layer
        learning_rate: Learning rate for gradient descent
        activation: Activation function to use
    """
    
    def __init__(
        self,
        layer_sizes: List[int],
        learning_rate: float = 0.01,
        activation: str = 'relu'
    ):
        """
        Initialize the neural network.
        
        Args:
            layer_sizes: List of integers representing the size of each layer
            learning_rate: Learning rate for gradient descent
            activation: Activation function ('relu', 'sigmoid', 'tanh')
        """
        self.layer_sizes = layer_sizes
        self.learning_rate = learning_rate
        self.activation_name = activation
        self.num_layers = len(layer_sizes)
        
        # Initialize weights and biases
        self.weights = []
        self.biases = []
        self.activation_func = self._get_activation(activation)
        self.activation_derivative = self._get_activation_derivative(activation)
        
        self._initialize_parameters()
    
    def _initialize_parameters(self) -> None:
        """Initialize weights with Xavier initialization and biases with zeros."""
        for i in range(self.num_layers - 1):
            # Xavier initialization
            limit = np.sqrt(6.0 / (self.layer_sizes[i] + self.layer_sizes[i + 1]))
            weight = np.random.uniform(
                -limit, limit, 
                (self.layer_sizes[i], self.layer_sizes[i + 1])
            )
            bias = np.zeros((1, self.layer_sizes[i + 1]))
            
            self.weights.append(weight)
            self.biases.append(bias)
    
    def _get_activation(self, name: str) -> Callable:
        """Get activation function by name."""
        if name == 'relu':
            return lambda x: np.maximum(0, x)
        elif name == 'sigmoid':
            return lambda x: 1 / (1 + np.exp(-np.clip(x, -500, 500)))
        elif name == 'tanh':
            return np.tanh
        else:
            raise ValueError(f"Unknown activation function: {name}")
    
    def _get_activation_derivative(self, name: str) -> Callable:
        """Get derivative of activation function."""
        if name == 'relu':
            return lambda x: (x > 0).astype(float)
        elif name == 'sigmoid':
            return lambda x: x * (1 - x)
        elif name == 'tanh':
            return lambda x: 1 - x ** 2
        else:
            raise ValueError(f"Unknown activation function: {name}")
    
    def _softmax(self, x: np.ndarray) -> np.ndarray:
        """Compute softmax activation for output layer."""
        exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
        return exp_x / np.sum(exp_x, axis=1, keepdims=True)
    
    def forward(self, X: np.ndarray) -> Tuple[np.ndarray, dict]:
        """
        Forward pass through the network.
        
        Args:
            X: Input data of shape (batch_size, input_features)
        
        Returns:
            output: Network output of shape (batch_size, output_size)
            cache: Dictionary containing activations and Z values for backprop
        """
        cache = {'A0': X}
        A = X
        
        # Forward pass through hidden layers
        for i in range(self.num_layers - 2):
            Z = np.dot(A, self.weights[i]) + self.biases[i]
            A = self.activation_func(Z)
            cache[f'Z{i+1}'] = Z
            cache[f'A{i+1}'] = A
        
        # Output layer with softmax
        Z_out = np.dot(A, self.weights[-1]) + self.biases[-1]
        output = self._softmax(Z_out)
        cache[f'Z{self.num_layers-1}'] = Z_out
        cache[f'A{self.num_layers-1}'] = output
        
        return output, cache
    
    def backward(
        self,
        X: np.ndarray,
        y: np.ndarray,
        output: np.ndarray,
        cache: dict
    ) -> None:
        """
        Backward pass and update weights.
        
        Args:
            X: Input data
            y: True labels (one-hot encoded)
            output: Network output
            cache: Cache from forward pass
        """
        batch_size = X.shape[0]
        
        # Output layer error
        dZ = output - y
        
        # Backprop through layers
        for i in range(self.num_layers - 2, -1, -1):
            A_prev = cache[f'A{i}']
            
            # Gradient of weights and biases
            dW = np.dot(A_prev.T, dZ) / batch_size
            db = np.sum(dZ, axis=0, keepdims=True) / batch_size
            
            # Update weights and biases
            self.weights[i] -= self.learning_rate * dW
            self.biases[i] -= self.learning_rate * db
            
            # Propagate error to previous layer
            if i > 0:
                dA_prev = np.dot(dZ, self.weights[i].T)
                Z_prev = cache[f'Z{i}']
                dZ = dA_prev * self.activation_derivative(
                    self.activation_func(Z_prev)
                )
    
    def train(
        self,
        X: np.ndarray,
        y: np.ndarray,
        epochs: int = 100,
        batch_size: int = 32,
        validation_data: Tuple[np.ndarray, np.ndarray] = None,
        verbose: bool = True
    ) -> dict:
        """
        Train the neural network.
        
        Args:
            X: Training input data
            y: Training labels (one-hot encoded)
            epochs: Number of epochs
            batch_size: Batch size for training
            validation_data: Tuple of (X_val, y_val) for validation
            verbose: Whether to print training progress
        
        Returns:
            history: Dictionary containing training history
        """
        history = {'loss': [], 'accuracy': [], 'val_loss': [], 'val_accuracy': []}
        
        for epoch in range(epochs):
            # Shuffle data
            indices = np.random.permutation(X.shape[0])
            X_shuffled = X[indices]
            y_shuffled = y[indices]
            
            epoch_loss = 0
            num_batches = (X.shape[0] + batch_size - 1) // batch_size
            
            # Mini-batch training
            for batch in range(num_batches):
                start_idx = batch * batch_size
                end_idx = min((batch + 1) * batch_size, X.shape[0])
                
                X_batch = X_shuffled[start_idx:end_idx]
                y_batch = y_shuffled[start_idx:end_idx]
                
                # Forward and backward pass
                output, cache = self.forward(X_batch)
                self.backward(X_batch, y_batch, output, cache)
                
                # Calculate loss
                batch_loss = -np.mean(y_batch * np.log(output + 1e-8))
                epoch_loss += batch_loss
            
            # Calculate metrics
            train_output, _ = self.forward(X)
            train_loss = -np.mean(y * np.log(train_output + 1e-8))
            train_accuracy = self._calculate_accuracy(y, train_output)
            
            history['loss'].append(train_loss)
            history['accuracy'].append(train_accuracy)
            
            # Validation
            if validation_data is not None:
                X_val, y_val = validation_data
                val_output, _ = self.forward(X_val)
                val_loss = -np.mean(y_val * np.log(val_output + 1e-8))
                val_accuracy = self._calculate_accuracy(y_val, val_output)
                
                history['val_loss'].append(val_loss)
                history['val_accuracy'].append(val_accuracy)
                
                if verbose:
                    print(
                        f"Epoch {epoch+1}/{epochs} - "
                        f"Loss: {train_loss:.4f}, Accuracy: {train_accuracy:.4f}, "
                        f"Val Loss: {val_loss:.4f}, Val Accuracy: {val_accuracy:.4f}"
                    )
            else:
                if verbose:
                    print(
                        f"Epoch {epoch+1}/{epochs} - "
                        f"Loss: {train_loss:.4f}, Accuracy: {train_accuracy:.4f}"
                    )
        return history
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Make predictions.
        
        Args:
            X: Input data
        
        Returns:
            predictions: Class predictions
        """
        output, _ = self.forward(X)
        return np.argmax(output, axis=1)
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Get prediction probabilities.
        
        Args:
            X: Input data
        
        Returns:
            probabilities: Output probabilities
        """
        output, _ = self.forward(X)
        return output
    
    def _calculate_accuracy(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Calculate accuracy from one-hot encoded labels."""
        predictions = np.argmax(y_pred, axis=1)
        true_labels = np.argmax(y_true, axis=1)
        return np.mean(predictions == true_labels)
    
    def save(self, filepath: str) -> None:
        """Save the model to a file."""
        model_data = {
            'weights': self.weights,
            'biases': self.biases,
            'layer_sizes': self.layer_sizes,
            'learning_rate': self.learning_rate,
            'activation': self.activation_name
        }
        np.save(filepath, model_data, allow_pickle=True)
    
    def load(self, filepath: str) -> None:
        """Load the model from a file."""
        model_data = np.load(filepath, allow_pickle=True).item()
        self.weights = model_data['weights']
        self.biases = model_data['biases']
        self.layer_sizes = model_data['layer_sizes']
        self.learning_rate = model_data['learning_rate']
        self.activation_name = model_data.get('activation', self.activation_name)
        self.num_layers = len(self.layer_sizes)
        self.activation_func = self._get_activation(self.activation_name)
        self.activation_derivative = self._get_activation_derivative(self.activation_name)
