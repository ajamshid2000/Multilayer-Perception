"""
Data loading and preprocessing utilities for Multilayer Perceptron.
"""

import csv
import numpy as np
from typing import Generator, Tuple, Optional
import os


class DataLoader:
    """Utilities for loading and preprocessing datasets."""
    
    @staticmethod
    def standardize(X: np.ndarray) -> np.ndarray:
        """
        Standardize data (zero mean, unit variance).
        
        Args:
            X: Input data
        
        Returns:
            Standardized data
        """
        mean = np.mean(X, axis=0)
        std = np.std(X, axis=0)
        return (X - mean) / (std + 1e-8)
    
    @staticmethod
    def one_hot_encode(y: np.ndarray, num_classes: Optional[int] = None) -> np.ndarray:
        """
        One-hot encode labels.
        
        Args:
            y: Array of class labels
            num_classes: Number of classes (inferred if None)
        
        Returns:
            One-hot encoded labels
        """
        if num_classes is None:
            num_classes = int(np.max(y)) + 1
        
        one_hot = np.zeros((y.shape[0], num_classes))
        one_hot[np.arange(y.shape[0]), y.astype(int)] = 1
        return one_hot
    

    @staticmethod
    def load_breast_cancer_csv(
        data_dir: str = 'data',
        filename: str = 'data.csv'
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Load the Wisconsin breast cancer dataset from a CSV file.

        Args:
            data_dir: Directory containing the CSV file
            filename: CSV filename

        Returns:
            X: Feature matrix
            y: Binary labels (0=benign, 1=malignant)
        """
        file_path = os.path.join(data_dir, filename)

        if os.path.exists(file_path):
            raw = np.genfromtxt(file_path, delimiter=',', dtype=str, skip_header=1)
            if raw.ndim == 1:
                raw = raw.reshape(1, -1)
            if raw.shape[1] < 3:
                raise ValueError('Expected breast cancer CSV with at least 3 columns.')

            diagnosis = raw[:, 1]
            y = np.where(diagnosis == 'M', 1, 0).astype(int)
            X = raw[:, 2:].astype(float)
            return X, y

        try:
            from sklearn.datasets import load_breast_cancer
        except ImportError as exc:
            raise FileNotFoundError(
                f"Breast cancer CSV not found at {file_path} and scikit-learn is unavailable."
            ) from exc

        print(f"Breast cancer CSV not found. Generating dataset from scikit-learn and writing {file_path}...")
        data = load_breast_cancer()
        X = data.data
        class_names = [name.lower() for name in data.target_names]
        malignant_index = class_names.index('malignant') if 'malignant' in class_names else 0
        y = (data.target == malignant_index).astype(int)

        os.makedirs(data_dir, exist_ok=True)
        with open(file_path, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            header = ['id', 'diagnosis'] + [str(name) for name in data.feature_names]
            writer.writerow(header)
            for i, (row, label) in enumerate(zip(X, y), start=1):
                writer.writerow([i, 'M' if label == 1 else 'B'] + row.tolist())

        return X, y

    @staticmethod
    def _load_breast_cancer_split_file(file_path: str) -> Tuple[np.ndarray, np.ndarray]:
        """
        Load a breast cancer CSV split file.

        Args:
            file_path: Path to the split CSV file

        Returns:
            X: Features
            y: Binary labels
        """
        raw = np.genfromtxt(file_path, delimiter=',', dtype=str, skip_header=1)
        if raw.ndim == 1:
            raw = raw.reshape(1, -1)
        diagnosis = raw[:, 1]
        y = np.where(diagnosis == 'M', 1, 0).astype(int)
        X = raw[:, 2:].astype(float)
        return X, y

    @staticmethod
    def save_breast_cancer_split(
        X: np.ndarray,
        y: np.ndarray,
        output_dir: str = 'data',
        train_filename: str = 'data_train.csv',
        valid_filename: str = 'data_valid.csv',
        random_state: Optional[int] = None,
        test_size: float = 0.2
    ) -> Tuple[str, str]:
        """
        Split breast cancer data and save train/validation CSV files.

        Args:
            X: Feature matrix
            y: Binary labels
            output_dir: Directory to save split CSVs
            train_filename: Training CSV filename
            valid_filename: Validation CSV filename
            random_state: Seed for split
            test_size: Validation split size

        Returns:
            Paths to saved training and validation CSV files
        """
        X_train, X_valid, y_train, y_valid = DataLoader.train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )

        os.makedirs(output_dir, exist_ok=True)
        train_path = os.path.join(output_dir, train_filename)
        valid_path = os.path.join(output_dir, valid_filename)

        header = ['id', 'diagnosis'] + [f'feature_{i}' for i in range(1, X.shape[1] + 1)]

        for path, data_X, data_y in [
            (train_path, X_train, y_train),
            (valid_path, X_valid, y_valid)
        ]:
            with open(path, 'w', newline='') as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow(header)
                for i, (row, label) in enumerate(zip(data_X, data_y), start=1):
                    writer.writerow([i, 'M' if label == 1 else 'B'] + row.tolist())

        return train_path, valid_path


    @staticmethod
    def train_test_split(
        X: np.ndarray,
        y: np.ndarray,
        test_size: float = 0.2,
        random_state: Optional[int] = None
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Split dataset into training and testing sets.
        
        Args:
            X: Input features
            y: Labels
            test_size: Proportion of test set
            random_state: Random seed
        
        Returns:
            (X_train, X_test, y_train, y_test)
        """
        if random_state is not None:
            np.random.seed(random_state)
        
        indices = np.random.permutation(X.shape[0])
        split_index = int(X.shape[0] * (1 - test_size))
        
        train_indices = indices[:split_index]
        test_indices = indices[split_index:]
        
        return X[train_indices], X[test_indices], y[train_indices], y[test_indices]
    
    @staticmethod
    def batch_iterator(
        X: np.ndarray,
        y: np.ndarray,
        batch_size: int = 32,
        shuffle: bool = True
    ) -> Generator[tuple, None, None]:
        """
        Iterate over batches of data.
        
        Args:
            X: Input features
            y: Labels
            batch_size: Size of each batch
            shuffle: Whether to shuffle data
        
        Yields:
            (X_batch, y_batch)
        """
        num_samples = X.shape[0]
        indices = np.arange(num_samples)
        
        if shuffle:
            np.random.shuffle(indices)
        
        for start_idx in range(0, num_samples, batch_size):
            end_idx = min(start_idx + batch_size, num_samples)
            batch_indices = indices[start_idx:end_idx]
            
            yield X[batch_indices], y[batch_indices]


def load_data_for_training(
    data_dir: str = 'data',
    validation_split: float = 0.2,
    random_state: Optional[int] = None
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Load and preprocess Wisconsin breast cancer dataset for training.
    
    Args:
        data_dir: Directory containing data
        validation_split: Validation split ratio
        random_state: Seed used for reproducibility when splitting
    
    Returns:
        (X_train, y_train, X_valid, y_valid) - Training and validation data
    """
    train_path = os.path.join(data_dir, 'data_train.csv')
    valid_path = os.path.join(data_dir, 'data_valid.csv')
    if os.path.exists(train_path) and os.path.exists(valid_path):
        X_train, y_train = DataLoader._load_breast_cancer_split_file(train_path)
        X_valid, y_valid = DataLoader._load_breast_cancer_split_file(valid_path)
    else:
        X, y = DataLoader.load_breast_cancer_csv(data_dir)
        X = DataLoader.standardize(X)
        X_train, X_valid, y_train, y_valid = DataLoader.train_test_split(
            X, y, test_size=validation_split, random_state=random_state
        )
        return (
            X_train,
            DataLoader.one_hot_encode(y_train, 2),
            X_valid,
            DataLoader.one_hot_encode(y_valid, 2)
        )

    X_train = DataLoader.standardize(X_train)
    X_valid = DataLoader.standardize(X_valid)
    y_train = DataLoader.one_hot_encode(y_train, 2)
    y_valid = DataLoader.one_hot_encode(y_valid, 2)
    return X_train, y_train, X_valid, y_valid
