"""
Utilities and visualization functions for Multilayer Perceptron. 
Author: École 42
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple
import os


class Visualization:
    """Visualization utilities for neural network training and results."""
    
    @staticmethod
    def plot_training_history(history: Dict[str, List[float]], save_path: str = None):
        """
        Plot training history.
        
        Args:
            history: Dictionary containing training history
            save_path: Optional path to save the figure
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        
        # Loss plot
        ax1.plot(history['loss'], label='Training Loss', linewidth=2)
        if 'val_loss' in history:
            ax1.plot(history['val_loss'], label='Validation Loss', linewidth=2)
        ax1.set_xlabel('Epoch', fontsize=12)
        ax1.set_ylabel('Loss', fontsize=12)
        ax1.set_title('Model Loss', fontsize=14, fontweight='bold')
        ax1.legend(fontsize=10)
        ax1.grid(True, alpha=0.3)
        
        # Accuracy plot
        ax2.plot(history['accuracy'], label='Training Accuracy', linewidth=2)
        if 'val_accuracy' in history:
            ax2.plot(history['val_accuracy'], label='Validation Accuracy', linewidth=2)
        ax2.set_xlabel('Epoch', fontsize=12)
        ax2.set_ylabel('Accuracy', fontsize=12)
        ax2.set_title('Model Accuracy', fontsize=14, fontweight='bold')
        ax2.legend(fontsize=10)
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            os.makedirs(os.path.dirname(save_path) or '.', exist_ok=True)
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Figure saved to: {save_path}")
        
        plt.show()
    
    @staticmethod
    def plot_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, save_path: str = None):
        """
        Plot confusion matrix.
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            save_path: Optional path to save the figure
        """
        from sklearn.metrics import confusion_matrix
        import seaborn as sns
        
        cm = confusion_matrix(y_true, y_pred)
        
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=True)
        plt.ylabel('True Label', fontsize=12)
        plt.xlabel('Predicted Label', fontsize=12)
        plt.title('Confusion Matrix', fontsize=14, fontweight='bold')
        
        if save_path:
            os.makedirs(os.path.dirname(save_path) or '.', exist_ok=True)
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Figure saved to: {save_path}")
        
        plt.show()
    

    @staticmethod
    def plot_distribution(
        data: np.ndarray,
        title: str = "Distribution",
        save_path: str = None
    ):
        """
        Plot distribution of data.
        
        Args:
            data: Data to plot
            title: Plot title
            save_path: Optional path to save the figure
        """
        plt.figure(figsize=(10, 6))
        plt.hist(data.flatten(), bins=50, edgecolor='black', alpha=0.7)
        plt.xlabel('Value', fontsize=12)
        plt.ylabel('Frequency', fontsize=12)
        plt.title(title, fontsize=14, fontweight='bold')
        plt.grid(True, alpha=0.3)
        
        if save_path:
            os.makedirs(os.path.dirname(save_path) or '.', exist_ok=True)
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Figure saved to: {save_path}")
        
        plt.show()


class Metrics:
    """Evaluation metrics."""
    
    @staticmethod
    def accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Calculate accuracy."""
        return np.mean(y_true == y_pred)
    
    @staticmethod
    def precision(y_true: np.ndarray, y_pred: np.ndarray, class_id: int = None) -> float:
        """Calculate precision."""
        if class_id is not None:
            tp = np.sum((y_pred == class_id) & (y_true == class_id))
            fp = np.sum((y_pred == class_id) & (y_true != class_id))
        else:
            tp = np.sum(y_pred == y_true)
            fp = np.sum(y_pred != y_true)
        
        if tp + fp == 0:
            return 0.0
        return tp / (tp + fp)
    
    @staticmethod
    def recall(y_true: np.ndarray, y_pred: np.ndarray, class_id: int = None) -> float:
        """Calculate recall."""
        if class_id is not None:
            tp = np.sum((y_pred == class_id) & (y_true == class_id))
            fn = np.sum((y_pred != class_id) & (y_true == class_id))
        else:
            tp = np.sum(y_pred == y_true)
            fn = np.sum(y_pred != y_true)
        
        if tp + fn == 0:
            return 0.0
        return tp / (tp + fn)
    
    @staticmethod
    def f1_score(y_true: np.ndarray, y_pred: np.ndarray, class_id: int = None) -> float:
        """Calculate F1 score."""
        precision = Metrics.precision(y_true, y_pred, class_id)
        recall = Metrics.recall(y_true, y_pred, class_id)
        
        if precision + recall == 0:
            return 0.0
        return 2 * (precision * recall) / (precision + recall)
    
    @staticmethod
    def classification_report(y_true: np.ndarray, y_pred: np.ndarray) -> str:
        """Generate a classification report."""
        num_classes = max(np.max(y_true), np.max(y_pred)) + 1
        
        report = "Classification Report\n"
        report += "=" * 60 + "\n"
        report += f"{'Class':<10} {'Precision':<15} {'Recall':<15} {'F1-Score':<15}\n"
        report += "-" * 60 + "\n"
        
        for class_id in range(num_classes):
            precision = Metrics.precision(y_true, y_pred, class_id)
            recall = Metrics.recall(y_true, y_pred, class_id)
            f1 = Metrics.f1_score(y_true, y_pred, class_id)
            
            report += f"{class_id:<10} {precision:<15.4f} {recall:<15.4f} {f1:<15.4f}\n"
        
        report += "-" * 60 + "\n"
        accuracy = Metrics.accuracy(y_true, y_pred)
        report += f"{'Accuracy':<40} {accuracy:<15.4f}\n"
        report += "=" * 60 + "\n"
        
        return report

