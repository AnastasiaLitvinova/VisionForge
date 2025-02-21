from typing import List
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

from src.utils import settings 


def plot_confusion_matrix(
    true_labels: List[int],
    predicted_labels: List[int],
    class_names: List[str],
    title: str = 'Confusion Matrix'
) -> None:
    """
    Plots a confusion matrix for a given set of true labels and predicted labels.
    """
    confusion_matrix_ = confusion_matrix(true_labels, predicted_labels)
    _, axes = plt.subplots(figsize=(8, 6))
    sns.heatmap(confusion_matrix_, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names)
    axes.set_title(title)
    axes.set_xlabel('Predicted Label')
    axes.set_ylabel('True Label')
    plt.show()


def plot_validation_accuracy_and_loss(
    validation_accuracies: List[float],
    training_losses: List[float],
    epochs: List[int] = list(range(1, settings.num_epochs + 1)),
) -> None:
    """Plot validation accuracy and training loss for models."""
    _, ax = plt.subplots(figsize=(10, 5))

    # Plot accuracy for model
    ax.plot(epochs, validation_accuracies, label='Model Validation Accuracy')
    ax.plot(epochs, training_losses, label='Model Training Loss')

    ax.set_xlabel('Epoch')
    ax.set_ylabel('Validation Accuracy (%) / Training Loss')
    ax.set_title('Model Validation Accuracy and Training Loss')
    ax.legend()
    ax.grid(True)
    plt.show()
