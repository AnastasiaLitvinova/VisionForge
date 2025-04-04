import torch
from torch.utils.data import DataLoader
from typing import Tuple, List


def evaluate_test(
    model: torch.nn.Module,
    test_loader: DataLoader,
    device: torch.device,
) -> Tuple[List[int], List[int]]:
    """Evaluate a model on a given test dataset."""
    model.eval()
    true_labels = []
    predicted_labels = []
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            true_labels.extend(labels.cpu().numpy())
            predicted_labels.extend(predicted.cpu().numpy())

    return true_labels, predicted_labels
