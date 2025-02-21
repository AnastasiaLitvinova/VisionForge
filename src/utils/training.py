import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau

from utils.experiment_config import settings


def train_model(
    model: nn.Module,
    trainloader: torch.utils.data.DataLoader,
    valloader: torch.utils.data.DataLoader,
    learning_rate: float = settings.learning_rate,
    num_epochs: int = settings.num_epochs,
    weight_decay: float = settings.weight_decay,
    patience: int = settings.patience,
    device: torch.device = settings.device,
    random_seed: int = 42
) -> tuple[list[float], list[float]]:
    """Trains a model and returns its training history."""
    torch.manual_seed(random_seed)
    model.to(device)
    criterion = nn.CrossEntropyLoss().to(device)
    optimizer = optim.Adam(
        model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = ReduceLROnPlateau(
        optimizer, mode='max', factor=0.1, patience=patience // 2)

    train_losses = []
    val_accuracies = []
    best_val_accuracy = float('-inf')
    epochs_no_improve = 0

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for inputs, labels in trainloader:
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        train_loss = running_loss / len(trainloader)
        train_losses.append(train_loss)
        print(f"Epoch {epoch+1}/{num_epochs}, Training Loss: {train_loss:.4f}")

        # Evaluate on validation set
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for inputs, labels in valloader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        val_accuracy = 100 * correct / total
        val_accuracies.append(val_accuracy)
        print(
            f"Epoch {epoch+1}/{num_epochs}, Validation Accuracy: {val_accuracy:.2f}%")

        # Early stopping
        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            epochs_no_improve = 0
            best_model_state = model.state_dict()
        else:
            epochs_no_improve += 1
            if epochs_no_improve == patience:
                print(f"Early stopping! No improvement for {patience} epochs.")
                model.load_state_dict(best_model_state)
                break

        scheduler.step(val_accuracy)
        current_lr = scheduler.get_last_lr()[0]
        if current_lr < learning_rate:
            print(f"Learning rate reduced to: {current_lr}")

    return train_losses, val_accuracies
