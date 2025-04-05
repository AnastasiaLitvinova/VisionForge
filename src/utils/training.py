import torch.nn as nn
import torch
from torch.utils.data import DataLoader
from typing import List, Tuple
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau


class ModelTrainer:
    def __init__(self, model: nn.Module, device: torch.device, random_seed: int = 42):
        self.model = model
        self.device = device
        self.random_seed = random_seed
        torch.manual_seed(self.random_seed)

    def train(self, trainloader: DataLoader, valloader: DataLoader,
              learning_rate: float, num_epochs: int,
              weight_decay: float, patience: int) -> Tuple[List[float], List[float]]:
        """Trains the model and returns its training history."""
        self.model.to(self.device)
        criterion = nn.CrossEntropyLoss().to(self.device)
        optimizer = optim.Adam(self.model.parameters(),
                               lr=learning_rate, weight_decay=weight_decay)
        scheduler = ReduceLROnPlateau(
            optimizer, mode='max', factor=0.1, patience=patience // 2, min_lr=0)

        train_losses = []
        val_accuracies = []
        best_val_accuracy = float('-inf')
        epochs_no_improve = 0

        for epoch in range(num_epochs):
            self.model.train()
            running_loss = 0.0
            for inputs, labels in trainloader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)

                optimizer.zero_grad()
                outputs = self.model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                running_loss += loss.item()

            train_loss = running_loss / len(trainloader)
            train_losses.append(train_loss)
            print(
                f"Epoch {epoch+1}/{num_epochs}, Training Loss: {train_loss:.4f}")

            # Evaluate on validation set
            val_accuracy = self.evaluate(valloader)
            val_accuracies.append(val_accuracy)

            # Early stopping
            if val_accuracy > best_val_accuracy:
                best_val_accuracy = val_accuracy
                epochs_no_improve = 0
                best_model_state = self.model.state_dict()
            else:
                epochs_no_improve += 1
                if epochs_no_improve == patience:
                    print(
                        f"Early stopping! No improvement for {patience} epochs.")
                    self.model.load_state_dict(best_model_state)
                    break

            scheduler.step(val_accuracy)
            current_lr = scheduler.get_last_lr()[0]
            if current_lr < learning_rate:
                print(f"Learning rate reduced to: {current_lr}")

        return train_losses, val_accuracies

    def evaluate(self, dataloader: DataLoader) -> float:
        """Evaluates the model on the given dataloader and returns the accuracy."""
        self.model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for inputs, labels in dataloader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                outputs = self.model(inputs)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        accuracy = 100 * correct / total
        print(f"Validation Accuracy: {accuracy:.2f}%")
        return accuracy
