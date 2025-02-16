import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import pandas as pd
from optuna.exceptions import TrialPruned


def calculate_accuracy(outputs, labels):
    """Вычисляет точность."""
    _, predicted = torch.max(outputs.data, 1)
    correct = (predicted == labels).sum().item()
    total = labels.size(0)
    return correct / total


def train_model(
        model,
        trainloader,
        valloader,
        learning_rate,
        weight_decay,
        step_size,
        gamma,
        num_epochs,
        device="cpu",
        random_seed=42,
        trial=None
):
    """Обучает модель и возвращает историю обучения."""
    torch.manual_seed(random_seed)

    criterion = nn.CrossEntropyLoss().to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)

    history = {"loss": [], "accuracy": [], "val_loss": [], "val_accuracy": []}
    best_val_accuracy = 0.0
    best_model_state = None  # Сохранить состояние лучшей модели
    patience = 3  # Количество эпох без улучшений для early stopping
    trigger = False
    last_loss = float('inf') # последний loss

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        correct_predictions = 0
        total_samples = 0
        for i, data in enumerate(trainloader, 0):
            inputs, labels = data[0].to(device), data[1].to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            accuracy = calculate_accuracy(outputs, labels)
            correct_predictions += accuracy * labels.size(0)
            total_samples += labels.size(0)

        scheduler.step()
        epoch_loss = running_loss / len(trainloader)
        epoch_accuracy = correct_predictions / total_samples
        history["loss"].append(epoch_loss)
        history["accuracy"].append(epoch_accuracy)

        # Оценка на валидационном наборе
        model.eval()
        val_loss = 0.0
        val_correct_predictions = 0
        val_total_samples = 0
        with torch.no_grad():
            for data in valloader:
                inputs, labels = data[0].to(device), data[1].to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                val_accuracy = calculate_accuracy(outputs, labels)
                val_correct_predictions += val_accuracy * labels.size(0)
                val_total_samples += labels.size(0)

        val_epoch_loss = val_loss / len(valloader)
        val_epoch_accuracy = val_correct_predictions / val_total_samples
        history["val_loss"].append(val_epoch_loss)
        history["val_accuracy"].append(val_epoch_accuracy)
        print(f"Epoch {epoch+1} - Loss: {epoch_loss:.4f}, Accuracy: {epoch_accuracy:.4f}, Val Loss: {val_epoch_loss:.4f}, Val Accuracy: {val_epoch_accuracy:.4f}, LR: {scheduler.get_last_lr()}")

        # Сохранение лучшей модели
        if val_epoch_accuracy > best_val_accuracy:
            best_val_accuracy = val_epoch_accuracy
            best_model_state = model.state_dict()
        # Early stopping
        if val_epoch_loss > last_loss:
            trigger = True
        if trigger:
            patience -= 1
            if patience <= 0:
                print("Early stopping triggered")
                break
        last_loss = val_epoch_loss
        if trial:
            trial.report(val_epoch_accuracy, epoch)
            # Handle pruning based on the intermediate value.
            if trial.should_prune():
                raise TrialPruned()

    print('Finished Training')
    # Загружаем лучшее состояние модели
    if best_model_state is not None:
        model.load_state_dict(best_model_state)

    return model, pd.DataFrame(history)
