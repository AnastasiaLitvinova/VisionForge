import torch


def evaluate_model(model, testloader, device="cpu"):
    """Оценивает модель на тестовом наборе и возвращает точность."""
    model.eval()
    test_correct_predictions = 0
    test_total_samples = 0
    with torch.no_grad():
        for data in testloader:
            inputs, labels = data[0].to(device), data[1].to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            correct = (predicted == labels).sum().item()
            test_correct_predictions += correct
            test_total_samples += len(labels)

    test_accuracy = test_correct_predictions / test_total_samples
    print(f"Test Accuracy: {test_accuracy:.4f}")
    return test_accuracy
