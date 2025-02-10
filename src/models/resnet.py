import torch
import torch.nn as nn
import torchvision.models as models
from torchvision.models.resnet import ResNet34_Weights


def create_resnet34(pretrained=True, device="cpu", num_classes=10):
    """Создает и переносит ResNet34 модель на указанное устройство."""
    weights = ResNet34_Weights.DEFAULT if pretrained else None  # Correct usage
    model = models.resnet34(pretrained=pretrained, weights=weights)

    # Заменяем последний слой (fc) для классификации на нужное количество классов
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, num_classes)

    model.to(device)  # Переносим модель на устройство
    if torch.__version__ >= "2.0":  # используем torch.compile()
        model = torch.compile(model)
    return model
