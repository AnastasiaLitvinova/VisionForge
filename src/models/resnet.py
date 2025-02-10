import torch
import torch.nn as nn
import torchvision.models as models
from torchvision.models.resnet import ResNet34_Weights


def create_resnet34(pretrained=True, device="cpu", num_classes=10):
    """Создает и переносит ResNet34 модель на указанное устройство."""
    if pretrained:
        weights = ResNet34_Weights.DEFAULT # Или другой вариант, например, IMAGENET1K_V1
    else:
        weights = None

    model = models.resnet34(weights=weights)

    # Заменяем последний слой (fc) для классификации на нужное количество классов
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, num_classes)

    model.to(device)  # Переносим модель на устройство
    if torch.__version__ >= "2.0":  # используем torch.compile()
        model = torch.compile(model)
    return model
