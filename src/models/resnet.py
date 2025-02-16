import torch
import torch.nn as nn
import torchvision.models as models
from torchvision.models.resnet import ResNet18_Weights


def create_resnet18(pretrained=True, device="cpu", num_classes=10):
    """Создает и переносит ResNet18 модель на указанное устройство."""
    if pretrained:
        weights = ResNet18_Weights.DEFAULT
    else:
        weights = None

    model = models.resnet18(weights=weights)

    # Замена последнего слоя (fc) для классификации на нужное количество классов
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, num_classes)

    model.to(device)  # Перенос модели на устройство
    if torch.__version__ >= "2.0":  # используя torch.compile()
        model = torch.compile(model)
    return model
