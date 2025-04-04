import torch
import torch.nn as nn
import torchvision.models as models
from torchvision.models.resnet import ResNet18_Weights


class SEBlock(nn.Module):
    def __init__(self, channel: int, reduction: int = 16) -> None:
        super(SEBlock, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)


class ResNet18SE(nn.Module):
    def __init__(
        self,
        device: torch.device = torch.device("cpu"),
        pretrained: bool = True,
        num_classes: int = 10
    ) -> None:
        """Initialize a ResNet18 model with attention."""
        super(ResNet18SE, self).__init__()
        weights = ResNet18_Weights.DEFAULT if pretrained else None
        self.resnet = models.resnet18(weights=weights)

        # Freeze weights
        for param in self.resnet.parameters():
            param.requires_grad = False

        # Add SE blocks after each convolutional block
        self._add_se_blocks()

        # Replace the final fully connected layer
        num_ftrs = self.resnet.fc.in_features
        self.resnet.fc = nn.Linear(num_ftrs, num_classes)

        # Move the model to the specified device
        self.resnet.to(device)
        if torch.__version__ >= "2.0":  # Use torch.compile() for optimization
            self.resnet = torch.compile(self.resnet)

    def _add_se_blocks(self) -> None:
        """Adds SE blocks after each convolutional block."""
        channels = [64, 128, 256, 512]
        for i, (layer, channel) in enumerate(zip(
            (self.resnet.layer1, self.resnet.layer2, self.resnet.layer3, self.resnet.layer4), channels)):
            layer.add_module(f"{i}_se", SEBlock(channel))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.resnet(x)

    def unfreeze_layers(self, num_layers: int) -> None:
        """Unfreezes the last num_layers layers for fine-tuning."""
        layers = [self.resnet.layer1, self.resnet.layer2,
                  self.resnet.layer3, self.resnet.layer4]
        for layer in layers[-num_layers:]:
            for param in layer.parameters():
                param.requires_grad = True
