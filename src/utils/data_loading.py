import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, random_split

from utils.experiment_config import settings


def load_and_preprocess_data(
    batch_size: int = settings.batch_size,
    num_workers: int = settings.num_workers,
    random_seed: int = 42,
    mean: tuple[float, float, float] = (0.4914, 0.4822, 0.4465),
    std: tuple[float, float, float] = (0.1953, 0.1925, 0.1942)
) -> tuple[DataLoader, DataLoader, DataLoader]:
    """Loads, splits, and preprocesses the CIFAR-10 dataset."""
    torch.manual_seed(random_seed)
    mean, std = mean, std

    transform_train = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomCrop(224, padding=4),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(15),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])

    transform_test = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])

    full_trainset = torchvision.datasets.CIFAR10(
        root='./data', train=True, download=True, transform=transform_train)
    testset = torchvision.datasets.CIFAR10(
        root='./data', train=False, download=True, transform=transform_test)

    train_size = int(0.8 * len(full_trainset))
    val_size = len(full_trainset) - train_size
    generator = torch.Generator().manual_seed(random_seed)
    trainset, valset = random_split(full_trainset, [
                                    train_size, val_size], generator=generator)

    # Create separate dataset object for validation set to apply the test transformation
    valset = torchvision.datasets.CIFAR10(root='./data', train=True, download=False, transform=transform_test)
    valset.data = full_trainset.data[trainset.indices[-val_size:]]
    valset.targets = [full_trainset.targets[i] for i in trainset.indices[-val_size:]]

    trainloader = DataLoader(trainset, batch_size=batch_size,
                             shuffle=True, num_workers=num_workers, pin_memory=True)
    valloader = DataLoader(valset, batch_size=batch_size,
                           shuffle=False, num_workers=num_workers, pin_memory=True)
    testloader = DataLoader(testset, batch_size=batch_size,
                            shuffle=False, num_workers=num_workers, pin_memory=True)

    return trainloader, valloader, testloader
