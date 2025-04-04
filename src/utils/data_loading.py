import torch
from typing import Tuple
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, random_split
from torchvision.datasets import CIFAR10
import numpy as np
import random


class CIFAR10DataLoader:
    def __init__(self, batch_size: int, num_workers: int = 2, random_seed: int = 42) -> None:
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.random_seed = random_seed
        # ResNet18 uses this size as its input size.
        self.image_size = (224, 224)
        # For normalization.
        self.mean = (0.4914, 0.4822, 0.4465)
        self.std = (0.1953, 0.1925, 0.1942)

        # For reproducibility
        torch.manual_seed(self.random_seed)
        self.generator = torch.Generator().manual_seed(self.random_seed)

    def _get_transforms(self) -> Tuple[transforms.Compose, transforms.Compose]:
        """Creates transformations for training and test datasets."""
        transform_train = transforms.Compose([
            transforms.Resize(self.image_size),
            transforms.RandomCrop(self.image_size[0], padding=4),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(15),
            transforms.ToTensor(),
            transforms.Normalize(self.mean, self.std),
        ])

        transform_test = transforms.Compose([
            transforms.Resize(self.image_size),
            transforms.ToTensor(),
            transforms.Normalize(self.mean, self.std),
        ])

        return transform_train, transform_test

    def load_data(self) -> Tuple[DataLoader, DataLoader, DataLoader]:
        """Loads, splits, and preprocesses the CIFAR-10 dataset."""
        transform_train, transform_test = self._get_transforms()

        # Load full training set
        full_trainset = CIFAR10(root='./data', train=True,
                                download=True, transform=transform_train)
        testset = CIFAR10(root='./data', train=False,
                          download=True, transform=transform_test)

        # Define dataset sizes
        train_size = int(0.8 * len(full_trainset))
        val_size = len(full_trainset) - train_size

        # Randomly split into training and validation sets
        trainset, valset = random_split(
            full_trainset, [train_size, val_size], generator=self.generator)

        def seed_worker(worker_id: int) -> None:
            """Initializes seed for each worker in DataLoader."""
            worker_seed = torch.initial_seed() % 2**32
            np.random.seed(worker_seed)
            random.seed(worker_seed)

        g = torch.Generator()
        g.manual_seed(self.random_seed)

        # Create data loaders
        trainloader = DataLoader(trainset, batch_size=self.batch_size,
                                 shuffle=True, num_workers=self.num_workers, pin_memory=True,
                                 worker_init_fn=seed_worker, generator=g)
        valloader = DataLoader(valset, batch_size=self.batch_size,
                               shuffle=False, num_workers=self.num_workers, pin_memory=True,
                               worker_init_fn=seed_worker, generator=g)
        testloader = DataLoader(testset, batch_size=self.batch_size,
                                shuffle=False, num_workers=self.num_workers, pin_memory=True,
                                worker_init_fn=seed_worker, generator=g)

        return trainloader, valloader, testloader
