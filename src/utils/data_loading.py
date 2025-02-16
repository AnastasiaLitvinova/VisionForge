import torch
from tqdm import tqdm
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, random_split

def calculate_mean_std(dataset):
    """Calculates the mean and std of each channel in the dataset."""
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1024, shuffle=False, num_workers=2, pin_memory=True)
    mean = torch.zeros(3)
    std = torch.zeros(3)
    total_images = 0

    for images, _ in tqdm(dataloader, desc='Calculating mean and std'):
        images = images.view(images.size(0), images.size(1), -1)
        mean += images.mean(2).sum(0)
        std += images.std(2).sum(0)
        total_images += images.size(0)

    mean /= total_images
    std /= total_images
    return mean, std


def load_and_preprocess_data(batch_size, random_seed=42):
    """Загружает, разделяет и преобразует данные CIFAR-10."""
    torch.manual_seed(random_seed)
    mean, std = calculate_mean_std(full_trainset)

    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.TrivialAugmentWide(num_magnitude_bins=31),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])

    full_trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
    testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)

    train_size = int(0.8 * len(full_trainset))
    val_size = len(full_trainset) - train_size
    trainset, valset = random_split(full_trainset, [train_size, val_size], generator=torch.Generator().manual_seed(random_seed))

    trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True)
    valloader = DataLoader(valset, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)
    testloader = DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)

    return trainloader, valloader, testloader
