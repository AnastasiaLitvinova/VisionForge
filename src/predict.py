import streamlit as st
import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import os
import random

# Constants
RANDOM_SEED = 42
BATCH_SIZE = 32
NUM_WORKERS = os.cpu_count()  # Number of worker threads for data loading
mean = (0.4914, 0.4822, 0.4465)
std = (0.2023, 0.1994, 0.2010)
CLASSES = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Transformations
transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean, std),
])

# Load trained model
@st.cache_resource # Cache the model loading
def load_model(model_path):
    """Loads the trained ResNet18 model."""
    model = torchvision.models.resnet18()
    model.fc = nn.Linear(model.fc.in_features, len(CLASSES))
    state_dict = torch.load(model_path, map_location=DEVICE)

    # Load only the keys that are present in both the model and the state_dict
    model_state_dict = model.state_dict()
    pretrained_dict = {
        k: v for k, v in state_dict.items()
        if k in model_state_dict and model_state_dict[k].size() == v.size()
    }

    # Update the model with the pre-trained weights
    model_state_dict.update(pretrained_dict)
    model.load_state_dict(model_state_dict)
    model.to(DEVICE)
    model.eval()  # Set to evaluation mode
    return model


def transform_image(image):
    """Transforms an input image for prediction."""
    transform = transforms.Compose([
        transforms.Resize((32, 32)),  # Resize to CIFAR-10 size
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])
    image = transform(image).unsqueeze(0)  # Add batch dimension
    return image.to(DEVICE)


def predict_image(model, image):
    """Predicts the class of the input image."""
    model.eval()
    with torch.no_grad():
        output = model(image)
        _, predicted = torch.max(output.data, 1)
    return CLASSES[predicted[0]]

def get_random_cifar10_image(dataset):
    """Fetches a random image and its label from CIFAR-10."""
    index = random.randint(0, len(dataset) - 1)
    image, label = dataset[index]
    return image, label
