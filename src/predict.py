import streamlit as st
import torch
import torchvision.transforms as transforms
import os
import random

from models.resnet18se import ResNet18SE

# Constants
RANDOM_SEED = 42
BATCH_SIZE = 64
NUM_WORKERS = os.cpu_count()  # Number of worker threads for data loading
mean = (0.4914, 0.4822, 0.4465)
std = (0.1953, 0.1925, 0.1942)
CLASSES = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# Load trained model
@st.cache_resource # Cache the model loading
def load_model(model_path):
    # Use the ResNet18SE model
    model = ResNet18SE(num_classes=len(CLASSES))

    try:
      state_dict = torch.load(model_path, map_location=DEVICE, weights_only=True)
    except Exception as e:
      print(f"Error loading model: {e}")
      return None

    # Check if the model state_dict and the loaded state_dict match
    model_state_dict = model.state_dict()
    pretrained_dict = {
        k: v for k, v in state_dict.items()
        if k in model_state_dict and model_state_dict[k].size() == v.size()
    }

    model_state_dict.update(pretrained_dict)
    model.load_state_dict(model_state_dict)
    model.to(DEVICE)
    model.eval()

    return model


def transform_image(image):
    """Transforms an input image for prediction."""
    transform = transforms.Compose([
        transforms.Resize((224, 224)),  # Resize to CIFAR-10 size
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
