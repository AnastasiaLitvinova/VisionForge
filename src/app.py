import streamlit as st
import torch
import torchvision
import torchvision.transforms as transforms
from PIL import Image
import io
import requests
import ssl
import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

# Import the functions from predict.py
from predict import load_model, transform_image, predict_image, get_random_cifar10_image

# Constants (Ensure consistency with predict.py)
CLASSES = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def main():
    st.title("CIFAR-10 Image Classifier")

    # Load the model
    model_path = "saved_models/model.pth"
    model = load_model(model_path)

    # Sidebar for image selection
    st.sidebar.header("Image Selection")
    image_source = st.sidebar.radio("Select Image Source:",
                                     ("Upload Image", "Image URL", "Random CIFAR-10 Image"))

    image = None
    # Image loading logic
    if image_source == "Upload Image":
        uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
        if uploaded_file is not None:
            try:
                image = Image.open(uploaded_file).convert('RGB') # Ensure RGB
            except Exception as e:
                st.error(f"Error opening image: {e}")
                image = None

    elif image_source == "Image URL":
        image_url = st.text_input("Enter Image URL:")
        if image_url:
            try:
                response = requests.get(image_url, stream=True)
                response.raise_for_status()  # Raise HTTPError for bad responses (4xx or 5xx)
                image = Image.open(io.BytesIO(response.content)).convert('RGB')
            except requests.exceptions.RequestException as e:
                st.error(f"Error fetching image from URL: {e}")
                image = None
            except Exception as e:
                st.error(f"Error opening image from URL: {e}")
                image = None

    elif image_source == "Random CIFAR-10 Image":
        try:
            _create_unverified_https_context = ssl._create_unverified_context  # For older Python versions
        except AttributeError:
            pass
        else:
            ssl._create_default_https_context = _create_unverified_https_context
        # Load CIFAR-10 dataset for fetching random images
        # Using ToTensor here for simplicity of getting the image
        cifar10_dataset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                                        download=True, transform=transforms.ToTensor())

        if st.sidebar.button("Get Random Image"):  # Button press is necessary to trigger dataset loading
            image, _ = get_random_cifar10_image(cifar10_dataset)
            image = transforms.ToPILImage()(image)  # Convert back to PIL for display


    # Display image and prediction
    if image:
        st.image(image, caption="Selected Image", use_container_width=True)
        processed_image = transform_image(image)
        if processed_image is not None:  # Check if transformation was successful
            prediction = predict_image(model, processed_image)
            st.write(f"**Prediction:** {prediction}")
        else:
            st.warning("Image transformation failed.")
    else:
        st.info("Please select an image.")


if __name__ == "__main__":
    import requests  # Import here, only used in main block and URL loading
    main()
