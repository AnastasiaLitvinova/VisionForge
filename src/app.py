import streamlit as st
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
import torchvision.models as models
import os

model = models.resnet34(pretrained=False)
model.fc = nn.Linear(model.fc.in_features, 10)
model.load_state_dict(torch.load(os.path.join(os.getcwd(), 'saved_models', 'model.pth'), map_location=torch.device('cpu')))

model.eval()

transform = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
])

# Определить классы CIFAR-10
classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

# Streamlit приложение
st.title('Классификация изображений CIFAR-10')

uploaded_file = st.file_uploader("Загрузите изображение", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Открыть изображение с помощью PIL
    image = Image.open(uploaded_file)
    st.image(image, caption='Загруженное изображение', use_container_width=True)

    # Преобразовать изображение
    input_tensor = transform(image)
    input_batch = input_tensor.unsqueeze(0) # batch с одним изображением

    # Получить предсказание
    with torch.no_grad():
        output = model(input_batch)

    # Получить вероятности и предсказанный класс
    probabilities = torch.softmax(output[0], dim=0)
    predicted_class_index = torch.argmax(probabilities).item()
    predicted_class = classes[predicted_class_index]
    confidence = probabilities[predicted_class_index].item()

    # Вывод результата
    st.write(f'Предсказанный класс: {predicted_class} (уверенность: {confidence:.4f})')

    # Вывод вероятности для каждого класса (опционально)
    st.write('Вероятности классов:')
    for i, class_name in enumerate(classes):
        st.write(f'{class_name}: {probabilities[i].item():.4f}')
