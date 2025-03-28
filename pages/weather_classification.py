import streamlit as st
from PIL import Image
import torch
from torchvision import models, transforms
import time
import requests
from io import BytesIO
import os

# --- Константы ---
MODEL_PATH = "models/resnet_weather.pt"
CLASS_NAMES = ['dew', 'fogsmog', 'frost', 'glaze', 'hail', 'lightning',
               'rain', 'rainbow', 'rime', 'sandstorm', 'snow']

# --- Трансформации ---
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

# --- Загрузка модели ---
@st.cache_resource
def load_model():
    model = models.resnet18(weights=None)
    model.fc = torch.nn.Linear(model.fc.in_features, len(CLASS_NAMES))
    model.load_state_dict(torch.load(MODEL_PATH, map_location='cpu'))
    model.eval()
    return model

model = load_model()

# --- Интерфейс ---
st.title("🌦️ Классификация приколов природы")
st.write("Загрузите изображение или вставьте ссылку — модель определит тип погодного явления.")

uploaded_files = st.file_uploader("📁 Загрузка изображений", type=["jpg", "jpeg", "png"], accept_multiple_files=True)
image_url = st.text_input("🔗 Или вставьте URL изображения")

images_to_classify = []

# Обработка ссылки
if image_url:
    try:
        response = requests.get(image_url)
        image = Image.open(BytesIO(response.content)).convert("RGB")
        images_to_classify.append(("URL", image))
    except Exception as e:
        st.error(f"Ошибка при загрузке изображения по ссылке: {e}")

# Обработка загруженных файлов
if uploaded_files:
    for file in uploaded_files:
        image = Image.open(file).convert("RGB")
        images_to_classify.append((file.name, image))

# Классификация
if images_to_classify:
    for name, image in images_to_classify:
        st.subheader(f"🖼️ {name}")
        st.image(image, width=300)

        input_tensor = transform(image).unsqueeze(0)

        with torch.no_grad():
            start = time.time()
            output = model(input_tensor)
            end = time.time()

        pred_class = CLASS_NAMES[output.argmax().item()]
        elapsed_time = end - start

        st.success(f"🌈 Определено: **{pred_class}**")
        st.info(f"⏱️ Время ответа: {elapsed_time:.3f} сек")

else:
    st.info("Пожалуйста, загрузите изображение или вставьте ссылку.")
