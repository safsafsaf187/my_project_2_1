import streamlit as st
from PIL import Image
import torch
from torchvision import models, transforms
import time
import requests
from io import BytesIO
import os



# --- Интерфейс ---
st.title("***🌦️ Приветствуем в малиннике, бомжи***")

st.title("*ОХОТА ОБЪЯВЛЯЕТСЯ ОТКРЫТОЙ*")

image_path1 = "pic1.jpg"
st.image(image_path1, width=1000)
image_path3 = "pic3.jpg"
st.image(image_path3, width=1000)

