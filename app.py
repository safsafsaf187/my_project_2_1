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
