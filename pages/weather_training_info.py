import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import pandas as pd
import numpy as np
import json
import os
from sklearn.metrics import confusion_matrix, f1_score, ConfusionMatrixDisplay

st.title("📊 Информация о тренировке модели")

# --- Пути к файлам с логами (можно сохранять в train_weather.py) ---
log_path = "models/train_log.json"
cm_path = "models/confusion.json"

# --- Кривая обучения ---
if os.path.exists(log_path):
    with open(log_path, "r") as f:
        log = json.load(f)

    losses = log.get("valid_losses", [])
    timesec = log.get("training_time", None)

    st.subheader("📉 Кривая обучения (Loss по эпохам)")
    fig, ax = plt.subplots()
    ax.plot(range(1, len(losses)+1), losses, marker="o")
    ax.set_xlabel("Эпоха")
    ax.set_ylabel("Loss")
    st.pyplot(fig)

    if timesec:
        minutes = round(timesec / 60, 2)
        st.info(f"⏱️ Общее время обучения: {minutes} минут")
else:
    st.warning("Лог обучения не найден. Обновите train_weather.py для записи логов.")

# --- Распределение классов ---
st.subheader("📊 Распределение классов в датасете")
class_counts = {
    'dew': 1051, 'fogsmog': 934, 'frost': 1067, 'glaze': 1075, 'hail': 983, 'lightning': 1074,
    'rain': 1000, 'rainbow': 980, 'rime': 1057, 'sandstorm': 1026, 'snow': 982
}
df_counts = pd.DataFrame.from_dict(class_counts, orient='index', columns=["Количество"])
st.bar_chart(df_counts)

# --- Confusion matrix ---
if os.path.exists(cm_path):
    with open(cm_path, "r") as f:
        cm_data = json.load(f)
        y_true = cm_data["true"]
        y_pred = cm_data["pred"]

    f1 = f1_score(y_true, y_pred, average='weighted')
    st.subheader("🎯 Confusion Matrix и F1-score")
    st.write(f"**F1-score**: {f1:.3f}")

    cm = confusion_matrix(y_true, y_pred)
    fig2, ax2 = plt.subplots(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=class_counts.keys(), yticklabels=class_counts.keys())
    st.pyplot(fig2)
else:
    st.warning("Файл с предсказаниями не найден. Добавьте сохранение в train_weather.py.")
