import os
import time
import json
import torch
from torch import nn
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
from sklearn.metrics import confusion_matrix
from tqdm import tqdm

# Пути
DATA_PATH = r"C:\Users\Galin22\.cache\kagglehub\datasets\jehanbhathena\weather-dataset\versions\3\dataset"
SAVE_PATH = "models/resnet_weather.pt"
LOG_PATH = "models/train_log.json"
CONFUSION_PATH = "models/confusion.json"
EPOCHS = 3
BATCH_SIZE = 16

# Трансформации
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

# Датасет и DataLoader
dataset = datasets.ImageFolder(DATA_PATH, transform=transform)
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
classes = dataset.classes
print("Классы:", classes)

# Модель
model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
for param in model.parameters():
    param.requires_grad = False
model.fc = nn.Linear(model.fc.in_features, len(classes))

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.fc.parameters(), lr=0.001)

# Логирование
losses = []
y_true_all = []
y_pred_all = []

start_time = time.time()

# Обучение
for epoch in range(EPOCHS):
    model.train()
    running_loss = 0

    for images, labels in tqdm(dataloader, desc=f"Эпоха {epoch+1}/{EPOCHS}"):
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

        # Для метрик
        preds = torch.argmax(outputs, dim=1)
        y_true_all.extend(labels.cpu().tolist())
        y_pred_all.extend(preds.cpu().tolist())

    epoch_loss = running_loss
    losses.append(epoch_loss)
    print(f"[{epoch+1}] Loss: {epoch_loss:.4f}")

# Время
total_time = time.time() - start_time
print("✅ Обучение завершено.")
print(f"⏱️ Время обучения: {round(total_time / 60, 2)} минут")

# Сохраняем модель
os.makedirs("models", exist_ok=True)
torch.save(model.state_dict(), SAVE_PATH)
print("💾 Модель сохранена в:", SAVE_PATH)

# Сохраняем лог
train_log = {
    "losses": losses,
    "training_time": total_time
}
with open(LOG_PATH, "w") as f:
    json.dump(train_log, f)
print("📝 Лог обучения сохранён.")

# Сохраняем confusion данные
conf_data = {
    "true": y_true_all,
    "pred": y_pred_all
}
with open(CONFUSION_PATH, "w") as f:
    json.dump(conf_data, f)
print("🧩 Данные для confusion matrix сохранены.")
