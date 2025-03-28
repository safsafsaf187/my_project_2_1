# import os
# import time
# import json
# import torch
# from torch import nn
# from torchvision import datasets, transforms, models
# from torch.utils.data import DataLoader
# from sklearn.metrics import confusion_matrix
# from tqdm import tqdm

# # Пути
# DATA_PATH = r"C:\Users\safar\OneDrive\Рабочий стол\elbrus\my_project_2_1\data\weather"
# SAVE_PATH = "models/resnet_weather.pt"
# LOG_PATH = "models/train_log.json"
# CONFUSION_PATH = "models/confusion.json"
# EPOCHS = 3
# BATCH_SIZE = 16

# # Трансформации
# transform = transforms.Compose([
#     transforms.Resize((224, 224)),
#     transforms.ToTensor(),
#     transforms.Normalize([0.485, 0.456, 0.406],
#                          [0.229, 0.224, 0.225])
# ])

# # Датасет и DataLoader
# dataset = datasets.ImageFolder(DATA_PATH, transform=transform)
# dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
# classes = dataset.classes
# print("Классы:", classes)

# # Модель
# model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
# for param in model.parameters():
#     param.requires_grad = False
# model.fc = nn.Linear(model.fc.in_features, len(classes))

# print(model.parameters())

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# model = model.to(device)

# criterion = nn.CrossEntropyLoss()
# optimizer = torch.optim.Adam(model.fc.parameters(), lr=0.001)

# # Логирование
# losses = []
# y_true_all = []
# y_pred_all = []

# start_time = time.time()

# # Обучение
# for epoch in range(EPOCHS):
#     model.train()
#     running_loss = 0

#     for images, labels in tqdm(dataloader, desc=f"Эпоха {epoch+1}/{EPOCHS}"):
#         images, labels = images.to(device), labels.to(device)

#         optimizer.zero_grad()
#         outputs = model(images)
#         loss = criterion(outputs, labels)
#         loss.backward()
#         optimizer.step()

#         running_loss += loss.item()

#         # Для метрик
#         preds = torch.argmax(outputs, dim=1)
#         y_true_all.extend(labels.cpu().tolist())
#         y_pred_all.extend(preds.cpu().tolist())

#     epoch_loss = running_loss
#     losses.append(epoch_loss)
#     print(f"[{epoch+1}] Loss: {epoch_loss:.4f}")

# # Время
# total_time = time.time() - start_time
# print("✅ Обучение завершено.")
# print(f"⏱️ Время обучения: {round(total_time / 60, 2)} минут")

# # Сохраняем модель
# os.makedirs("models", exist_ok=True)
# torch.save(model.state_dict(), SAVE_PATH)
# print("💾 Модель сохранена в:", SAVE_PATH)

# # Сохраняем лог
# train_log = {
#     "losses": losses,
#     "training_time": total_time
# }
# with open(LOG_PATH, "w") as f:
#     json.dump(train_log, f)
# print("📝 Лог обучения сохранён.")

# # Сохраняем confusion данные
# conf_data = {
#     "true": y_true_all,
#     "pred": y_pred_all
# }
# with open(CONFUSION_PATH, "w") as f:
#     json.dump(conf_data, f)
# print("🧩 Данные для confusion matrix сохранены.")



import os
import time
import json
import torch
from torch import nn
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader, random_split
from sklearn.metrics import accuracy_score
from tqdm import tqdm

# Пути
DATA_PATH = r"C:\Users\safar\OneDrive\Рабочий стол\elbrus\my_project_2_1\data\weather"
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

# Разделение на train и valid
train_size = int(0.8 * len(dataset))  # 80% для обучения
valid_size = len(dataset) - train_size  # 20% для валидации
train_dataset, valid_dataset = random_split(dataset, [train_size, valid_size])

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
valid_loader = DataLoader(valid_dataset, batch_size=BATCH_SIZE, shuffle=False)

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
train_losses, valid_losses = [], []
train_accuracies, valid_accuracies = [], []

start_time = time.time()

# Обучение
for epoch in range(EPOCHS):
    # Train
    model.train()
    running_loss = 0
    y_true_train, y_pred_train = [], []

    for images, labels in tqdm(train_loader, desc=f"Train Эпоха {epoch+1}/{EPOCHS}"):
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

        # Для метрик
        preds = torch.argmax(outputs, dim=1)
        y_true_train.extend(labels.cpu().tolist())
        y_pred_train.extend(preds.cpu().tolist())

    train_loss = running_loss / len(train_loader)
    train_accuracy = accuracy_score(y_true_train, y_pred_train)
    train_losses.append(train_loss)
    train_accuracies.append(train_accuracy)

    # Validation
    model.eval()
    running_loss = 0
    y_true_valid, y_pred_valid = [], []

    with torch.no_grad():  # Отключаем вычисление градиентов
        for images, labels in tqdm(valid_loader, desc=f"Valid Эпоха {epoch+1}/{EPOCHS}"):
            images, labels = images.to(device), labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)
            running_loss += loss.item()

            # Для метрик
            preds = torch.argmax(outputs, dim=1)
            y_true_valid.extend(labels.cpu().tolist())
            y_pred_valid.extend(preds.cpu().tolist())

    valid_loss = running_loss / len(valid_loader)
    valid_accuracy = accuracy_score(y_true_valid, y_pred_valid)
    valid_losses.append(valid_loss)
    valid_accuracies.append(valid_accuracy)

    # Вывод метрик
    print(f"[{epoch+1}] Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.4f}")
    print(f"[{epoch+1}] Valid Loss: {valid_loss:.4f}, Valid Accuracy: {valid_accuracy:.4f}")

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
    "train_losses": train_losses,
    "train_accuracies": train_accuracies,
    "valid_losses": valid_losses,
    "valid_accuracies": valid_accuracies,
    "training_time": total_time
}
with open(LOG_PATH, "w") as f:
    json.dump(train_log, f)
print("📝 Лог обучения сохранён.")

# Сохраняем confusion данные
conf_data = {
    "true": y_true_valid,
    "pred": y_pred_valid
}
with open(CONFUSION_PATH, "w") as f:
    json.dump(conf_data, f)
print("🧩 Данные для confusion matrix сохранены.")