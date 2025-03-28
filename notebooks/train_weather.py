import os
import torch
from torch import nn
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
from tqdm import tqdm

# Путь к данным
DATA_PATH = r"C:\Users\Galin22\.cache\kagglehub\datasets\jehanbhathena\weather-dataset\versions\3\dataset"
SAVE_PATH = "models/resnet_weather.pt"
BATCH_SIZE = 16
EPOCHS = 3  # Можно потом увеличить

# Трансформации
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

# Загрузка датасета
dataset = datasets.ImageFolder(DATA_PATH, transform=transform)
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

# Названия классов
classes = dataset.classes
print("Классы:", classes)

# Модель
model = models.resnet18(pretrained=True)

# Замораживаем все слои
for param in model.parameters():
    param.requires_grad = False

# Меняем классификатор
num_classes = len(classes)
model.fc = nn.Linear(model.fc.in_features, num_classes)

# Устройство
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

# Оптимизатор и функция потерь
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.fc.parameters(), lr=0.001)

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

    print(f"[{epoch+1}] Loss: {running_loss:.4f}")

# Сохраняем модель
os.makedirs("models", exist_ok=True)
torch.save(model.state_dict(), SAVE_PATH)
print("✅ Модель сохранена в:", SAVE_PATH)
