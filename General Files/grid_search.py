import os

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader, random_split
from binary_model import BinaryCNN
import numpy as np

# 🔧 Grid parameters
learning_rates = [0.001, 0.0005]
batch_sizes = [16, 32]
optimizers = ['adam', 'sgd']
epochs = 5

# 📁 Dataset path
dataset_dir = "/Users/yannisbalasis/Documents/thesis/dataset_binary_class/Training"

# 🌀 Transform
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.Grayscale(),
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])
])

# 📂 Load dataset
full_dataset = ImageFolder(dataset_dir, transform=transform)

# 🔁 Binary label wrapper
class BinaryLabelDataset(torch.utils.data.Dataset):
    def __init__(self, dataset):
        self.dataset = dataset
        self.classes = dataset.classes

    def __getitem__(self, idx):
        image, label = self.dataset[idx]
        label_name = self.classes[label]
        binary_label = 0 if label_name == 'notumor' else 1
        return image, torch.tensor(binary_label).float()

    def __len__(self):
        return len(self.dataset)

# ⏱ Evaluation
def evaluate(model, dataloader, device):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device).unsqueeze(1)
            outputs = model(inputs)
            predicted = (outputs > 0.5).float()
            correct += (predicted == labels).sum().item()
            total += labels.size(0)
    return correct / total

# 🚀 Grid Search
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
results = []

for lr in learning_rates:
    for bs in batch_sizes:
        for opt in optimizers:
            print(f"🔍 Testing: lr={lr}, batch_size={bs}, optimizer={opt}")

            # Reload dataset and split
            dataset = BinaryLabelDataset(full_dataset)
            train_size = int(0.8 * len(dataset))
            val_size = len(dataset) - train_size
            train_set, val_set = random_split(dataset, [train_size, val_size])

            train_loader = DataLoader(train_set, batch_size=bs, shuffle=True)
            val_loader = DataLoader(val_set, batch_size=bs)

            # Model
            model = BinaryCNN().to(device)
            criterion = nn.BCEWithLogitsLoss()
            if opt == 'adam':
                optimizer = optim.Adam(model.parameters(), lr=lr)
            else:
                optimizer = optim.SGD(model.parameters(), lr=lr)

            for epoch in range(epochs):
                model.train()
                for inputs, labels in train_loader:
                    inputs, labels = inputs.to(device), labels.to(device).unsqueeze(1)
                    optimizer.zero_grad()
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                    loss.backward()
                    optimizer.step()

            val_acc = evaluate(model, val_loader, device)
            print(f" Accuracy: {val_acc:.4f}")
            results.append((lr, bs, opt, val_acc))

# 🔚 Summary
best = max(results, key=lambda x: x[3])
print(f" Best: lr={best[0]}, batch_size={best[1]}, optimizer={best[2]}, val_acc={best[3]:.4f}")



