import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import numpy as np
from binary_model import BinaryCNN
from train_binary import BinaryLabelDataset

# ⚙️ Config
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dataset_path = "/Users/yannisbalasis/Documents/thesis/dataset_binary_class/Training"
num_epochs = 5
batch_size = 32
k_folds = 5

# 🔄 Transform
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.Grayscale(),
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])
])

# 📦 Dataset
full_dataset = BinaryLabelDataset(dataset_path, transform=transform)

# 📁 Folds
kfold = KFold(n_splits=k_folds, shuffle=True, random_state=42)

# 📊 Metrics storage
all_acc, all_prec, all_rec, all_f1 = [], [], [], []

print(f"🔎 Starting {k_folds}-Fold Cross-Validation...\n")

for fold, (train_ids, val_ids) in enumerate(kfold.split(full_dataset)):
    print(f"🔁 Fold {fold+1}")

    # Dataset split
    train_subset = torch.utils.data.Subset(full_dataset, train_ids)
    val_subset = torch.utils.data.Subset(full_dataset, val_ids)

    train_loader = torch.utils.data.DataLoader(train_subset, batch_size=batch_size, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_subset, batch_size=batch_size, shuffle=False)

    # 🧠 Model init
    model = BinaryCNN().to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # 🏋️ Train
    for epoch in range(num_epochs):
        model.train()
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device).float().unsqueeze(1)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

    # 🧪 Evaluate
    model.eval()
    y_true = []
    y_pred = []
    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            preds = (outputs > 0.5).int().cpu().numpy().flatten()
            y_pred.extend(preds)
            y_true.extend(labels.numpy())

    # 📈 Metrics
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred)
    rec = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)

    print(f"📌 Fold {fold+1} - Acc: {acc:.4f} | Prec: {prec:.4f} | Rec: {rec:.4f} | F1: {f1:.4f}\n")

    all_acc.append(acc)
    all_prec.append(prec)
    all_rec.append(rec)
    all_f1.append(f1)

# 📊 Overall Results
print("✅ Cross-Validation Results:")
print(f"Accuracy:   {np.mean(all_acc):.4f} ± {np.std(all_acc):.4f}")
print(f"Precision:  {np.mean(all_prec):.4f} ± {np.std(all_prec):.4f}")
print(f"Recall:     {np.mean(all_rec):.4f} ± {np.std(all_rec):.4f}")
print(f"F1 Score:   {np.mean(all_f1):.4f} ± {np.std(all_f1):.4f}")

import matplotlib.pyplot as plt

# 📊 Boxplot για όλες τις μετρικές
metrics = {
    "Accuracy": all_acc,
    "Precision": all_prec,
    "Recall": all_rec,
    "F1 Score": all_f1
}

plt.figure(figsize=(10, 6))
plt.boxplot(metrics.values(), labels=metrics.keys(), patch_artist=True)
plt.title(f"{k_folds}-Fold Cross-Validation Performance")
plt.ylabel("Score")
plt.grid(True)
plt.ylim(0.8, 1.01)  # Προσαρμόζεται ανάλογα με το performance
plt.tight_layout()
plt.show()
