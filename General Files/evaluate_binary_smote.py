import os
import torch
from torchvision import transforms
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from train_binary_all import BinaryLabelDataset
from binary_model import BinaryCNN


from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

#  Paths
test_dir = "/Users/yannisbalasis/Documents/thesis/balanced_split/test"


#  Transform
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.Grayscale(),
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])
])

#  Dataset
test_dataset = BinaryLabelDataset(test_dir, transform=transform)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# Load Model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = BinaryCNN().to(device)
model.load_state_dict(torch.load("binary_cnn_smote.pth", map_location=device))
model.eval()

#  Evaluate
all_preds = []
all_labels = []

with torch.no_grad():
    for inputs, labels in test_loader:
        inputs = inputs.to(device)
        outputs = model(inputs)
        preds = (outputs > 0.5).int().cpu().numpy().flatten()
        all_preds.extend(preds)
        all_labels.extend(labels.numpy())

#  Results
print("Classification Report:")
print(classification_report(all_labels, all_preds, target_names=["notumor", "tumor"]))

# Confusion Matrix
cm = confusion_matrix(all_labels, all_preds)
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["notumor", "tumor"], yticklabels=["notumor", "tumor"])
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.show()
