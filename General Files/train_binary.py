import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from PIL import Image
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.nn.functional as F
import torch
from binary_model import BinaryCNN


# 🔁 Βοηθητική Κλάση για Binary Labels
class BinaryLabelDataset(ImageFolder):
    def __getitem__(self, index):
        path, target = self.samples[index]
        image = self.loader(path)
        if self.transform is not None:
            image = self.transform(image)
        # 0 = notumor, 1 = tumor (glioma, meningioma, pituitary)
        label_name = self.classes[target]
        binary_label = 0 if label_name == 'notumor' else 1
        return image, binary_label

# 📁 Paths
dataset_dir = "/Users/yannisbalasis/Documents/thesis/dataset_binary_class"
train_dir = os.path.join(dataset_dir, "Training")
test_dir = os.path.join(dataset_dir, "Testing")

# 🌀 Image Transformations
train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.Grayscale(),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(degrees=15),
    transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])
])

test_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.Grayscale(),
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])
])


# 📂 Load Datasets
train_dataset = BinaryLabelDataset(train_dir, transform=train_transform)
test_dataset = BinaryLabelDataset(test_dir, transform=test_transform)

print("Classes (Binary):", train_dataset.classes)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32)



# 🧠 Initialize model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = BinaryCNN().to(device)
criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 🏋️ Training loop
epochs = 10
for epoch in range(epochs):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device).float().unsqueeze(1)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        predicted = (outputs > 0.5).float()
        correct += (predicted == labels).sum().item()
        total += labels.size(0)

    epoch_loss = running_loss / len(train_loader)
    epoch_acc = 100 * correct / total
    print(f"[Epoch {epoch+1}] Loss: {epoch_loss:.4f} | Accuracy: {epoch_acc:.2f}%")

# 💾 Save model
torch.save(model.state_dict(), "binary_cnn.pth")
print("✔️ Model saved as binary_cnn.pth")
