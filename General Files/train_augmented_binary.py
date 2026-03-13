import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from binary_model import BinaryCNN  # ✅ import καθαρής αρχιτεκτονικής
import torch.nn.functional as F


# 🔁 Βοηθητική Κλάση για Binary Labels
class BinaryLabelDataset(ImageFolder):
    def __getitem__(self, index):
        path, target = self.samples[index]
        image = self.loader(path)
        if self.transform is not None:
            image = self.transform(image)
        label_name = self.classes[target]
        binary_label = 0 if label_name == 'notumor' else 1
        return image, binary_label

# 📁 Paths
dataset_dir = "/Users/yannisbalasis/Documents/thesis/dataset_binary_class"
train_dir = os.path.join(dataset_dir, "Training")

# 🧪 Augmented Transforms
transform_augmented = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomRotation(10),
    transforms.RandomHorizontalFlip(),
    transforms.RandomAffine(degrees=0, translate=(0.05, 0.05)),
    transforms.Grayscale(),
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])
])

# 📂 Dataset
train_dataset = BinaryLabelDataset(train_dir, transform=transform_augmented)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

# 🧠 Model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = BinaryCNN().to(device)
criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.parameters(), lr=0.0005)

# 🏋️ Training
epochs = 10
for epoch in range(epochs):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for inputs, labels in train_loader:
        inputs = inputs.to(device)
        labels = labels.to(device).float().unsqueeze(1)
        optimizer.zero_grad()
        outputs = torch.sigmoid(model(inputs))
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

# 💾 Save Model
torch.save(model.state_dict(), "binary_cnn_augmented.pth")
print("✔️ Model saved as binary_cnn_augmented.pth")
