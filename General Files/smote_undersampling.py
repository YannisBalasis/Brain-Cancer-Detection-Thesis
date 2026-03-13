import os
import numpy as np
from PIL import Image
from imblearn.combine import SMOTEENN
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
import torch

# Paths
original_data_dir = "/Users/yannisbalasis/Documents/thesis/dataset_binary_class_split/Training"
output_dir = "/Users/yannisbalasis/Documents/thesis/smote_balanced_data"

# Ensure output folders exist
for class_name in ["notumor", "tumor"]:
    os.makedirs(os.path.join(output_dir, class_name), exist_ok=True)

# Preprocessing (same as training)
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.Grayscale(),
    transforms.ToTensor()
])

# Load original dataset
dataset = ImageFolder(original_data_dir, transform=transform)
dataloader = DataLoader(dataset, batch_size=len(dataset), shuffle=False)

# Load data into memory
for inputs, labels in dataloader:
    X = inputs.view(inputs.size(0), -1).numpy()  # Flatten images
    y = labels.numpy()
    break  # One batch only

# Apply SMOTE + ENN
smote_enn = SMOTEENN(random_state=42, n_jobs=-1)
X_resampled, y_resampled = smote_enn.fit_resample(X, y)

print("Resampled shape:", X_resampled.shape)

# Reshape back to image format: (N, 1, 224, 224)
X_images = X_resampled.reshape(-1, 1, 224, 224)

# Save images to new folder structure
for i, (img_arr, label) in enumerate(zip(X_images, y_resampled)):
    img_tensor = torch.tensor(img_arr)
    img_pil = transforms.ToPILImage()(img_tensor)

    label_name = dataset.classes[label]  # "notumor" or "tumor"
    filename = f"img_{i:05d}.jpg"
    save_path = os.path.join(output_dir, label_name, filename)
    img_pil.save(save_path)

print(" Αποθήκευση ολοκληρώθηκε στους φακέλους:", output_dir)
