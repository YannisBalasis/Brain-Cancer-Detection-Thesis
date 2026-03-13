import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import cv2
from binary_model import BinaryCNN

# Config
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_path = "binary_cnn_smote.pth"
image_path = "/Users/yannisbalasis/Documents/thesis/dataset_binary_class_split/Testing/tumor/Te-pi_0080.jpg"
classes = ["notumor", "tumor"]

# Load Model
model = BinaryCNN()
model.load_state_dict(torch.load(model_path, map_location=device))
model.to(device)
model.eval()

# Hooks
gradients = []
activations = []

def forward_hook(module, input, output):

    activations.append(output)

def backward_hook(module, grad_input, grad_output):

    gradients.append(grad_output[0])


# Register hooks
target_layer = model.conv2
target_layer.register_forward_hook(forward_hook)
target_layer.register_full_backward_hook(backward_hook)

# Preprocess image
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.Grayscale(),
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])
])

img = Image.open(image_path).convert("L")

input_tensor = transform(img).unsqueeze(0).to(device)

# Forward
output = model(input_tensor)
probability = torch.sigmoid(output).item()
pred_class = 1 if probability > 0.5 else 0

print(f"Prediction: {classes[pred_class]}")

# Backward
model.zero_grad()
output.squeeze().backward()

# Grad-CAM Calculation
if len(gradients) == 0 or len(activations) == 0:
    raise ValueError(" Hooks did not trigger. Check target_layer and model architecture.")

grads_val = gradients[0].cpu().data.numpy()[0]
activations_val = activations[0].cpu().data.numpy()[0]
print("Activations collected:", len(activations))
print("Gradients collected:", len(gradients))

if len(gradients) > 0:
    print("Gradient shape:", gradients[0].shape)
    print("Gradient stats → min:", gradients[0].min().item(), 
          "max:", gradients[0].max().item(), 
          "mean:", gradients[0].mean().item())

if len(activations) > 0:
    print("Activation shape:", activations[0].shape)
    print("Activation stats → min:", activations[0].min().item(), 
          "max:", activations[0].max().item(), 
          "mean:", activations[0].mean().item())

weights = np.mean(grads_val, axis=(1, 2))

cam = np.zeros(activations_val.shape[1:], dtype=np.float32)
for i, w in enumerate(weights):
    cam += w * activations_val[i]

cam = np.maximum(cam, 0)
cam = cv2.GaussianBlur(cam, (5, 5), 0)
cam = cv2.resize(cam, (224, 224))
if np.max(cam) != 0:
    cam = cam / np.max(cam)

# Visualization
img_np = np.array(img.resize((224, 224)).convert("RGB"), dtype=np.float32) / 255.0
heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
heatmap = np.float32(heatmap) / 255
overlay = 0.4 * heatmap + 0.6 * img_np
overlay = overlay / np.max(overlay)

#  Show
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.imshow(img_np)
plt.title(f"Prediction: {classes[pred_class]}")
plt.axis("off")

plt.subplot(1, 2, 2)
plt.imshow(overlay)
plt.title("Grad-CAM")
plt.axis("off")
plt.tight_layout()
plt.show()

print("CAM stats → min:", np.min(cam), "max:", np.max(cam), "mean:", np.mean(cam))
