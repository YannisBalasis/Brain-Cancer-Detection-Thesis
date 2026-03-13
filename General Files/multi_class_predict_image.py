import os, argparse, json
import torch
import numpy as np
from PIL import Image
from torchvision import transforms
import matplotlib.pyplot as plt

from multi_class_cnn_model import get_model

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD  = (0.229, 0.224, 0.225)

def preprocess(img_path, img_size):
    tfms = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])
    img = Image.open(img_path).convert("RGB")
    return tfms(img).unsqueeze(0)

def visualize(img_path, labels, probs, pred_label, save_path=None, show=True):
    # ταξινόμηση κατά πιθανότητα (φθίνουσα)
    order = np.argsort(probs)[::-1]
    probs_sorted  = np.array(probs)[order]
    labels_sorted = [labels[i] for i in order]

    # άνοιγμα αρχικής εικόνας για εμφάνιση
    img = Image.open(img_path).convert("RGB")

    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    # αριστερά: εικόνα
    axes[0].imshow(img)
    axes[0].axis("off")
    axes[0].set_title(f"Input image")

    # δεξιά: bar chart πιθανοτήτων
    axes[1].barh(labels_sorted[::-1], probs_sorted[::-1])
    axes[1].set_xlim(0, 1)
    axes[1].set_xlabel("Probability")
    axes[1].set_title(f"Prediction: {pred_label}")
    for i, p in enumerate(probs_sorted[::-1]):
        axes[1].text(p + 0.01, i, f"{p:.3f}", va="center")

    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Saved visualization to: {save_path}")
    if show:
        plt.show()
    else:
        plt.close(fig)

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--weights", type=str, required=True, help="Path στο .pth checkpoint")
    p.add_argument("--image",   type=str, required=True, help="Path εικόνας (π.χ. .jpg/.png)")
    p.add_argument("--no_show", action="store_true", help="Μην ανοίξεις παράθυρο matplotlib")
    p.add_argument("--save_vis", type=str, default="", help="Αν δοθεί, αποθηκεύει το plot εδώ (π.χ. reports/prediction.png)")
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Φόρτωση checkpoint
    state = torch.load(args.weights, map_location=device)
    class_indices = state["class_indices"]  # dict: "0":"glioma", ...
    labels = [class_indices[i] for i in sorted(class_indices.keys(), key=lambda x: int(x))]
    num_classes = len(labels)
    img_size = state.get("img_size", 224)

    # Μοντέλο
    model = get_model(num_classes=num_classes, backbone="custom")
    model.load_state_dict(state["model"])
    model.to(device).eval()

    # Preprocess & inference
    x = preprocess(args.image, img_size).to(device)
    with torch.no_grad():
        logits = model(x)
        probs = torch.softmax(logits, dim=1).cpu().numpy().squeeze().tolist()

    top_idx = int(torch.argmax(logits, dim=1).item())
    pred_label = labels[top_idx]

    # Εκτύπωση στην κονσόλα
    print(f"Prediction: {pred_label}")
    print("Probabilities:")
    for i, p in enumerate(probs):
        print(f"  {labels[i]}: {p:.4f}")

    # Οπτικοποίηση (εικόνα + bar chart)
    save_path = args.save_vis if args.save_vis else None
    visualize(args.image, labels, probs, pred_label, save_path=save_path, show=not args.no_show)

if __name__ == "__main__":
    main()
