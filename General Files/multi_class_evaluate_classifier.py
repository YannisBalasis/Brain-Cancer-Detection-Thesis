# scripts/evaluate_classifier.py
import os, json, argparse
import torch
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
from multi_class_data_loader import create_dataloaders
from multi_class_cnn_model import get_model
from multi_class_metrics import compute_basic_metrics

@torch.no_grad()
def collect_logits_targets(model, loader, device):
    model.eval()
    all_preds, all_targets = [], []
    for imgs, labels in tqdm(loader, desc="Test", leave=False):
        imgs = imgs.to(device)
        logits = model(imgs)
        preds = torch.argmax(logits, dim=1).cpu().numpy().tolist()
        all_preds.extend(preds)
        all_targets.extend(labels.numpy().tolist())
    return all_preds, all_targets

def plot_confusion_matrix(cm, labels, out_path):
    fig, ax = plt.subplots(figsize=(6,6))
    im = ax.imshow(cm, interpolation='nearest')
    ax.figure.colorbar(im, ax=ax)
    ax.set(xticks=np.arange(len(labels)), yticks=np.arange(len(labels)),
           xticklabels=labels, yticklabels=labels, ylabel='True', xlabel='Predicted')
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    # annotate
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], 'd'),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data_dir", type=str, default="data")
    p.add_argument("--weights", type=str, required=True)
    p.add_argument("--report_dir", type=str, default="reports/multiclass")
    args = p.parse_args()

    os.makedirs(args.report_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load state
    state = torch.load(args.weights, map_location=device)
    class_indices = state["class_indices"]
    labels = [class_indices[i] for i in sorted(class_indices.keys(), key=lambda x: int(x))]

    # Dataloaders (eval tfms only; augment handled inside)
    loaders, _, _ = create_dataloaders(args.data_dir, augment="none")
    num_classes = len(labels)

    # Model
    model = get_model(num_classes=num_classes, backbone="custom")
    model.load_state_dict(state["model"]); model.to(device)

    # Collect preds/targets from test set
    preds, targets = collect_logits_targets(model, loaders["test"], device)

    # Metrics
    from sklearn.metrics import confusion_matrix
    cm = confusion_matrix(targets, preds)
    metrics = compute_basic_metrics(targets, preds, labels)
    with open(os.path.join(args.report_dir, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    # Save classification report (pretty)
    from sklearn.metrics import classification_report
    report_txt = classification_report(targets, preds, target_names=labels, zero_division=0)
    with open(os.path.join(args.report_dir, "classification_report.txt"), "w") as f:
        f.write(report_txt)

    # Confusion matrix plot
    plot_confusion_matrix(cm, labels, os.path.join(args.report_dir, "confusion_matrix.png"))
    print("Saved metrics and confusion matrix to", args.report_dir)

if __name__ == "__main__":
    main()
