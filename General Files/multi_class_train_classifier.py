# scripts/train_classifier.py
import os, json, argparse, random
import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam, SGD
from torch.optim.lr_scheduler import StepLR, CosineAnnealingLR
from tqdm import tqdm

from multi_class_data_loader import create_dataloaders
from multi_class_cnn_model import get_model

def set_seed(seed: int = 42):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True; torch.backends.cudnn.benchmark = False

def count_class_samples(dataloader):
    counts = {}
    for _, targets in dataloader:
        for t in targets.tolist():
            counts[t] = counts.get(t, 0) + 1
    return counts

def get_class_weights(train_loader, num_classes):
    counts = count_class_samples(train_loader)
    total = sum(counts.get(i, 0) for i in range(num_classes))
    weights = [total / max(1, counts.get(i, 1)) for i in range(num_classes)]
    w = torch.tensor(weights, dtype=torch.float32)
    return w / w.sum() * num_classes  # normalize around 1.0

def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    running_loss, correct, total = 0.0, 0, 0
    for imgs, labels in tqdm(loader, desc="Train", leave=False):
        imgs, labels = imgs.to(device), labels.to(device)
        optimizer.zero_grad()
        logits = model(imgs)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * imgs.size(0)
        preds = logits.argmax(1)
        correct += (preds == labels).sum().item()
        total += imgs.size(0)
    return running_loss/total, correct/total

@torch.no_grad()
def eval_one_epoch(model, loader, criterion, device):
    model.eval()
    running_loss, correct, total = 0.0, 0, 0
    for imgs, labels in tqdm(loader, desc="Val", leave=False):
        imgs, labels = imgs.to(device), labels.to(device)
        logits = model(imgs)
        loss = criterion(logits, labels)
        running_loss += loss.item() * imgs.size(0)
        preds = logits.argmax(1)
        correct += (preds == labels).sum().item()
        total += imgs.size(0)
    return running_loss/total, correct/total


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data_dir", type=str, default="data")
    p.add_argument("--img_size", type=int, default=224)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--augment", type=str, default="light", choices=["none","light"])
    p.add_argument("--optimizer", type=str, default="adam", choices=["adam","sgd"])
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--weight_decay", type=float, default=5e-4)
    p.add_argument("--scheduler", type=str, default="steplr", choices=["none","steplr","cosine"])
    p.add_argument("--step_size", type=int, default=10)
    p.add_argument("--gamma", type=float, default=0.1)
    p.add_argument("--early_stopping", type=int, default=10)
    p.add_argument("--dropout", type=float, default=0.4)
    p.add_argument("--backbone", type=str, default="custom_plus")
    p.add_argument("--save_path", type=str, default="models/best_multiclass.pth")
    p.add_argument("--report_dir", type=str, default="reports/multiclass")
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    os.makedirs(os.path.dirname(args.save_path), exist_ok=True)
    os.makedirs(args.report_dir, exist_ok=True)

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    loaders, class_to_idx, img_size = create_dataloaders(
        data_root=args.data_dir,
        img_size=args.img_size,
        batch_size=args.batch_size,
        num_workers=4,
        augment=args.augment
    )
    idx_to_class = {v:k for k,v in class_to_idx.items()}
    with open(os.path.join(args.report_dir, "class_indices.json"), "w") as f:
        json.dump(idx_to_class, f, indent=2)

    num_classes = len(class_to_idx)
    model = get_model(num_classes=num_classes, backbone=args.backbone, dropout=args.dropout).to(device)

    # class weights (optional but helpful)
    class_weights = get_class_weights(loaders["train"], num_classes).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    if args.optimizer == "adam":
        optimizer = Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    else:
        optimizer = SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=args.weight_decay, nesterov=True)

    if args.scheduler == "steplr":
        scheduler = StepLR(optimizer, step_size=args.step_size, gamma=args.gamma)
    elif args.scheduler == "cosine":
        scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs)
    else:
        scheduler = None

    best_val_acc, best_state, patience = -1.0, None, args.early_stopping
    history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}

    for epoch in range(1, args.epochs+1):
        print(f"\nEpoch {epoch}/{args.epochs}")
        tr_loss, tr_acc = train_one_epoch(model, loaders["train"], criterion, optimizer, device)
        val_loss, val_acc = eval_one_epoch(model, loaders["val"], criterion, device)
        history["train_loss"].append(tr_loss); history["val_loss"].append(val_loss)
        history["train_acc"].append(tr_acc);   history["val_acc"].append(val_acc)

        print(f"Train: loss={tr_loss:.4f} acc={tr_acc:.4f} | Val: loss={val_loss:.4f} acc={val_acc:.4f}")

        if scheduler is not None:
            scheduler.step()

        # Early stopping on val_acc
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state = {
                "model": model.state_dict(),
                "class_indices": idx_to_class,
                "img_size": img_size,
                "args": vars(args)
            }
            torch.save(best_state, args.save_path)
            patience = args.early_stopping
            print(f"✔ Saved new best model to {args.save_path}")
        else:
            patience -= 1
            if patience == 0:
                print("Early stopping triggered.")
                break

    # Save history
    with open(os.path.join(args.report_dir, "training_history.json"), "w") as f:
        json.dump(history, f, indent=2)

if __name__ == "__main__":
    main()
