import json
import matplotlib.pyplot as plt
import os

# path στο history που έχει σωθεί από το training
history_path = "reports/multiclass/training_history.json"

with open(history_path, "r") as f:
    history = json.load(f)

train_loss = history["train_loss"]
val_loss   = history["val_loss"]
train_acc  = history["train_acc"]
val_acc    = history["val_acc"]

epochs = range(1, len(train_loss) + 1)

# Plot Loss
plt.figure(figsize=(8,6))
plt.plot(epochs, train_loss, label="Train Loss")
plt.plot(epochs, val_loss, label="Val Loss")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.title("Training and Validation Loss")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("reports/multiclass/loss_curve.png", dpi=300)
plt.show()

# Plot Accuracy
plt.figure(figsize=(8,6))
plt.plot(epochs, train_acc, label="Train Accuracy")
plt.plot(epochs, val_acc, label="Val Accuracy")
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.title("Training and Validation Accuracy")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("reports/multiclass/accuracy_curve.png", dpi=300)
plt.show()
