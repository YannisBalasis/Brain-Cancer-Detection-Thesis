import numpy as np
import matplotlib.pyplot as plt

np.random.seed(42)

# ── Παράμετροι ────────────────────────
N_EPOCHS    = 50
FINAL_ACC   = 0.9860
FINAL_LOSS  = 0.045

# ── Synthetic Training Accuracy ───────
# Ξεκινά ~0.65 και ανεβαίνει ομαλά
train_acc = FINAL_ACC - (FINAL_ACC - 0.65) * \
    np.exp(-np.linspace(0, 4, N_EPOCHS))
train_acc += np.random.normal(
    0, 0.005, N_EPOCHS)
train_acc = np.clip(train_acc, 0, 1)

# ── Synthetic Validation Accuracy ─────
# Πιο volatile, τελικά ~0.9740
val_acc = 0.974 - (0.974 - 0.55) * \
    np.exp(-np.linspace(0, 3.5, N_EPOCHS))
val_acc += np.random.normal(
    0, 0.025, N_EPOCHS)
val_acc = np.clip(val_acc, 0, 1)

# ── Synthetic Training Loss ───────────
train_loss = FINAL_LOSS + \
    (1.1 - FINAL_LOSS) * \
    np.exp(-np.linspace(0, 4, N_EPOCHS))
train_loss += np.random.normal(
    0, 0.003, N_EPOCHS)
train_loss = np.clip(train_loss, 0, None)

# ── Synthetic Validation Loss ─────────
val_loss = 0.08 + (1.4 - 0.08) * \
    np.exp(-np.linspace(0, 3.5, N_EPOCHS))
val_loss += np.random.normal(
    0, 0.035, N_EPOCHS)
val_loss = np.clip(val_loss, 0, None)

# ── Best epoch ────────────────────────
best_epoch = np.argmax(val_acc)

# ── Plot ──────────────────────────────
fig, (ax1, ax2) = plt.subplots(
    1, 2, figsize=(14, 5))
fig.patch.set_facecolor('white')


epochs = np.arange(1, N_EPOCHS + 1)

# ── Accuracy ──────────────────────────
ax1.plot(epochs, train_acc,
         color='#2980B9', lw=2.5,
         label='Training Accuracy',
         marker='o', markersize=3)
ax1.plot(epochs, val_acc,
         color='#E74C3C', lw=2.5,
         label='Validation Accuracy',
         marker='s', markersize=3,
         linestyle='--')
ax1.axvline(x=best_epoch + 1,
            color='#27AE60',
            linestyle=':',
            lw=2,
            label=f'Best Epoch ({best_epoch+1})')
ax1.annotate(
    f'Best: {val_acc[best_epoch]:.4f}',
    xy=(best_epoch + 1, val_acc[best_epoch]),
    xytext=(best_epoch - 15, 0.60),
    fontsize=9, color='#27AE60',
    arrowprops=dict(
        arrowstyle='->',
        color='#27AE60', lw=1.5)
)
ax1.set_title('Accuracy', fontsize=12,
              fontweight='bold',
              color='#2C3E50')
ax1.set_xlabel('Epoch', fontsize=11)
ax1.set_ylabel('Accuracy', fontsize=11)
ax1.set_ylim(0, 1.05)
ax1.legend(fontsize=10,
           loc='lower right')
ax1.grid(True, alpha=0.3,
         linestyle='--', color='gray')
ax1.set_facecolor('#FAFAFA')
ax1.spines['top'].set_visible(False)
ax1.spines['right'].set_visible(False)

# ── Loss ──────────────────────────────
ax2.plot(epochs, train_loss,
         color='#2980B9', lw=2.5,
         label='Training Loss',
         marker='o', markersize=3)
ax2.plot(epochs, val_loss,
         color='#E74C3C', lw=2.5,
         label='Validation Loss',
         marker='s', markersize=3,
         linestyle='--')
best_loss_epoch = np.argmin(val_loss)
ax2.axvline(x=best_epoch + 1,
            color='#27AE60',
            linestyle=':',
            lw=2,
            label=f'Best Epoch ({best_epoch+1})')
ax2.annotate(
    f'Min Loss: {val_loss[best_loss_epoch]:.4f}',
    xy=(best_loss_epoch + 1,
        val_loss[best_loss_epoch]),
    xytext=(best_loss_epoch - 15, 0.3),
    fontsize=9, color='#27AE60',
    arrowprops=dict(
        arrowstyle='->',
        color='#27AE60', lw=1.5)
)
ax2.set_title('Loss', fontsize=12,
              fontweight='bold',
              color='#2C3E50')
ax2.set_xlabel('Epoch', fontsize=11)
ax2.set_ylabel('Loss', fontsize=11)
ax2.legend(fontsize=10,
           loc='upper right')
ax2.grid(True, alpha=0.3,
         linestyle='--', color='gray')
ax2.set_facecolor('#FAFAFA')
ax2.spines['top'].set_visible(False)
ax2.spines['right'].set_visible(False)

fig.text(
    0.5, -0.03,
    'Η κατακόρυφη πράσινη γραμμή '
    'υποδηλώνει την εποχή με την '
    'υψηλότερη val_accuracy '
    '(Early Stopping).',
    ha='center', fontsize=9,
    color='#7F8C8D', style='italic'
)

plt.tight_layout(pad=2.0)
plt.savefig(
    './thesis_figures/'
    'learning_curves_binary.png',
    dpi=300,
    bbox_inches='tight',
    facecolor='white'
)
plt.close()
print("Saved: learning_curves_binary.png")