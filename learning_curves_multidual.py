import json
import matplotlib.pyplot as plt
import numpy as np

with open('./multi_dual_system_experiment_20260310_100816'
          '/training_history.json') as f:
    history = json.load(f)

# Fusion output accuracy
train_acc = history['fusion_output_fusion_accuracy']
val_acc   = history['val_fusion_output_fusion_accuracy']
train_loss = history['loss']
val_loss   = history['val_loss']

epochs = list(range(1, len(train_acc) + 1))
best_epoch = int(np.argmax(val_acc))

fig, (ax1, ax2) = plt.subplots(
    1, 2, figsize=(14, 5))
fig.patch.set_facecolor('white')
fig.suptitle(
    'Learning Curves — Multi-Dual System\n'
    '(Fusion Output)',
    fontsize=14, fontweight='bold',
    color='#2C3E50', y=1.02)

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
            linestyle=':', lw=2,
            label=f'Best ({best_epoch+1})')
ax1.annotate(
    f'Best: {val_acc[best_epoch]:.4f}',
    xy=(best_epoch + 1, val_acc[best_epoch]),
    xytext=(best_epoch - 10, 0.60),
    fontsize=9, color='#27AE60',
    arrowprops=dict(
        arrowstyle='->', color='#27AE60',
        lw=1.5))
ax1.set_title('Accuracy (Fusion Output)',
              fontsize=12, fontweight='bold',
              color='#2C3E50')
ax1.set_xlabel('Epoch', fontsize=11)
ax1.set_ylabel('Accuracy', fontsize=11)
ax1.set_ylim(0, 1.05)
ax1.legend(fontsize=10, loc='lower right')
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
ax2.axvline(x=best_epoch + 1,
            color='#27AE60',
            linestyle=':', lw=2,
            label=f'Best ({best_epoch+1})')
best_loss_epoch = int(np.argmin(val_loss))
ax2.annotate(
    f'Min: {val_loss[best_loss_epoch]:.4f}',
    xy=(best_loss_epoch + 1,
        val_loss[best_loss_epoch]),
    xytext=(best_loss_epoch - 10, 0.5),
    fontsize=9, color='#27AE60',
    arrowprops=dict(
        arrowstyle='->', color='#27AE60',
        lw=1.5))
ax2.set_title('Total Loss', fontsize=12,
              fontweight='bold', color='#2C3E50')
ax2.set_xlabel('Epoch', fontsize=11)
ax2.set_ylabel('Loss', fontsize=11)
ax2.legend(fontsize=10, loc='upper right')
ax2.grid(True, alpha=0.3,
         linestyle='--', color='gray')
ax2.set_facecolor('#FAFAFA')
ax2.spines['top'].set_visible(False)
ax2.spines['right'].set_visible(False)

plt.tight_layout(pad=2.0)
plt.savefig(
    './thesis_figures/'
    'learning_curves_multidual.png',
    dpi=300, bbox_inches='tight',
    facecolor='white')
plt.close()
print("Saved: learning_curves_multidual.png")