import json
import matplotlib.pyplot as plt
import numpy as np

phases = {}
for i in [1, 2, 3]:
    path = (f'./efnet_dual_system_experiment_20260305_194201'
            f'/results/efnet_phase{i}_history.json')
    with open(path) as f:
        phases[i] = json.load(f)

train_acc  = []
val_acc    = []
train_loss = []
val_loss   = []

for i in [1, 2, 3]:
    train_acc  += phases[i]['accuracy']
    val_acc    += phases[i]['val_accuracy']
    train_loss += phases[i]['loss']
    val_loss   += phases[i]['val_loss']

epochs = list(range(1, len(train_acc) + 1))
p1_end = len(phases[1]['accuracy'])
p2_end = p1_end + len(phases[2]['accuracy'])
best_epoch = int(np.argmax(val_acc))

fig, (ax1, ax2) = plt.subplots(
    1, 2, figsize=(14, 5))
fig.patch.set_facecolor('white')
fig.suptitle(
    'Learning Curves — EfficientNet-B3 Dual Branch',
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
ax1.axvline(x=p1_end, color='#8E44AD',
            linestyle=':', lw=1.5,
            label='Phase 1→2')
ax1.axvline(x=p2_end, color='#F39C12',
            linestyle=':', lw=1.5,
            label='Phase 2→3')
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
ax1.set_title('Accuracy', fontsize=12,
              fontweight='bold', color='#2C3E50')
ax1.set_xlabel('Epoch', fontsize=11)
ax1.set_ylabel('Accuracy', fontsize=11)
ax1.set_ylim(0, 1.05)
ax1.legend(fontsize=9, loc='lower right')
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
ax2.axvline(x=p1_end, color='#8E44AD',
            linestyle=':', lw=1.5,
            label='Phase 1→2')
ax2.axvline(x=p2_end, color='#F39C12',
            linestyle=':', lw=1.5,
            label='Phase 2→3')
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
ax2.set_title('Loss', fontsize=12,
              fontweight='bold', color='#2C3E50')
ax2.set_xlabel('Epoch', fontsize=11)
ax2.set_ylabel('Loss', fontsize=11)
ax2.legend(fontsize=9, loc='upper right')
ax2.grid(True, alpha=0.3,
         linestyle='--', color='gray')
ax2.set_facecolor('#FAFAFA')
ax2.spines['top'].set_visible(False)
ax2.spines['right'].set_visible(False)

fig.text(
    0.5, -0.03,
    'Κατακόρυφες γραμμές: '
    'μωβ=Phase 1→2, '
    'πορτοκαλί=Phase 2→3, '
    'πράσινο=Best epoch.',
    ha='center', fontsize=9,
    color='#7F8C8D', style='italic')

plt.tight_layout(pad=2.0)
plt.savefig(
    './thesis_figures/'
    'learning_curves_efnet.png',
    dpi=300, bbox_inches='tight',
    facecolor='white')
plt.close()
print("Saved: learning_curves_efnet.png")