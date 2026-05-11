import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np

# ── Load data ─────────────────────────
df = pd.read_csv(
    './multiclass_4class_experiment_20251217_141321/'
    'multiclass_4class_training_log.csv'
)

# ── Find best epoch (Early Stopping) ──
best_epoch = df['val_accuracy'].idxmax()
print(f"Best epoch: {best_epoch + 1} "
      f"(val_acc={df['val_accuracy'][best_epoch]:.4f})")

# ── Figure setup ──────────────────────
fig, (ax1, ax2) = plt.subplots(
    1, 2, figsize=(14, 5))
fig.patch.set_facecolor('white')
fig.suptitle(
    'Learning Curves — 4-Class CNN',
    fontsize=14, fontweight='bold',
    color='#2C3E50', y=1.02
)

epochs = df['epoch'] + 1  # 1-indexed

# ════════════════════════════════════
# SUBPLOT 1 — Accuracy
# ════════════════════════════════════
ax1.plot(epochs, df['accuracy'],
         color='#2980B9', lw=2.5,
         label='Training Accuracy',
         marker='o', markersize=3)
ax1.plot(epochs, df['val_accuracy'],
         color='#E74C3C', lw=2.5,
         label='Validation Accuracy',
         marker='s', markersize=3,
         linestyle='--')

# Best epoch line
ax1.axvline(x=best_epoch + 1,
            color='#27AE60',
            linestyle=':',
            lw=2,
            label=f'Best Epoch ({best_epoch + 1})')

# Best val accuracy annotation
ax1.annotate(
    f'Best: {df["val_accuracy"][best_epoch]:.4f}',
    xy=(best_epoch + 1,
        df['val_accuracy'][best_epoch]),
    xytext=(best_epoch + 3,
            df['val_accuracy'][best_epoch] - 0.05),
    fontsize=9,
    color='#27AE60',
    arrowprops=dict(
        arrowstyle='->',
        color='#27AE60',
        lw=1.5
    )
)

ax1.set_title('Accuracy', fontsize=12,
              fontweight='bold', color='#2C3E50')
ax1.set_xlabel('Epoch', fontsize=11)
ax1.set_ylabel('Accuracy', fontsize=11)
ax1.set_ylim(0, 1.05)
ax1.legend(fontsize=10, loc='lower right')
ax1.grid(True, alpha=0.3,
         linestyle='--', color='gray')
ax1.set_facecolor('#FAFAFA')
ax1.spines['top'].set_visible(False)
ax1.spines['right'].set_visible(False)

# ════════════════════════════════════
# SUBPLOT 2 — Loss
# ════════════════════════════════════
ax2.plot(epochs, df['loss'],
         color='#2980B9', lw=2.5,
         label='Training Loss',
         marker='o', markersize=3)
ax2.plot(epochs, df['val_loss'],
         color='#E74C3C', lw=2.5,
         label='Validation Loss',
         marker='s', markersize=3,
         linestyle='--')

# Best epoch line
best_loss_epoch = df['val_loss'].idxmin()
ax2.axvline(x=best_epoch + 1,
            color='#27AE60',
            linestyle=':',
            lw=2,
            label=f'Best Epoch ({best_epoch + 1})')

# Best val loss annotation
# SUBPLOT 1 — αντικατέστησε το annotation
ax1.annotate(
    f'Best: {df["val_accuracy"][best_epoch]:.4f}',
    xy=(best_epoch + 1,
        df['val_accuracy'][best_epoch]),
    xytext=(best_epoch - 15, 0.05),
    fontsize=9,
    color='#27AE60',
    arrowprops=dict(
        arrowstyle='->',
        color='#27AE60',
        lw=1.5
    )
)

# SUBPLOT 2 — αντικατέστησε το annotation
ax2.annotate(
    f'Min Loss: {df["val_loss"][best_loss_epoch]:.4f}',
    xy=(best_loss_epoch + 1,
        df['val_loss'][best_loss_epoch]),
    xytext=(best_loss_epoch - 15, 0.3),
    fontsize=9,
    color='#27AE60',
    arrowprops=dict(
        arrowstyle='->',
        color='#27AE60',
        lw=1.5
    )
)

# ── Shared note ───────────────────────
fig.text(0.5, -0.03,
         'Η κατακόρυφη διακεκομμένη πράσινη '
         'γραμμή υποδηλώνει την εποχή με την '
         'υψηλότερη val_accuracy (Early Stopping).',
         ha='center', fontsize=9,
         color='#7F8C8D', style='italic')

plt.tight_layout(pad=2.0)
plt.savefig(
    './thesis_figures/learning_curves_4class.png',
    dpi=300,
    bbox_inches='tight',
    facecolor='white'
)
plt.close()
print("Saved: learning_curves_4class.png")