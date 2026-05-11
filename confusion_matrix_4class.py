import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# ── Confusion Matrix από per_class_metrics ──
# Σειρά: Glioma=0, Meningioma=1, No Tumor=2, Pituitary=3
# Glioma: 158 σωστά, 3→Meningioma, 1→Pituitary
# Meningioma: 161 σωστά, 3→No Tumor, 1→Pituitary
# No Tumor: 200 σωστά, 0 λάθη
# Pituitary: 175 σωστά, 1→Meningioma

CLASS_NAMES = ['Glioma', 'Meningioma',
               'No Tumor', 'Pituitary']

cm = np.array([
    [158,  3,  0,  1],  # Glioma
    [  0, 161, 3,  1],  # Meningioma
    [  0,  0, 200, 0],  # No Tumor
    [  0,  1,  0, 175]  # Pituitary
])

# ── Percentages ───────────────────────
cm_percent = cm.astype('float') / \
             cm.sum(axis=1)[:, np.newaxis] * 100

# ── Plot ──────────────────────────────
fig, ax = plt.subplots(figsize=(9, 7))
fig.patch.set_facecolor('white')

cmap = sns.color_palette("Blues", as_cmap=True)

sns.heatmap(
    cm,
    annot=False,
    cmap=cmap,
    xticklabels=CLASS_NAMES,
    yticklabels=CLASS_NAMES,
    linewidths=0.5,
    linecolor='#BDC3C7',
    ax=ax,
    cbar_kws={'label': 'Αριθμός Δειγμάτων'}
)

# ── Custom annotations ────────────────
for i in range(4):
    for j in range(4):
        count   = cm[i, j]
        percent = cm_percent[i, j]

        text_color = 'white' \
            if cm[i, j] > cm.max() * 0.5 \
            else '#2C3E50'

        ax.text(j + 0.5, i + 0.35,
                f'{count}',
                ha='center', va='center',
                fontsize=14,
                fontweight='bold',
                color=text_color)

        ax.text(j + 0.5, i + 0.65,
                f'({percent:.1f}%)',
                ha='center', va='center',
                fontsize=10,
                color=text_color)

# ── Diagonal highlight ────────────────
for i in range(4):
    ax.add_patch(plt.Rectangle(
        (i, i), 1, 1,
        fill=False,
        edgecolor='#27AE60',
        lw=2.5,
        zorder=3
    ))

# ── Labels ────────────────────────────
ax.set_xlabel('Προβλεπόμενη Κλάση',
              fontsize=12,
              fontweight='bold',
              labelpad=10)
ax.set_ylabel('Πραγματική Κλάση',
              fontsize=12,
              fontweight='bold',
              labelpad=10)
ax.set_title(
    'Confusion Matrix 4-Class CNN\n'
    'Test Set (n=703 δείγματα)',
    fontsize=13,
    fontweight='bold',
    color='#2C3E50',
    pad=15
)

ax.set_xticklabels(
    CLASS_NAMES,
    rotation=30,
    ha='right',
    fontsize=11
)
ax.set_yticklabels(
    CLASS_NAMES,
    rotation=0,
    fontsize=11
)

# ── Overall accuracy ──────────────────
total_correct = np.diag(cm).sum()
total         = cm.sum()
overall_acc   = total_correct / total * 100

fig.text(
    0.5, -0.02,
    f'Overall Accuracy: {overall_acc:.2f}% '
    f'| Πράσινο πλαίσιο = σωστές ταξινομήσεις',
    ha='center', fontsize=10,
    color='#7F8C8D', style='italic'
)

plt.tight_layout()
plt.savefig(
    './thesis_figures/confusion_matrix_4class.png',
    dpi=300,
    bbox_inches='tight',
    facecolor='white'
)
plt.close()
print(f"Saved! Overall Accuracy: {overall_acc:.2f}%")
print("\nPer-class accuracy:")
for i, cls in enumerate(CLASS_NAMES):
    print(f"  {cls}: {cm_percent[i,i]:.1f}%")