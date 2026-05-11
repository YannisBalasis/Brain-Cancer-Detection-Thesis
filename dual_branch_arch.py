import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.patheffects as pe
import numpy as np

fig, ax = plt.subplots(1, 1, figsize=(14, 10))
fig.patch.set_facecolor('white')
ax.set_xlim(0, 14)
ax.set_ylim(0, 10)
ax.axis('off')

# ── Helper functions ──────────────────

def draw_box(ax, x, y, w, h, label,
             color='#3498DB', fontsize=10,
             text_color='white', alpha=1.0,
             multiline=False):
    rect = mpatches.FancyBboxPatch(
        (x, y), w, h,
        boxstyle="round,pad=0.15",
        facecolor=color,
        edgecolor='#2C3E50',
        linewidth=1.8,
        alpha=alpha
    )
    ax.add_patch(rect)
    ax.text(x + w/2, y + h/2, label,
            ha='center', va='center',
            fontsize=fontsize,
            fontweight='bold',
            color=text_color,
            linespacing=1.4)

def draw_arrow(ax, x1, y1, x2, y2,
               color='#2C3E50', lw=2,
               style='->',
               connectionstyle='arc3,rad=0'):
    ax.annotate('',
        xy=(x2, y2),
        xytext=(x1, y1),
        arrowprops=dict(
            arrowstyle=style,
            color=color,
            lw=lw,
            connectionstyle=connectionstyle
        )
    )

def draw_label(ax, x, y, text,
               fontsize=9, color='#7F8C8D',
               style='italic'):
    ax.text(x, y, text,
            ha='center', va='center',
            fontsize=fontsize,
            color=color,
            style=style)

# ═══════════════════════════════════════
# 1. INPUT IMAGE
# ═══════════════════════════════════════
draw_box(ax, 5.5, 8.8, 3.0, 0.85,
         'Input MRI Image (224×224)',
         color='#2C3E50', fontsize=10)

# Arrow down from input
draw_arrow(ax, 7.0, 8.8, 7.0, 8.2)

# ═══════════════════════════════════════
# 2. SPLIT POINT
# ═══════════════════════════════════════
ax.plot(7.0, 8.0, 'o',
        color='#2C3E50',
        markersize=10, zorder=5)
ax.text(7.5, 8.05, 'Split',
        fontsize=9, color='#7F8C8D',
        style='italic')

# Arrow left → Branch 1
draw_arrow(ax, 7.0, 8.0, 3.5, 8.0,
           color='#E74C3C', lw=2,
           connectionstyle='arc3,rad=0')

# Arrow right → Branch 2
draw_arrow(ax, 7.0, 8.0, 10.5, 8.0,
           color='#3498DB', lw=2,
           connectionstyle='arc3,rad=0')

# ═══════════════════════════════════════
# 3. BRANCH 1 — Custom 4-Class CNN
# ═══════════════════════════════════════

# Branch label
ax.text(2.5, 8.35, 'Branch 1',
        ha='center', fontsize=11,
        fontweight='bold', color='#E74C3C')

# Arrow down
draw_arrow(ax, 3.5, 8.0, 3.5, 7.3,
           color='#E74C3C', lw=2)

# Conv Block 1
draw_box(ax, 1.8, 6.4, 3.4, 0.8,
         'Conv Block 1\n(64 filters)',
         color='#E74C3C', fontsize=9)

draw_arrow(ax, 3.5, 6.4, 3.5, 5.7,
           color='#E74C3C', lw=1.8)

# Conv Block 2
draw_box(ax, 1.8, 4.8, 3.4, 0.8,
         'Conv Block 2\n(128 filters)',
         color='#C0392B', fontsize=9)

draw_arrow(ax, 3.5, 4.8, 3.5, 4.1,
           color='#E74C3C', lw=1.8)

# Conv Block 3
draw_box(ax, 1.8, 3.2, 3.4, 0.8,
         'Conv Block 3\n(256 filters)',
         color='#A93226', fontsize=9)

draw_arrow(ax, 3.5, 3.2, 3.5, 2.5,
           color='#E74C3C', lw=1.8)

# Dense Layer Branch 1
draw_box(ax, 1.8, 1.6, 3.4, 0.8,
         'Dense Layer\n(512 units)',
         color='#922B21', fontsize=9)

# Branch label bottom
ax.text(3.5, 1.3,
        'Custom 4-Class CNN',
        ha='center', fontsize=9,
        color='#E74C3C', style='italic')

# ═══════════════════════════════════════
# 4. BRANCH 2 — ResNet-50 / EfficientNet
# ═══════════════════════════════════════

# Branch label
ax.text(11.5, 8.35, 'Branch 2',
        ha='center', fontsize=11,
        fontweight='bold', color='#3498DB')

# Arrow down
draw_arrow(ax, 10.5, 8.0, 10.5, 7.3,
           color='#3498DB', lw=2)

# Pretrained Backbone
draw_box(ax, 8.8, 6.4, 3.4, 0.8,
         'Pretrained Backbone\n(ImageNet weights)',
         color='#3498DB', fontsize=9)

draw_arrow(ax, 10.5, 6.4, 10.5, 5.7,
           color='#3498DB', lw=1.8)

# ResNet / EfficientNet block
draw_box(ax, 8.8, 4.8, 3.4, 0.8,
         'ResNet-50 /\nEfficientNet-B3',
         color='#2980B9', fontsize=9)

draw_arrow(ax, 10.5, 4.8, 10.5, 4.1,
           color='#3498DB', lw=1.8)

# Global Average Pooling
draw_box(ax, 8.8, 3.2, 3.4, 0.8,
         'Global Average\nPooling',
         color='#1F618D', fontsize=9)

draw_arrow(ax, 10.5, 3.2, 10.5, 2.5,
           color='#3498DB', lw=1.8)

# Dense Layer Branch 2
draw_box(ax, 8.8, 1.6, 3.4, 0.8,
         'Dense Layer\n(512 units)',
         color='#154360', fontsize=9)

# Branch label bottom
ax.text(10.5, 1.3,
        'ResNet-50 / EfficientNet-B3',
        ha='center', fontsize=9,
        color='#3498DB', style='italic')

# ═══════════════════════════════════════
# 5. CONCATENATION LAYER
# ═══════════════════════════════════════

# Arrows from both branches to concat
draw_arrow(ax, 3.5, 1.6, 5.8, 0.85,
           color='#8E44AD', lw=2,
           connectionstyle='arc3,rad=0.2')

draw_arrow(ax, 10.5, 1.6, 8.2, 0.85,
           color='#8E44AD', lw=2,
           connectionstyle='arc3,rad=-0.2')

# Concat box
draw_box(ax, 5.0, 0.3, 4.0, 0.9,
         'Concatenation Layer\n(1024 units)',
         color='#8E44AD', fontsize=10)

# ═══════════════════════════════════════
# 6. DENSE + SOFTMAX → OUTPUT
# ═══════════════════════════════════════

# Arrow down from concat
draw_arrow(ax, 7.0, 0.3, 7.0, -0.5,
           color='#27AE60', lw=2)

# Dense + Softmax
draw_box(ax, 5.0, -1.4, 4.0, 0.85,
         'Dense + Softmax\n(4 units)',
         color='#27AE60', fontsize=10)

# Arrow to output
draw_arrow(ax, 7.0, -1.4, 7.0, -2.1,
           color='#27AE60', lw=2)

# Output box
draw_box(ax, 4.2, -3.0, 5.6, 0.85,
         'Output (4 Classes)',
         color='#1E8449', fontsize=10)

# Output class labels
classes = ['Glioma', 'Meningioma',
           'No Tumor', 'Pituitary']
colors_out = ['#E74C3C', '#3498DB',
              '#2ECC71', '#F39C12']
xs_out = [4.5, 5.8, 7.2, 8.5]

for cls, col, x in zip(classes,
                        colors_out,
                        xs_out):
    draw_arrow(ax, 7.0, -3.0,
               x, -3.6,
               color=col, lw=1.5)
    draw_box(ax, x - 0.6, -4.4,
             1.2, 0.7, cls,
             color=col, fontsize=8)

# ═══════════════════════════════════════
# 7. FROZEN WEIGHTS annotation
# ═══════════════════════════════════════
ax.annotate('Frozen/Fine-tuned\nweights',
    xy=(10.5, 6.4),
    xytext=(12.5, 6.8),
    fontsize=8,
    color='#3498DB',
    style='italic',
    arrowprops=dict(
        arrowstyle='->',
        color='#3498DB',
        lw=1
    )
)

# ═══════════════════════════════════════
# 8. TITLE
# ═══════════════════════════════════════
ax.set_title(
    'Αρχιτεκτονική Dual Branch System',
    fontsize=14, fontweight='bold',
    color='#2C3E50', pad=15
)

# ═══════════════════════════════════════
# 9. LEGEND
# ═══════════════════════════════════════
legend_elements = [
    mpatches.Patch(facecolor='#E74C3C',
                   label='Branch 1: Custom CNN'),
    mpatches.Patch(facecolor='#3498DB',
                   label='Branch 2: Pretrained'),
    mpatches.Patch(facecolor='#8E44AD',
                   label='Concatenation'),
    mpatches.Patch(facecolor='#27AE60',
                   label='Output Layer'),
]
ax.legend(handles=legend_elements,
          loc='lower left',
          fontsize=9,
          framealpha=0.9)

plt.tight_layout()
ax.set_ylim(-5, 10)

plt.savefig(
    './thesis_figures/dual_branch_architecture.png',
    dpi=300,
    bbox_inches='tight',
    facecolor='white'
)
plt.close()
print("Saved: dual_branch_architecture.png")