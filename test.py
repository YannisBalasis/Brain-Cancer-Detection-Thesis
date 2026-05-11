import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

fig, ax = plt.subplots(1, 1, figsize=(14, 10))
fig.patch.set_facecolor('white')
ax.set_xlim(0, 14)
ax.set_ylim(0, 10)
ax.axis('off')

# ── Helper functions ──────────────────

def draw_box(ax, x, y, w, h, label,
             color='#3498DB', fontsize=9,
             text_color='white'):
    rect = mpatches.FancyBboxPatch(
        (x, y), w, h,
        boxstyle="round,pad=0.15",
        facecolor=color,
        edgecolor='#2C3E50',
        linewidth=1.8
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
               connectionstyle='arc3,rad=0'):
    ax.annotate('',
        xy=(x2, y2), xytext=(x1, y1),
        arrowprops=dict(
            arrowstyle='->',
            color=color, lw=lw,
            connectionstyle=connectionstyle
        )
    )

# ═══════════════════════════════════════
# 1. INPUT
# ═══════════════════════════════════════
draw_box(ax, 5.0, 8.8, 4.0, 0.85,
         'Input MRI Image (224×224)',
         color='#2C3E50', fontsize=10)

# Split arrows
draw_arrow(ax, 7.0, 8.8, 2.5, 7.9,
           color='#E74C3C', lw=2,
           connectionstyle='arc3,rad=0.1')
draw_arrow(ax, 7.0, 8.8, 7.0, 7.9,
           color='#3498DB', lw=2)
draw_arrow(ax, 7.0, 8.8, 11.5, 7.9,
           color='#F39C12', lw=2,
           connectionstyle='arc3,rad=-0.1')

# ═══════════════════════════════════════
# 2. THREE BINARY MODELS
# ═══════════════════════════════════════

# Model 1 — Glioma vs Meningioma
draw_box(ax, 0.5, 6.0, 4.0, 1.8,
         'Model 1\nGlioma vs Meningioma\n'
         'Binary CNN\n(1.276.449 params)',
         color='#E74C3C', fontsize=9)

# Model 2 — Glioma vs Pituitary
draw_box(ax, 5.0, 6.0, 4.0, 1.8,
         'Model 2\nGlioma vs Pituitary\n'
         'Binary CNN\n(1.276.449 params)',
         color='#3498DB', fontsize=9)

# Model 3 — Meningioma vs Pituitary
draw_box(ax, 9.5, 6.0, 4.0, 1.8,
         'Model 3\nMeningioma vs Pituitary\n'
         'Binary CNN\n(1.276.449 params)',
         color='#F39C12', fontsize=9)

# ═══════════════════════════════════════
# 3. VOTE BOXES
# ═══════════════════════════════════════

# Arrows down to votes
draw_arrow(ax, 2.5, 6.0, 2.5, 5.0,
           color='#E74C3C', lw=2)
draw_arrow(ax, 7.0, 6.0, 7.0, 5.0,
           color='#3498DB', lw=2)
draw_arrow(ax, 11.5, 6.0, 11.5, 5.0,
           color='#F39C12', lw=2)

# Vote 1
draw_box(ax, 1.0, 4.0, 3.0, 0.85,
         'Vote 1\nGlioma / Meningioma',
         color='#E74C3C', fontsize=8)

# Vote 2
draw_box(ax, 5.5, 4.0, 3.0, 0.85,
         'Vote 2\nGlioma / Pituitary',
         color='#3498DB', fontsize=8)

# Vote 3
draw_box(ax, 10.0, 4.0, 3.0, 0.85,
         'Vote 3\nMeningioma / Pituitary',
         color='#F39C12', fontsize=8)

# ═══════════════════════════════════════
# 4. MAJORITY VOTING
# ═══════════════════════════════════════

# Arrows to voting
draw_arrow(ax, 2.5, 4.0, 5.5, 3.1,
           color='#8E44AD', lw=2,
           connectionstyle='arc3,rad=0.15')
draw_arrow(ax, 7.0, 4.0, 7.0, 3.1,
           color='#8E44AD', lw=2)
draw_arrow(ax, 11.5, 4.0, 8.5, 3.1,
           color='#8E44AD', lw=2,
           connectionstyle='arc3,rad=-0.15')

draw_box(ax, 4.5, 2.2, 5.0, 0.85,
         'Majority Voting\n'
         '(κλάση με τις περισσότερες ψήφους)',
         color='#8E44AD', fontsize=9)

# ═══════════════════════════════════════
# 5. OUTPUT
# ═══════════════════════════════════════
draw_arrow(ax, 7.0, 2.2, 7.0, 1.4,
           color='#27AE60', lw=2)

draw_box(ax, 4.5, 0.4, 5.0, 0.9,
         'Final Prediction\n'
         'Glioma / Meningioma / Pituitary',
         color='#27AE60', fontsize=9)

# ═══════════════════════════════════════
# 6. LEGEND
# ═══════════════════════════════════════
legend_elements = [
    mpatches.Patch(facecolor='#E74C3C',
                   label='Model 1: Glioma vs Meningioma'),
    mpatches.Patch(facecolor='#3498DB',
                   label='Model 2: Glioma vs Pituitary'),
    mpatches.Patch(facecolor='#F39C12',
                   label='Model 3: Meningioma vs Pituitary'),
    mpatches.Patch(facecolor='#8E44AD',
                   label='Majority Voting'),
    mpatches.Patch(facecolor='#27AE60',
                   label='Final Output'),
]
ax.legend(handles=legend_elements,
          loc='lower left',
          fontsize=8,
          framealpha=0.9)

# ═══════════════════════════════════════
# 7. TITLE
# ═══════════════════════════════════════
ax.set_title(
    'Αρχιτεκτονική 1-vs-1 Ensemble',
    fontsize=14, fontweight='bold',
    color='#2C3E50', pad=15
)

plt.tight_layout()
plt.savefig(
    './thesis_figures/architecture_ensemble.png',
    dpi=300,
    bbox_inches='tight',
    facecolor='white'
)
plt.close()
print("Saved: architecture_ensemble.png")