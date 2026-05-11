import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
from PIL import Image

# ── Paths ─────────────────────────────
GRADCAM = './xai_results/gradcam/gradcam_comparison_meningioma.png'
SHAP    = './xai_results/shap/shap_comparison_meningioma.png'
IG      = './xai_results/integrated_gradients/ig_comparison_meningioma.png'
OCC     = './xai_results/occlusion/occlusion_comparison_meningioma.png'

# ── Load images ───────────────────────
imgs = {
    'Grad-CAM':              mpimg.imread(GRADCAM),
    'SHAP':                  mpimg.imread(SHAP),
    'Integrated Gradients':  mpimg.imread(IG),
    'Occlusion Analysis':    mpimg.imread(OCC)
}

labels = ['(α) Grad-CAM',
          '(β) SHAP',
          '(γ) Integrated Gradients',
          '(δ) Occlusion Analysis']

methods = list(imgs.keys())

# ── Plot ──────────────────────────────
fig, axes = plt.subplots(
    4, 1,
    figsize=(16, 20)
)
fig.patch.set_facecolor('white')

fig.suptitle(
    'Σύγκριση 4 Μεθόδων XAI  Meningioma\n'
    'Αποτελέσματα και των 6 Μοντέλων',
    fontsize=14,
    fontweight='bold',
    color='#2C3E50',
    y=1.01
)

colors = ['#E74C3C', '#F39C12',
          '#8E44AD', '#27AE60']

for ax, (method, img), label, color \
        in zip(axes, imgs.items(),
               labels, colors):

    ax.imshow(img)
    ax.axis('off')

    # Label αριστερά
    ax.text(
        -0.01, 0.5,
        label,
        transform=ax.transAxes,
        fontsize=12,
        fontweight='bold',
        color=color,
        va='center',
        ha='right',
        rotation=0
    )

    # Colored border
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_color(color)
        spine.set_linewidth(3)

plt.tight_layout(pad=1.5)
plt.savefig(
    './thesis_figures/xai_comparison_meningioma.png',
    dpi=200,
    bbox_inches='tight',
    facecolor='white'
)
plt.close()
print("Saved: xai_comparison_meningioma.png")