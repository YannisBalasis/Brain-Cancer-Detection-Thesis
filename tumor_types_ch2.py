#!/usr/bin/env python3
"""
Chapter 2 - Brain Tumor Types Visualization
=============================================
Creates a clean 1x4 figure showing one representative
MRI image per tumor class with clinical annotations.

Author: Yannis Balasis
Date: March 2026
"""

import os
import random
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from pathlib import Path
from PIL import Image

# ── Configuration ──────────────────────────────────────────
DATA_PATH   = '/users/yannisbalasis/documents/thesis/data_multiclass'
OUTPUT_PATH = './thesis_figures'
SEED        = 42

random.seed(SEED)
np.random.seed(SEED)
os.makedirs(OUTPUT_PATH, exist_ok=True)

# ── Class definitions with clinical info ───────────────────
CLASSES = [
    {
        'folder':      'glioma',
        'title':       'Glioma\n(Γλοίωμα)',
        'color':       '#e74c3c',
        'description': 'Διηθητικός όγκος\nΑσαφή όρια\nΠεριεστιακό οίδημα',
        'label':       '(α)'
    },
    {
        'folder':      'meningioma',
        'title':       'Meningioma\n(Μηνιγγίωμα)',
        'color':       '#3498db',
        'description': 'Εξω-αξονικός όγκος\nΣαφή όρια\nΈντονη ενίσχυση',
        'label':       '(β)'
    },
    {
        'folder':      'pituitary',
        'title':       'Pituitary Tumor\n(Όγκος Υπόφυσης)',
        'color':       '#2ecc71',
        'description': 'Αδένωμα υπόφυσης\nΒάση κρανίου\nΟρμονικές επιπτώσεις',
        'label':       '(γ)'
    },
    {
        'folder':      'no_tumor',
        'title':       'No Tumor\n(Υγιής Εγκέφαλος)',
        'color':       '#95a5a6',
        'description': 'Φυσιολογική ανατομία\nΑπουσία παθολογίας\nΟμάδα ελέγχου',
        'label':       '(δ)'
    }
]


def plot_tumor_types(data_path: str, output_path: str):
    """
    Creates a clean 1x4 figure with one representative
    MRI image per class - no titles or descriptions.
    """
    fig = plt.figure(figsize=(16, 5))
    fig.patch.set_facecolor('black')

    for idx, cls in enumerate(CLASSES):
        # Load random image from class
        cls_path = Path(data_path) / cls['folder']
        images   = list(cls_path.glob('*.jpg')) + \
                   list(cls_path.glob('*.png')) + \
                   list(cls_path.glob('*.jpeg'))
        img_path = random.choice(images)
        img      = Image.open(img_path).convert('RGB').resize((224, 224))
        img_arr  = np.array(img)

        # Create subplot
        ax = fig.add_subplot(1, 4, idx + 1)
        ax.imshow(img_arr, cmap='gray')
        ax.axis('off')

        # Μόνο το γράμμα (α), (β), (γ), (δ) κάτω αριστερά
        ax.text(
            0.05, 0.05,
            cls['label'],
            transform=ax.transAxes,
            ha='left', va='bottom',
            fontsize=13,
            fontweight='bold',
            color='white'
        )

        # Colored border
        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_color(cls['color'])
            spine.set_linewidth(3)

    plt.tight_layout(pad=0.5)

    save_path = os.path.join(output_path, 'tumor_types_ch2.png')
    plt.savefig(
        save_path, dpi=300,
        bbox_inches='tight',
        facecolor='black'
    )
    plt.close()
    print(f'Saved: {save_path}')


if __name__ == '__main__':
    print('Generating Chapter 2 tumor types figure...')
    plot_tumor_types(DATA_PATH, OUTPUT_PATH)
    print('Done!')