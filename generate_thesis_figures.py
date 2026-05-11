#!/usr/bin/env python3
"""
Dataset Visualization Script
=============================
Generates sample images from each class for the thesis.

Author: Yannis Balasis
Date: March 2026
"""

import os
import random
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from pathlib import Path
from PIL import Image

# ── Configuration ──────────────────────────────────────────
DATA_PATH    = '/users/yannisbalasis/documents/thesis/data_multiclass'
OUTPUT_PATH  = './thesis_figures'
SEED         = 42

CLASS_NAMES  = ['glioma', 'meningioma', 'no_tumor', 'pituitary']
CLASS_LABELS = {
    'glioma':      'Glioma (Γλοίωμα)',
    'meningioma':  'Meningioma (Μηνιγγίωμα)',
    'no_tumor':    'No Tumor (Υγιής)',
    'pituitary':   'Pituitary Tumor (Όγκος Υπόφυσης)'
}

random.seed(SEED)
np.random.seed(SEED)
os.makedirs(OUTPUT_PATH, exist_ok=True)


# ══════════════════════════════════════════════════════════
# 1. FIGURE 1: Αντιπροσωπευτικά δείγματα (4 κλάσεις x 4 εικόνες)
# ══════════════════════════════════════════════════════════

def plot_dataset_samples(data_path: str, output_path: str,
                          n_samples: int = 4):
    """
    Creates a grid of n_samples images per class.
    Saves as dataset_samples.png
    """
    fig, axes = plt.subplots(
        len(CLASS_NAMES), n_samples,
        figsize=(n_samples * 3, len(CLASS_NAMES) * 3)
    )
    fig.suptitle(
        'Αντιπροσωπευτικά Δείγματα ανά Κλάση\n'
        'Brain Tumor MRI Dataset',
        fontsize=14, fontweight='bold', y=1.01
    )

    for row, cls in enumerate(CLASS_NAMES):
        cls_path = Path(data_path) / cls
        images   = list(cls_path.glob('*.jpg')) + \
                   list(cls_path.glob('*.png')) + \
                   list(cls_path.glob('*.jpeg'))

        selected = random.sample(images, min(n_samples, len(images)))

        for col, img_path in enumerate(selected):
            img = Image.open(img_path).convert('RGB')
            img = img.resize((224, 224))

            axes[row, col].imshow(img, cmap='gray')
            axes[row, col].axis('off')

            # Label only on first column
            if col == 0:
                axes[row, col].set_ylabel(
                    CLASS_LABELS[cls],
                    fontsize=10, fontweight='bold',
                    rotation=90, labelpad=10
                )

            # Letter label (α, β, γ, δ) on first row
            if row == 0:
                letter = ['(α)', '(β)', '(γ)', '(δ)'][col]
                axes[row, col].set_title(
                    letter, fontsize=10, pad=4
                )

    plt.tight_layout()
    save_path = os.path.join(output_path, 'dataset_samples.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f'Saved: {save_path}')


# ══════════════════════════════════════════════════════════
# 2. FIGURE 2: Κατανομή κλάσεων (bar chart)
# ══════════════════════════════════════════════════════════

def plot_class_distribution(data_path: str, output_path: str):
    """
    Creates a bar chart showing class distribution.
    Saves as class_distribution.png
    """
    counts = {}
    for cls in CLASS_NAMES:
        cls_path = Path(data_path) / cls
        images   = list(cls_path.glob('*.jpg')) + \
                   list(cls_path.glob('*.png')) + \
                   list(cls_path.glob('*.jpeg'))
        counts[cls] = len(images)

    labels = [CLASS_LABELS[c] for c in CLASS_NAMES]
    values = [counts[c] for c in CLASS_NAMES]
    colors = ['#e74c3c', '#3498db', '#2ecc71', '#f39c12']

    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.bar(labels, values, color=colors,
                  edgecolor='grey', linewidth=0.8)

    # Value labels on bars
    for bar, val in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 20,
            str(val), ha='center', va='bottom',
            fontsize=11, fontweight='bold'
        )

    # Split lines for train/val/test
    for bar, val in zip(bars, values):
        train = int(val * 0.70)
        val_n = int(val * 0.20)
        x     = bar.get_x()
        w     = bar.get_width()
        ax.plot([x, x + w], [train, train],
                color='white', linewidth=1.5,
                linestyle='--', alpha=0.8)
        ax.plot([x, x + w], [train + val_n, train + val_n],
                color='white', linewidth=1.5,
                linestyle=':', alpha=0.8)

    ax.set_title(
        'Κατανομή Εικόνων ανά Κλάση\n'
        '(διακεκομμένες γραμμές: όρια Train/Val/Test split)',
        fontsize=13, fontweight='bold'
    )
    ax.set_ylabel('Αριθμός Εικόνων', fontsize=12)
    ax.set_xlabel('Κλάση', fontsize=12)
    ax.set_ylim(0, max(values) * 1.15)
    ax.tick_params(axis='x', labelsize=10)
    ax.grid(axis='y', alpha=0.3)

    total = sum(values)
    ax.text(
        0.98, 0.97, f'Σύνολο: {total} εικόνες',
        transform=ax.transAxes,
        ha='right', va='top', fontsize=11,
        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    )

    plt.tight_layout()
    save_path = os.path.join(output_path, 'class_distribution.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f'Saved: {save_path}')


# ══════════════════════════════════════════════════════════
# 3. FIGURE 3: Data Augmentation παραδείγματα
# ══════════════════════════════════════════════════════════

def plot_augmentation_examples(data_path: str, output_path: str):
    """
    Shows original image + 5 augmented versions side by side.
    Saves as augmentation_examples.png
    """
    from tensorflow.keras.preprocessing.image import ImageDataGenerator

    # Pick one random glioma image
    cls_path = Path(data_path) / 'glioma'
    images   = list(cls_path.glob('*.jpg')) + \
               list(cls_path.glob('*.png'))
    img_path = random.choice(images)

    # Φόρτωση χωρίς normalization — το datagen θα το κάνει
    img       = Image.open(img_path).convert('RGB').resize((224, 224))
    img_array = np.array(img, dtype=np.float32)  # τιμές [0, 255]
    img_batch = np.expand_dims(img_array, axis=0)

    datagen = ImageDataGenerator(
        rotation_range=20,
        width_shift_range=0.15,
        height_shift_range=0.15,
        horizontal_flip=True,
        zoom_range=0.1,
        brightness_range=[0.8, 1.2],
        fill_mode='nearest'
    )

    aug_titles = [
        'Πρωτότυπη',
        'Περιστροφή',
        'Μετατόπιση',
        'Αναστροφή',
        'Zoom',
        'Λαμπρότητα'
    ]

    fig, axes = plt.subplots(1, 6, figsize=(18, 3))
    fig.suptitle(
        'Παραδείγματα Data Augmentation σε εικόνα MRI (Glioma)',
        fontsize=13, fontweight='bold'
    )

    # Original — normalize για εμφάνιση
    axes[0].imshow(img_array.astype(np.uint8))
    axes[0].set_title(aug_titles[0], fontsize=10, fontweight='bold')
    axes[0].axis('off')

    # Augmented versions
    aug_gen = datagen.flow(img_batch, batch_size=1, seed=SEED)
    for i in range(1, 6):
        aug_img = next(aug_gen)[0]  # shape (224, 224, 3), range [0,255]
        # Clip και convert σε uint8 για σωστή εμφάνιση
        axes[i].imshow(np.clip(aug_img, 0, 255).astype(np.uint8))
        axes[i].set_title(aug_titles[i], fontsize=10)
        axes[i].axis('off')

    plt.tight_layout()
    save_path = os.path.join(output_path, 'augmentation_examples.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f'Saved: {save_path}')


# ══════════════════════════════════════════════════════════
# 4. FIGURE 4: Pie chart κατανομής Train/Val/Test
# ══════════════════════════════════════════════════════════

def plot_split_distribution(output_path: str):
    """
    Creates a pie chart showing train/val/test split.
    Saves as split_distribution.png
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle(
        'Κατανομή Συνόλου Δεδομένων',
        fontsize=13, fontweight='bold'
    )

    # Pie chart — overall split
    splits  = [70, 20, 10]
    labels  = ['Training Set\n(70%)', 
               'Validation Set\n(20%)', 
               'Test Set\n(10%)']
    colors  = ['#3498db', '#f39c12', '#e74c3c']
    explode = [0.05, 0.05, 0.05]

    axes[0].pie(
        splits, labels=labels, colors=colors,
        explode=explode, autopct='%1.0f%%',
        startangle=90, textprops={'fontsize': 11}
    )
    axes[0].set_title('Train / Val / Test Split', fontweight='bold')

    # Bar chart — per class per split
    class_counts = {
        'glioma':     1621,
        'meningioma': 1645,
        'pituitary':  1757,
        'no_tumor':   2000
    }

    x      = np.arange(len(CLASS_NAMES))
    width  = 0.25
    train  = [int(v * 0.70) for v in class_counts.values()]
    val    = [int(v * 0.20) for v in class_counts.values()]
    test   = [int(v * 0.10) for v in class_counts.values()]

    axes[1].bar(x - width, train, width,
                label='Train', color='#3498db')
    axes[1].bar(x,          val,   width,
                label='Val',   color='#f39c12')
    axes[1].bar(x + width,  test,  width,
                label='Test',  color='#e74c3c')

    axes[1].set_xticks(x)
    axes[1].set_xticklabels(
        [l.split(' ')[0] for l in CLASS_NAMES],
        fontsize=10
    )
    axes[1].set_ylabel('Αριθμός Εικόνων', fontsize=11)
    axes[1].set_title(
        'Κατανομή ανά Κλάση και Split',
        fontweight='bold'
    )
    axes[1].legend(fontsize=10)
    axes[1].grid(axis='y', alpha=0.3)

    plt.tight_layout()
    save_path = os.path.join(output_path, 'split_distribution.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f'Saved: {save_path}')


# ══════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════

if __name__ == '__main__':
    print('Generating thesis figures for Chapter 4...')
    print(f'Data path:   {DATA_PATH}')
    print(f'Output path: {OUTPUT_PATH}')
    print()

    print('[1/4] Dataset samples grid...')
    plot_dataset_samples(DATA_PATH, OUTPUT_PATH, n_samples=4)

    print('[2/4] Class distribution chart...')
    plot_class_distribution(DATA_PATH, OUTPUT_PATH)

    print('[3/4] Augmentation examples...')
    plot_augmentation_examples(DATA_PATH, OUTPUT_PATH)

    print('[4/4] Split distribution...')
    plot_split_distribution(OUTPUT_PATH)

    print()
    print('All figures saved to:', OUTPUT_PATH)
    print('Files generated:')
    print('  - dataset_samples.png      → Figure για δείγματα dataset')
    print('  - class_distribution.png   → Figure για κατανομή κλάσεων')
    print('  - augmentation_examples.png → Figure για Data Augmentation')
    print('  - split_distribution.png   → Figure για Train/Val/Test split')