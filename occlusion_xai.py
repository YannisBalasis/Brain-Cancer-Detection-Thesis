#!/usr/bin/env python3
"""
Occlusion Analysis for Brain Tumor Classification Models
=========================================================
Systematically occludes regions of the input image and measures
the drop in prediction confidence to identify critical regions.

Formula:
    O(i,j) = f(x) - f(x ⊕ M_ij)

Where M_ij is a sliding window mask centered at (i,j).

Author: Yannis Balasis
Date: March 2026
"""

import os
import sys
import random
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
from PIL import Image
import tensorflow as tf

# ── Configuration ──────────────────────────────────────────
DATA_PATH   = '/users/yannisbalasis/documents/thesis/data_multiclass'
OUTPUT_PATH = './xai_results/occlusion'
SEED        = 42

# Occlusion window size (pixels) — larger = faster but less precise
WINDOW_SIZE = 32
# Stride — how many pixels to move the window each step
STRIDE      = 8
# Occlusion value — what to replace the occluded region with
OCCLUDE_VAL = 0.0  # black (baseline)

random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

os.makedirs(OUTPUT_PATH, exist_ok=True)

CLASS_NAMES_4 = ['glioma', 'meningioma', 'no_tumor', 'pituitary']
CLASS_NAMES_3 = ['glioma', 'meningioma', 'pituitary']
CLASS_LABELS  = {
    'glioma':      'Glioma',
    'meningioma':  'Meningioma',
    'no_tumor':    'No Tumor',
    'pituitary':   'Pituitary'
}

MODEL_CONFIG = {
    'Binary_CNN': {
        'path':    './best_binary_model.h5',
        'classes': CLASS_NAMES_4,
        'type':    'binary'
    },
    '4Class_CNN': {
        'path':    './multiclass_4class_experiment_20251217_141321/best_multiclass_4class_model.h5',
        'classes': CLASS_NAMES_4,
        'type':    '4class'
    },
    '3Class_CNN': {
        'path':    './multiclass_3class_experiment_20260201_161822/best_multiclass_3class_model.h5',
        'classes': CLASS_NAMES_3,
        'type':    '3class'
    },
    'ResNet_Dual': {
        'path':    './dual_system_experiment_20260227_114800/models/best_dual_system_model.h5',
        'classes': CLASS_NAMES_4,
        'type':    'dual'
    },
    'EfficientNet_Dual': {
        'path':    './efnet_dual_system_experiment_20260305_194201/models/efnet_dual_phase3.h5',
        'classes': CLASS_NAMES_4,
        'type':    'dual'
    },
    'Multi_Dual': {
        'path':    './multi_dual_system_experiment_20260310_100816/best_multi_dual_model.h5',
        'classes': CLASS_NAMES_4,
        'type':    'multi_dual',
        'custom_objects': True
    },
}


# ══════════════════════════════════════════════════════════
# MODEL LOADING
# ══════════════════════════════════════════════════════════

def load_model(model_name: str, config: dict):
    """Load model with appropriate handling."""
    if config.get('custom_objects'):
        sys.path.append('.')
        try:
            from multi_dual_system_architecture import \
                MultiDualSystemArchitecture
            arch = MultiDualSystemArchitecture()
            custom_objects = {
                '_masked_sparse_crossentropy':
                    arch._masked_sparse_crossentropy,
                '_masked_sparse_accuracy':
                    arch._masked_sparse_accuracy
            }
            model = tf.keras.models.load_model(
                config['path'],
                custom_objects=custom_objects,
                compile=False
            )
        except Exception as e:
            print(f"   Warning: {e}")
            model = tf.keras.models.load_model(
                config['path'], compile=False)
    else:
        model = tf.keras.models.load_model(
            config['path'], compile=False)

    # Build Sequential models
    if isinstance(model, tf.keras.Sequential):
        dummy = tf.zeros((1, 224, 224, 3))
        _ = model(dummy)
        print(f"   Built Sequential model")

    print(f"   Loaded: {config['path']}")
    return model


def predict(model, img_array: np.ndarray) -> np.ndarray:
    """
    Run prediction and return probability array.
    Always returns 1D array of probabilities.
    """
    img_batch = tf.cast(img_array[np.newaxis], tf.float32)
    preds     = model(img_batch, training=False)

    if isinstance(preds, dict):
        preds = list(preds.values())[0]
    elif isinstance(preds, (list, tuple)):
        preds = preds[0]

    preds = preds.numpy()[0]

    # Binary sigmoid → convert to [no_tumor_prob, tumor_prob]
    if preds.shape == () or len(preds) == 1:
        p = float(np.squeeze(preds))
        preds = np.array([1 - p, p])

    return preds


# ══════════════════════════════════════════════════════════
# IMAGE LOADING
# ══════════════════════════════════════════════════════════

def load_image(img_path: str, size: int = 224) -> np.ndarray:
    """Load and preprocess image."""
    img = Image.open(img_path).convert('RGB').resize((size, size))
    return np.array(img, dtype=np.float32) / 255.0


def get_sample_images(data_path: str, classes: list,
                      seed: int = 42) -> dict:
    """Get one random image per class."""
    random.seed(seed)
    samples = {}
    for cls in classes:
        cls_path = Path(data_path) / cls
        images   = list(cls_path.glob('*.jpg')) + \
                   list(cls_path.glob('*.png')) + \
                   list(cls_path.glob('*.jpeg'))
        samples[cls] = str(random.choice(images))
    return samples


# ══════════════════════════════════════════════════════════
# OCCLUSION ANALYSIS
# ══════════════════════════════════════════════════════════

def compute_occlusion_map(model, img_array: np.ndarray,
                           class_idx: int,
                           window_size: int = 32,
                           stride: int = 8,
                           occlude_val: float = 0.0
                           ) -> np.ndarray:
    """
    Compute occlusion sensitivity map.

    Slides a window across the image, replaces each region
    with occlude_val, and measures the drop in confidence
    for the target class.

    Args:
        model:       Keras model
        img_array:   Input image (H, W, 3) normalized [0,1]
        class_idx:   Target class index
        window_size: Size of occlusion window in pixels
        stride:      Stride of sliding window
        occlude_val: Value to fill occluded region

    Returns:
        occlusion_map: (H, W) sensitivity map
                       Higher = more important region
    """
    H, W, C     = img_array.shape
    occ_map     = np.zeros((H, W), dtype=np.float32)
    count_map   = np.zeros((H, W), dtype=np.float32)

    # Original prediction confidence for target class
    orig_preds  = predict(model, img_array)
    safe_idx    = min(class_idx, len(orig_preds) - 1)
    orig_conf   = orig_preds[safe_idx]

    half = window_size // 2

    # Slide window
    for y in range(0, H, stride):
        for x in range(0, W, stride):
            # Define window boundaries
            y1 = max(0, y - half)
            y2 = min(H, y + half)
            x1 = max(0, x - half)
            x2 = min(W, x + half)

            # Create occluded image
            occluded        = img_array.copy()
            occluded[y1:y2, x1:x2, :] = occlude_val

            # Predict on occluded image
            occ_preds = predict(model, occluded)
            occ_conf  = occ_preds[safe_idx]

            # Sensitivity = drop in confidence
            sensitivity = orig_conf - occ_conf

            # Accumulate (a pixel may be covered by multiple windows)
            occ_map[y1:y2, x1:x2]   += sensitivity
            count_map[y1:y2, x1:x2] += 1

    # Average over overlapping windows
    count_map = np.maximum(count_map, 1)
    occ_map   = occ_map / count_map

    return occ_map


def normalize_map(occ_map: np.ndarray) -> np.ndarray:
    """Normalize occlusion map to [0, 1]."""
    min_val = occ_map.min()
    max_val = occ_map.max()
    if max_val - min_val < 1e-8:
        return np.zeros_like(occ_map)
    return (occ_map - min_val) / (max_val - min_val)


# ══════════════════════════════════════════════════════════
# VISUALIZATION
# ══════════════════════════════════════════════════════════

def plot_occlusion_for_model(model_name: str, config: dict,
                              samples: dict, output_dir: str):
    """
    Creates Occlusion visualization for one model.
    Layout per class:
        [Original | Occlusion Map | Overlay | Confidence Bar]
    """
    classes   = config['classes']
    n_classes = len(classes)

    fig, axes = plt.subplots(
        n_classes, 4,
        figsize=(16, n_classes * 3.5)
    )
    if n_classes == 1:
        axes = axes[np.newaxis, :]

    fig.suptitle(
        f'Occlusion Analysis — {model_name.replace("_", " ")}',
        fontsize=14, fontweight='bold', y=1.01
    )

    col_titles = [
        'Πρωτότυπη Εικόνα',
        'Occlusion Map\n(υψηλό = κρίσιμη περιοχή)',
        'Επικάλυψη',
        'Confidence\nDrop (%)'
    ]
    for col, title in enumerate(col_titles):
        axes[0, col].set_title(
            title, fontsize=9, fontweight='bold', pad=8)

    model = load_model(model_name, config)

    for row, cls in enumerate(classes):
        img_path  = samples[cls]
        img_array = load_image(img_path)

        # Determine class index
        if config['type'] == '3class':
            map_4_to_3 = {'glioma': 0, 'meningioma': 1,
                           'pituitary': 2}
            class_idx  = map_4_to_3.get(cls, 0)
        elif config['type'] == 'binary':
            class_idx = 1  # tumor present
        else:
            class_idx = CLASS_NAMES_4.index(cls) \
                if cls in CLASS_NAMES_4 else 0

        print(f"   Computing occlusion for {cls} "
              f"(class_idx={class_idx}, "
              f"window={WINDOW_SIZE}, stride={STRIDE})...")

        try:
            occ_map = compute_occlusion_map(
                model, img_array,
                class_idx=class_idx,
                window_size=WINDOW_SIZE,
                stride=STRIDE,
                occlude_val=OCCLUDE_VAL
            )
        except Exception as e:
            print(f"   Occlusion failed for {cls}: {e}")
            occ_map = np.zeros((224, 224))

        occ_norm = normalize_map(occ_map)

        # Row label
        axes[row, 0].text(
            -0.15, 0.5,
            CLASS_LABELS.get(cls, cls),
            transform=axes[row, 0].transAxes,
            fontsize=11, fontweight='bold',
            ha='center', va='center',
            rotation=90
        )

        # Col 1: Original image
        axes[row, 0].imshow(img_array)
        axes[row, 0].axis('off')

        # Col 2: Occlusion map
        axes[row, 1].imshow(occ_norm, cmap='RdYlGn_r',
                             vmin=0, vmax=1)
        axes[row, 1].axis('off')

        # Col 3: Overlay on original
        colormap = matplotlib.colormaps.get_cmap('RdYlGn_r')
        heat_rgb = colormap(occ_norm)[:, :, :3]
        overlay  = np.clip(
            img_array * 0.5 + heat_rgb * 0.5, 0, 1)
        axes[row, 2].imshow(overlay)
        axes[row, 2].axis('off')

        # Col 4: Confidence drop bar chart
        # Show top-5 most sensitive regions' average drop
        orig_conf = predict(model, img_array)
        safe_idx  = min(class_idx, len(orig_conf) - 1)

        # Compute drops for a few key regions
        regions = {
            'Top-left':     (0,   0,   112, 112),
            'Top-right':    (0,   112, 112, 224),
            'Center':       (56,  56,  168, 168),
            'Bottom-left':  (112, 0,   224, 112),
            'Bottom-right': (112, 112, 224, 224),
        }

        drops = {}
        for region_name, (y1, x1, y2, x2) in regions.items():
            occluded = img_array.copy()
            occluded[y1:y2, x1:x2, :] = OCCLUDE_VAL
            occ_conf = predict(model, occluded)
            drop = (orig_conf[safe_idx] -
                    occ_conf[safe_idx]) * 100
            drops[region_name] = drop

        region_names = list(drops.keys())
        drop_values  = list(drops.values())
        colors       = ['#e74c3c' if d > 0 else '#2ecc71'
                        for d in drop_values]

        axes[row, 3].barh(region_names, drop_values,
                           color=colors, edgecolor='grey',
                           linewidth=0.5)
        axes[row, 3].axvline(x=0, color='black',
                              linewidth=0.8)
        axes[row, 3].set_xlabel('Confidence Drop (%)',
                                 fontsize=8)
        axes[row, 3].tick_params(labelsize=7)
        axes[row, 3].set_title(
            f'Orig: {orig_conf[safe_idx]*100:.1f}%',
            fontsize=8)

    # Colorbar
    fig.subplots_adjust(right=0.88)
    cbar_ax = fig.add_axes([0.91, 0.15, 0.02, 0.7])
    sm = plt.cm.ScalarMappable(
        cmap='RdYlGn_r',
        norm=plt.Normalize(vmin=0, vmax=1)
    )
    sm.set_array([])
    cbar = fig.colorbar(sm, cax=cbar_ax)
    cbar.set_label('Sensitivity', fontsize=10)
    cbar.set_ticks([0, 0.5, 1])
    cbar.set_ticklabels(['Low', 'Medium', 'High'])

    save_path = os.path.join(
        output_dir, f'occlusion_{model_name}.png')
    plt.savefig(save_path, dpi=150,
                bbox_inches='tight', facecolor='white')
    plt.close()
    del model
    print(f"   Saved: {save_path}")


def plot_occlusion_comparison(all_occ: dict,
                               all_images: dict,
                               output_dir: str):
    """
    Comparison figure: Occlusion maps across all models per class.
    """
    model_names = list(all_occ.keys())
    n_models    = len(model_names)

    for cls in CLASS_NAMES_4:
        fig, axes = plt.subplots(
            1, n_models, figsize=(n_models * 3, 4))
        fig.suptitle(
            f'Occlusion Comparison — '
            f'{CLASS_LABELS.get(cls, cls)}',
            fontsize=13, fontweight='bold'
        )

        for col, model_name in enumerate(model_names):
            if cls in all_occ[model_name]:
                occ_map   = all_occ[model_name][cls]
                img_array = all_images[cls]
                occ_norm  = normalize_map(occ_map)

                colormap = matplotlib.colormaps.get_cmap('RdYlGn_r')
                heat_rgb = colormap(occ_norm)[:, :, :3]
                overlay  = np.clip(
                    img_array * 0.5 + heat_rgb * 0.5, 0, 1)

                axes[col].imshow(overlay)
                axes[col].set_title(
                    model_name.replace('_', '\n'),
                    fontsize=9, fontweight='bold'
                )
            else:
                axes[col].set_facecolor('#f0f0f0')
                axes[col].text(
                    0.5, 0.6, 'N/A',
                    ha='center', va='center',
                    transform=axes[col].transAxes,
                    fontsize=14, fontweight='bold',
                    color='#666666'
                )
                axes[col].text(
                    0.5, 0.35,
                    '(δεν περιλαμβάνει\nκλάση No Tumor)',
                    ha='center', va='center',
                    transform=axes[col].transAxes,
                    fontsize=8, color='#888888'
                )
                axes[col].set_title(
                    model_name.replace('_', '\n'),
                    fontsize=9, fontweight='bold'
                )
            axes[col].axis('off')

        plt.tight_layout()
        save_path = os.path.join(
            output_dir, f'occlusion_comparison_{cls}.png')
        plt.savefig(save_path, dpi=150,
                    bbox_inches='tight', facecolor='white')
        plt.close()
        print(f"   Saved comparison: {save_path}")


# ══════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════

def main():
    print('=' * 60)
    print('OCCLUSION ANALYSIS')
    print('=' * 60)
    print(f'Window size: {WINDOW_SIZE}px')
    print(f'Stride:      {STRIDE}px')
    print(f'Occlude val: {OCCLUDE_VAL} (black)')
    print()

    samples_4class = get_sample_images(
        DATA_PATH, CLASS_NAMES_4, seed=SEED)
    samples_3class = get_sample_images(
        DATA_PATH, CLASS_NAMES_3, seed=SEED)

    all_occ    = {name: {} for name in MODEL_CONFIG}
    all_images = {
        cls: load_image(samples_4class[cls])
        for cls in CLASS_NAMES_4
    }

    for model_name, config in MODEL_CONFIG.items():
        print(f'\n[{model_name}]')

        samples = samples_3class \
            if config['type'] == '3class' \
            else samples_4class

        # Per-model figure
        plot_occlusion_for_model(
            model_name, config, samples, OUTPUT_PATH)

        # Store for comparison
        try:
            model = load_model(model_name, config)

            for cls in config['classes']:
                img_array = load_image(samples[cls])

                if config['type'] == '3class':
                    map_4_to_3 = {
                        'glioma': 0,
                        'meningioma': 1,
                        'pituitary': 2
                    }
                    class_idx = map_4_to_3.get(cls, 0)
                elif config['type'] == 'binary':
                    class_idx = 1
                else:
                    class_idx = CLASS_NAMES_4.index(cls) \
                        if cls in CLASS_NAMES_4 else 0

                occ_map = compute_occlusion_map(
                    model, img_array,
                    class_idx=class_idx,
                    window_size=WINDOW_SIZE,
                    stride=STRIDE,
                    occlude_val=OCCLUDE_VAL
                )
                all_occ[model_name][cls] = occ_map

            del model

        except Exception as e:
            print(f"   Warning: {e}")

    # Comparison figures
    print('\nGenerating comparison figures...')
    plot_occlusion_comparison(all_occ, all_images, OUTPUT_PATH)

    print('\nOcclusion Analysis completed!')
    print(f'Results saved to: {OUTPUT_PATH}')
    print('\nFiles generated:')
    for name in MODEL_CONFIG:
        print(f'  - occlusion_{name}.png')
    for cls in CLASS_NAMES_4:
        print(f'  - occlusion_comparison_{cls}.png')


if __name__ == '__main__':
    main()