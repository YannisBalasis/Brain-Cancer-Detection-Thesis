#!/usr/bin/env python3
"""
SHAP Visualization for all Brain Tumor Classification Models
=============================================================
Generates SHAP explanations for 1 image per class per model.
Uses GradientExplainer for CNN models (faster than KernelExplainer).

Author: Yannis Balasis
Date: March 2026
"""

import os
import sys
import random
import warnings
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
from PIL import Image
import tensorflow as tf
import shap

warnings.filterwarnings('ignore')

# ── Configuration ──────────────────────────────────────────
DATA_PATH   = '/users/yannisbalasis/documents/thesis/data_multiclass'
OUTPUT_PATH = './xai_results/shap'
SEED        = 42

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

# Number of background samples for SHAP
# Higher = more accurate but slower
N_BACKGROUND = 20

MODEL_CONFIG = {
    'Binary_CNN': {
        'path':    './best_binary_model.h5',
        'classes': CLASS_NAMES_4,
        'type':    'binary',
        'n_outputs': 1
    },
    '4Class_CNN': {
        'path':    './multiclass_4class_experiment_20251217_141321/best_multiclass_4class_model.h5',
        'classes': CLASS_NAMES_4,
        'type':    '4class',
        'n_outputs': 4
    },
    '3Class_CNN': {
        'path':    './multiclass_3class_experiment_20260201_161822/best_multiclass_3class_model.h5',
        'classes': CLASS_NAMES_3,
        'type':    '3class',
        'n_outputs': 3
    },
    'ResNet_Dual': {
        'path':    './dual_system_experiment_20260227_114800/models/best_dual_system_model.h5',
        'classes': CLASS_NAMES_4,
        'type':    'dual',
        'n_outputs': 4
    },
    'EfficientNet_Dual': {
        'path':    './efnet_dual_system_experiment_20260305_194201/models/efnet_dual_phase3.h5',
        'classes': CLASS_NAMES_4,
        'type':    'dual',
        'n_outputs': 4
    },
    'Multi_Dual': {
        'path':    './multi_dual_system_experiment_20260310_100816/best_multi_dual_model.h5',
        'classes': CLASS_NAMES_4,
        'type':    'multi_dual',
        'n_outputs': 4,
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
            from multi_dual_system_architecture import MultiDualSystemArchitecture
            arch = MultiDualSystemArchitecture()
            custom_objects = {
                '_masked_sparse_crossentropy': arch._masked_sparse_crossentropy,
                '_masked_sparse_accuracy':     arch._masked_sparse_accuracy
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


def get_model_predict_fn(model, config: dict):
    """
    Returns a prediction function that always outputs
    a 2D array (n_samples, n_classes) regardless of model type.
    Needed for SHAP compatibility.
    """
    def predict_fn(x):
        x_tensor = tf.cast(x, tf.float32)
        preds    = model(x_tensor, training=False)

        # Handle dict outputs (Multi-Dual)
        if isinstance(preds, dict):
            preds = list(preds.values())[0]
        elif isinstance(preds, list):
            preds = preds[0]

        preds = preds.numpy()

        # Handle binary sigmoid → convert to 2-class
        if preds.shape[-1] == 1:
            preds = np.concatenate(
                [1 - preds, preds], axis=1)

        return preds

    return predict_fn


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


def get_background_images(data_path: str, classes: list,
                           n: int = 20, seed: int = 42) -> np.ndarray:
    """
    Load n random background images for SHAP.
    Background represents the 'baseline' distribution.
    """
    random.seed(seed)
    all_images = []
    per_class  = max(1, n // len(classes))

    for cls in classes:
        cls_path = Path(data_path) / cls
        images   = list(cls_path.glob('*.jpg')) + \
                   list(cls_path.glob('*.png')) + \
                   list(cls_path.glob('*.jpeg'))
        selected = random.sample(images, min(per_class, len(images)))
        for img_path in selected:
            all_images.append(load_image(str(img_path)))

    background = np.array(all_images[:n], dtype=np.float32)
    print(f"   Background samples: {len(background)}")
    return background


# ══════════════════════════════════════════════════════════
# SHAP COMPUTATION
# ══════════════════════════════════════════════════════════

def compute_shap_values(model, predict_fn, img_array: np.ndarray,
                         background: np.ndarray,
                         class_idx: int = None) -> np.ndarray:
    """
    Compute SHAP values. Always returns shape (H, W, 3).
    """
    img_batch = img_array[np.newaxis, ...]

    if class_idx is None:
        preds     = predict_fn(img_batch)
        class_idx = int(np.argmax(preds[0]))

    try:
        # Convert Sequential to Functional if needed
        if isinstance(model, tf.keras.Sequential):
            inp = tf.keras.Input(shape=(224, 224, 3))
            out = model(inp)
            explain_model = tf.keras.Model(
                inputs=inp, outputs=out)
        elif isinstance(model.output, dict):
            explain_model = tf.keras.Model(
                inputs=model.inputs,
                outputs=list(model.output.values())[0]
            )
        elif isinstance(model.output, list):
            explain_model = tf.keras.Model(
                inputs=model.inputs,
                outputs=model.output[0]
            )
        else:
            explain_model = model

        explainer   = shap.GradientExplainer(
            explain_model, background)
        shap_values = explainer.shap_values(img_batch)

        if isinstance(shap_values, list):
            # Clamp class_idx to valid range
            safe_idx = min(class_idx, len(shap_values) - 1)
            sv = shap_values[safe_idx][0]
        else:
            sv = np.array(shap_values)
            if sv.ndim == 5:
                # (1, H, W, 3, n_classes)
                safe_idx = min(class_idx, sv.shape[-1] - 1)
                sv = sv[0, :, :, :, safe_idx]
            elif sv.ndim == 4:
                sv = sv[0]
            elif sv.ndim == 3:
                pass

        sv = np.squeeze(sv)

        if sv.ndim == 4:
            safe_idx = min(class_idx, sv.shape[-1] - 1)
            sv = sv[:, :, :, safe_idx]
        elif sv.ndim == 2:
            sv = np.stack([sv] * 3, axis=-1)

        return sv

    except Exception as e:
        print(f"   SHAP computation error: {e}")
        return np.zeros_like(img_array)

def shap_to_heatmap(shap_values: np.ndarray) -> np.ndarray:
    """
    Convert SHAP values (H, W, 3) to a single heatmap (H, W).
    Uses the mean absolute value across color channels.
    """
    return np.mean(np.abs(shap_values), axis=2)


def normalize_shap(shap_values: np.ndarray) -> np.ndarray:
    """Normalize SHAP values to [-1, 1] for visualization."""
    max_val = np.max(np.abs(shap_values)) + 1e-8
    return shap_values / max_val


# ══════════════════════════════════════════════════════════
# VISUALIZATION
# ══════════════════════════════════════════════════════════

def plot_shap_for_model(model_name: str, config: dict,
                         samples: dict, background: np.ndarray,
                         output_dir: str):
    """
    Creates SHAP visualization for one model.
    Layout per class: [Original | SHAP Heatmap | Positive | Negative]
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
        f'SHAP Visualization — {model_name.replace("_", " ")}',
        fontsize=14, fontweight='bold', y=1.01
    )

    col_titles = [
        'Πρωτότυπη Εικόνα',
        'SHAP Heatmap\n(|contribution|)',
        'Θετική Συνεισφορά\n(κόκκινο = ωθεί προς κλάση)',
        'Αρνητική Συνεισφορά\n(μπλε = ωθεί μακριά)'
    ]
    for col, title in enumerate(col_titles):
        axes[0, col].set_title(
            title, fontsize=9, fontweight='bold', pad=8)

    model      = load_model(model_name, config)
    predict_fn = get_model_predict_fn(model, config)

    for row, cls in enumerate(classes):
        img_path  = samples[cls]
        img_array = load_image(img_path)

        # Determine class index
        preds     = predict_fn(img_array[np.newaxis, ...])
        pred_idx  = int(np.argmax(preds[0]))

        # Map 4-class index to 3-class if needed
        if config['type'] == '3class':
            map_4_to_3 = {'glioma': 0, 'meningioma': 1, 'pituitary': 2}
            class_idx  = map_4_to_3.get(cls, 0)
        else:
            class_idx  = CLASS_NAMES_4.index(cls) \
                if cls in CLASS_NAMES_4 else pred_idx

        print(f"   Computing SHAP for {cls} "
              f"(class_idx={class_idx})...")
        print(f"   Computing SHAP for {cls} "
              f"(class_idx={class_idx})...")

        try:
            shap_vals = compute_shap_values(
                model, predict_fn, img_array,
                background, class_idx
            )
        except Exception as e:
            print(f"   SHAP failed for {cls}: {e}")
            shap_vals = np.zeros_like(img_array)

        # Row label



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

        # Col 2: SHAP absolute heatmap
        heatmap = shap_to_heatmap(shap_vals)
        heatmap_norm = (heatmap - heatmap.min()) / \
                       (heatmap.max() - heatmap.min() + 1e-8)
        im2 = axes[row, 1].imshow(
            heatmap_norm, cmap='hot', vmin=0, vmax=1)
        axes[row, 1].axis('off')

       # Col 3: Positive contributions (red overlay)
        shap_norm = normalize_shap(shap_vals)
        positive  = np.mean(shap_norm, axis=2)
        positive  = np.where(positive > 0, positive, 0)
        # Normalize positive to [0,1] for visibility
        if positive.max() > 0:
            positive = positive / positive.max()

        red_overlay = np.zeros((224, 224, 3))
        red_overlay[:, :, 0] = positive  # Red channel
        # Show red on black background, not overlaid
        vis_pos = np.clip(
            img_array * 0.3 + red_overlay * 0.7, 0, 1)
        axes[row, 2].imshow(vis_pos)
        axes[row, 2].axis('off')

        # Col 4: Negative contributions (blue overlay)
        negative = np.mean(shap_norm, axis=2)
        negative = np.where(negative < 0, -negative, 0)
        # Normalize negative to [0,1] for visibility
        if negative.max() > 0:
            negative = negative / negative.max()

        blue_overlay = np.zeros((224, 224, 3))
        blue_overlay[:, :, 2] = negative  # Blue channel
        vis_neg = np.clip(
            img_array * 0.3 + blue_overlay * 0.7, 0, 1)
        axes[row, 3].imshow(vis_neg)
        axes[row, 3].axis('off')
        
    # Colorbar for heatmap
    fig.subplots_adjust(right=0.88)
    cbar_ax = fig.add_axes([0.91, 0.15, 0.02, 0.7])
    sm = plt.cm.ScalarMappable(
        cmap='hot', norm=plt.Normalize(vmin=0, vmax=1))
    sm.set_array([])
    cbar = fig.colorbar(sm, cax=cbar_ax)
    cbar.set_label('|SHAP Value|', fontsize=10)
    cbar.set_ticks([0, 0.5, 1])
    cbar.set_ticklabels(['Low', 'Medium', 'High'])

    save_path = os.path.join(
        output_dir, f'shap_{model_name}.png')
    plt.savefig(save_path, dpi=150,
                bbox_inches='tight', facecolor='white')
    plt.close()
    del model
    print(f"   Saved: {save_path}")


def plot_shap_comparison(all_shap: dict, all_images: dict,
                          output_dir: str):
    """
    Comparison figure: SHAP heatmaps across all models per class.
    """
    model_names = list(all_shap.keys())
    n_models    = len(model_names)

    for cls in CLASS_NAMES_4:
        fig, axes = plt.subplots(
            1, n_models, figsize=(n_models * 3, 4))
        fig.suptitle(
            f'SHAP Comparison — {CLASS_LABELS.get(cls, cls)}',
            fontsize=13, fontweight='bold'
        )

        for col, model_name in enumerate(model_names):
            if cls in all_shap[model_name]:
                shap_vals = all_shap[model_name][cls]
                img_array = all_images[cls]

                heatmap      = shap_to_heatmap(shap_vals)
                heatmap_norm = (heatmap - heatmap.min()) / \
                               (heatmap.max() - heatmap.min() + 1e-8)

                # Overlay heatmap on image
                colormap = matplotlib.colormaps.get_cmap('hot')
                heat_rgb = colormap(heatmap_norm)[:, :, :3]
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
            output_dir, f'shap_comparison_{cls}.png')
        plt.savefig(save_path, dpi=150,
                    bbox_inches='tight', facecolor='white')
        plt.close()
        print(f"   Saved comparison: {save_path}")


# ══════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════

def main():
    print('=' * 60)
    print('SHAP VISUALIZATION')
    print('=' * 60)
    print(f'SHAP version: {shap.__version__}')
    print(f'Background samples per model: {N_BACKGROUND}')
    print()

    # Load sample images
    samples_4class = get_sample_images(
        DATA_PATH, CLASS_NAMES_4, seed=SEED)
    samples_3class = get_sample_images(
        DATA_PATH, CLASS_NAMES_3, seed=SEED)

    # Store SHAP values for comparison
    all_shap   = {name: {} for name in MODEL_CONFIG}
    all_images = {
        cls: load_image(samples_4class[cls])
        for cls in CLASS_NAMES_4
    }

    for model_name, config in MODEL_CONFIG.items():
        print(f'\n[{model_name}]')

        samples = samples_3class \
            if config['type'] == '3class' \
            else samples_4class

        # Load background for this model
        background = get_background_images(
            DATA_PATH, config['classes'],
            n=N_BACKGROUND, seed=SEED
        )

        # Per-model SHAP figure
        plot_shap_for_model(
            model_name, config,
            samples, background, OUTPUT_PATH
        )

        # Store for comparison
        try:
            model      = load_model(model_name, config)
            predict_fn = get_model_predict_fn(model, config)

            for cls in config['classes']:
                img_array = load_image(samples[cls])

                if config['type'] == '3class':
                    map_4_to_3 = {
                        'glioma': 0,
                        'meningioma': 1,
                        'pituitary': 2
                    }
                    class_idx = map_4_to_3.get(cls, 0)
                else:
                    class_idx = CLASS_NAMES_4.index(cls) \
                        if cls in CLASS_NAMES_4 else 0

                shap_vals = compute_shap_values(
                    model, predict_fn, img_array,
                    background, class_idx
                )
                all_shap[model_name][cls] = shap_vals

            del model

        except Exception as e:
            print(f"   Warning: {e}")

    # Comparison figures
    print('\nGenerating comparison figures...')
    plot_shap_comparison(all_shap, all_images, OUTPUT_PATH)

    print('\nSHAP completed!')
    print(f'Results saved to: {OUTPUT_PATH}')
    print('\nFiles generated:')
    for name in MODEL_CONFIG:
        print(f'  - shap_{name}.png')
    for cls in CLASS_NAMES_4:
        print(f'  - shap_comparison_{cls}.png')


if __name__ == '__main__':
    main()