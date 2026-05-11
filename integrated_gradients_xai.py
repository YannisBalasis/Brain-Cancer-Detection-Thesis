#!/usr/bin/env python3
"""
Integrated Gradients Visualization for Brain Tumor Classification Models
=========================================================================
Computes Integrated Gradients attributions for 1 image per class per model.

Integrated Gradients formula:
    IG_i(x) = (x_i - x'_i) * integral from 0 to 1 of
              dF(x' + alpha*(x-x')) / dx_i  d_alpha

Where x' is the baseline (black image) and x is the input image.

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
OUTPUT_PATH = './xai_results/integrated_gradients'
SEED        = 42

# Number of steps for approximating the integral
# Higher = more accurate but slower (50 is a good balance)
N_STEPS = 50

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

    # Build Sequential models and wrap as Functional
    if isinstance(model, tf.keras.Sequential):
        dummy = tf.zeros((1, 224, 224, 3))
        _ = model(dummy)
        # Wrap as Functional for consistent API
        inp = tf.keras.Input(shape=(224, 224, 3))
        out = model(inp)
        model = tf.keras.Model(inputs=inp, outputs=out)
        print(f"   Wrapped Sequential as Functional model")

    print(f"   Loaded: {config['path']}")
    return model


def get_single_output_model(model):
    """
    Returns a model with a single output tensor.
    For multi-output models, uses the first output.
    """
    outputs = model.output

    if isinstance(outputs, dict):
        first_output = list(outputs.values())[0]
        return tf.keras.Model(
            inputs=model.inputs,
            outputs=first_output
        )
    elif isinstance(outputs, list):
        return tf.keras.Model(
            inputs=model.inputs,
            outputs=outputs[0]
        )
    else:
        return model


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
# INTEGRATED GRADIENTS
# ══════════════════════════════════════════════════════════

def interpolate_images(baseline: np.ndarray,
                        image: np.ndarray,
                        n_steps: int) -> tf.Tensor:
    """
    Generate interpolated images between baseline and input.

    Returns tensor of shape (n_steps, H, W, C).
    """
    alphas = tf.linspace(0.0, 1.0, n_steps + 1)
    # baseline: (H, W, C) → (1, H, W, C)
    # image:    (H, W, C) → (1, H, W, C)
    baseline_t = tf.cast(baseline[np.newaxis], tf.float32)
    image_t    = tf.cast(image[np.newaxis], tf.float32)

    # Interpolate: (n_steps+1, H, W, C)
    alphas_x   = alphas[:, tf.newaxis, tf.newaxis, tf.newaxis]
    interpolated = baseline_t + alphas_x * (image_t - baseline_t)

    return interpolated


@tf.function
def compute_gradients(model, interpolated_images: tf.Tensor,
                       class_idx: int) -> tf.Tensor:
    """
    Compute gradients of the model output w.r.t. interpolated images.

    Returns gradients tensor of shape (n_steps+1, H, W, C).
    """
    with tf.GradientTape() as tape:
        tape.watch(interpolated_images)
        predictions = model(interpolated_images, training=False)

        # Handle dict/list outputs
        if isinstance(predictions, dict):
            predictions = list(predictions.values())[0]
        elif isinstance(predictions, (list, tuple)):
            predictions = predictions[0]

        # Handle binary sigmoid output
        if predictions.shape[-1] == 1:
            output = predictions[:, 0]
        else:
            safe_idx = tf.minimum(
                tf.cast(class_idx, tf.int32),
                tf.shape(predictions)[1] - 1
            )
            output = predictions[:, safe_idx]

    gradients = tape.gradient(output, interpolated_images)
    return gradients


def integral_approximation(gradients: tf.Tensor) -> tf.Tensor:
    """
    Approximate the integral using the trapezoidal rule.
    gradients shape: (n_steps+1, H, W, C)
    Returns: (H, W, C)
    """
    # Average adjacent gradients (trapezoidal rule)
    grads = (gradients[:-1] + gradients[1:]) / 2.0
    # Average over steps
    avg_grads = tf.reduce_mean(grads, axis=0)
    return avg_grads


def compute_integrated_gradients(model, image: np.ndarray,
                                  class_idx: int = None,
                                  n_steps: int = 50,
                                  baseline: np.ndarray = None
                                  ) -> np.ndarray:
    """
    Compute Integrated Gradients for a single image.

    Args:
        model:      Keras model (single output)
        image:      Input image (H, W, 3) normalized [0,1]
        class_idx:  Class to explain (None = predicted class)
        n_steps:    Number of interpolation steps
        baseline:   Baseline image (None = black image)

    Returns:
        ig_attributions: (H, W, 3) attribution map
    """
    # Default baseline: black image
    if baseline is None:
        baseline = np.zeros_like(image)

    # Determine predicted class
    if class_idx is None:
        img_batch  = image[np.newaxis, ...]
        preds      = model(tf.cast(img_batch, tf.float32),
                           training=False)
        if isinstance(preds, dict):
            preds = list(preds.values())[0]
        elif isinstance(preds, (list, tuple)):
            preds = preds[0]
        preds      = preds.numpy()
        if preds.shape[-1] == 1:
            class_idx = 0
        else:
            class_idx = int(np.argmax(preds[0]))

    # Generate interpolated images
    interpolated = interpolate_images(baseline, image, n_steps)

    # Compute gradients at each interpolation step
    # Process in batches to avoid memory issues
    batch_size = 10
    all_gradients = []

    for i in range(0, n_steps + 1, batch_size):
        batch = interpolated[i:i + batch_size]
        grads = compute_gradients(model, batch, class_idx)
        all_gradients.append(grads)

    all_gradients = tf.concat(all_gradients, axis=0)

    # Approximate integral
    avg_grads = integral_approximation(all_gradients)

    # Multiply by (input - baseline) — the IG formula
    ig = (image - baseline) * avg_grads.numpy()

    return ig  # (H, W, 3)


def ig_to_heatmap(ig: np.ndarray) -> np.ndarray:
    """
    Convert IG attributions (H, W, 3) to heatmap (H, W).
    Uses sum of absolute values across channels.
    """
    return np.sum(np.abs(ig), axis=2)


def normalize_attribution(attr: np.ndarray) -> np.ndarray:
    """Normalize to [-1, 1]."""
    max_val = np.max(np.abs(attr)) + 1e-8
    return attr / max_val


# ══════════════════════════════════════════════════════════
# VISUALIZATION
# ══════════════════════════════════════════════════════════

def plot_ig_for_model(model_name: str, config: dict,
                       samples: dict, output_dir: str):
    """
    Creates Integrated Gradients visualization for one model.
    Layout per class:
        [Original | IG Heatmap | Positive | Negative]
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
        f'Integrated Gradients — {model_name.replace("_", " ")}',
        fontsize=14, fontweight='bold', y=1.01
    )

    col_titles = [
        'Πρωτότυπη Εικόνα',
        'IG Heatmap\n(Σ|attribution|)',
        'Θετική Συνεισφορά\n(ωθεί προς κλάση)',
        'Αρνητική Συνεισφορά\n(ωθεί μακριά)'
    ]
    for col, title in enumerate(col_titles):
        axes[0, col].set_title(
            title, fontsize=9, fontweight='bold', pad=8)

    # Load model and wrap for single output
    raw_model    = load_model(model_name, config)
    model        = get_single_output_model(raw_model)

    for row, cls in enumerate(classes):
        img_path  = samples[cls]
        img_array = load_image(img_path)

        # Determine class index
        if config['type'] == '3class':
            map_4_to_3 = {'glioma': 0, 'meningioma': 1,
                           'pituitary': 2}
            class_idx  = map_4_to_3.get(cls, 0)
        else:
            preds = model(
                tf.cast(img_array[np.newaxis], tf.float32),
                training=False)
            if isinstance(preds, dict):
                preds = list(preds.values())[0]
            preds     = preds.numpy()
            if preds.shape[-1] == 1:
                class_idx = 0
            else:
                class_idx = CLASS_NAMES_4.index(cls) \
                    if cls in CLASS_NAMES_4 else int(np.argmax(preds[0]))

        print(f"   Computing IG for {cls} "
              f"(class_idx={class_idx}, steps={N_STEPS})...")

        try:
            ig = compute_integrated_gradients(
                model, img_array,
                class_idx=class_idx,
                n_steps=N_STEPS
            )
        except Exception as e:
            print(f"   IG failed for {cls}: {e}")
            ig = np.zeros_like(img_array)

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

        # Col 2: IG heatmap (absolute values)
        heatmap      = ig_to_heatmap(ig)
        heatmap_norm = (heatmap - heatmap.min()) / \
                       (heatmap.max() - heatmap.min() + 1e-8)
        axes[row, 1].imshow(heatmap_norm, cmap='inferno',
                             vmin=0, vmax=1)
        axes[row, 1].axis('off')

        # Col 3: Positive attributions (red)
        ig_norm  = normalize_attribution(ig)
        positive = np.mean(ig_norm, axis=2)
        positive = np.where(positive > 0, positive, 0)
        if positive.max() > 0:
            positive = positive / positive.max()

        red_overlay          = np.zeros((224, 224, 3))
        red_overlay[:, :, 0] = positive
        vis_pos = np.clip(
            img_array * 0.3 + red_overlay * 0.7, 0, 1)
        axes[row, 2].imshow(vis_pos)
        axes[row, 2].axis('off')

        # Col 4: Negative attributions (blue)
        negative = np.mean(ig_norm, axis=2)
        negative = np.where(negative < 0, -negative, 0)
        if negative.max() > 0:
            negative = negative / negative.max()

        blue_overlay          = np.zeros((224, 224, 3))
        blue_overlay[:, :, 2] = negative
        vis_neg = np.clip(
            img_array * 0.3 + blue_overlay * 0.7, 0, 1)
        axes[row, 3].imshow(vis_neg)
        axes[row, 3].axis('off')

    # Colorbar
    fig.subplots_adjust(right=0.88)
    cbar_ax = fig.add_axes([0.91, 0.15, 0.02, 0.7])
    sm = plt.cm.ScalarMappable(
        cmap='inferno',
        norm=plt.Normalize(vmin=0, vmax=1)
    )
    sm.set_array([])
    cbar = fig.colorbar(sm, cax=cbar_ax)
    cbar.set_label('IG Attribution', fontsize=10)
    cbar.set_ticks([0, 0.5, 1])
    cbar.set_ticklabels(['Low', 'Medium', 'High'])

    save_path = os.path.join(
        output_dir, f'ig_{model_name}.png')
    plt.savefig(save_path, dpi=150,
                bbox_inches='tight', facecolor='white')
    plt.close()
    del raw_model, model
    print(f"   Saved: {save_path}")


def plot_ig_comparison(all_ig: dict, all_images: dict,
                        output_dir: str):
    """
    Comparison figure: IG heatmaps across all models per class.
    """
    model_names = list(all_ig.keys())
    n_models    = len(model_names)

    for cls in CLASS_NAMES_4:
        fig, axes = plt.subplots(
            1, n_models, figsize=(n_models * 3, 4))
        fig.suptitle(
            f'Integrated Gradients Comparison — '
            f'{CLASS_LABELS.get(cls, cls)}',
            fontsize=13, fontweight='bold'
        )

        for col, model_name in enumerate(model_names):
            if cls in all_ig[model_name]:
                ig        = all_ig[model_name][cls]
                img_array = all_images[cls]

                heatmap      = ig_to_heatmap(ig)
                heatmap_norm = (heatmap - heatmap.min()) / \
                               (heatmap.max() - heatmap.min() + 1e-8)

                colormap = matplotlib.colormaps.get_cmap('inferno')
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
            output_dir, f'ig_comparison_{cls}.png')
        plt.savefig(save_path, dpi=150,
                    bbox_inches='tight', facecolor='white')
        plt.close()
        print(f"   Saved comparison: {save_path}")


# ══════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════

def main():
    print('=' * 60)
    print('INTEGRATED GRADIENTS VISUALIZATION')
    print('=' * 60)
    print(f'Integration steps: {N_STEPS}')
    print(f'Baseline: black image (zeros)')
    print()

    samples_4class = get_sample_images(
        DATA_PATH, CLASS_NAMES_4, seed=SEED)
    samples_3class = get_sample_images(
        DATA_PATH, CLASS_NAMES_3, seed=SEED)

    all_ig     = {name: {} for name in MODEL_CONFIG}
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
        plot_ig_for_model(
            model_name, config, samples, OUTPUT_PATH)

        # Store IG for comparison figure
        try:
            raw_model = load_model(model_name, config)
            model     = get_single_output_model(raw_model)

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

                ig = compute_integrated_gradients(
                    model, img_array,
                    class_idx=class_idx,
                    n_steps=N_STEPS
                )
                all_ig[model_name][cls] = ig

            del raw_model, model

        except Exception as e:
            print(f"   Warning: {e}")

    # Comparison figures
    print('\nGenerating comparison figures...')
    plot_ig_comparison(all_ig, all_images, OUTPUT_PATH)

    print('\nIntegrated Gradients completed!')
    print(f'Results saved to: {OUTPUT_PATH}')
    print('\nFiles generated:')
    for name in MODEL_CONFIG:
        print(f'  - ig_{name}.png')
    for cls in CLASS_NAMES_4:
        print(f'  - ig_comparison_{cls}.png')


if __name__ == '__main__':
    main()