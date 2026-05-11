#!/usr/bin/env python3
"""
Grad-CAM Visualization for all Brain Tumor Classification Models
=================================================================
Generates Grad-CAM heatmaps for 1 image per class per model.

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
import matplotlib.cm as cm
from pathlib import Path
from PIL import Image
import tensorflow as tf

# ── Configuration ──────────────────────────────────────────
DATA_PATH   = '/users/yannisbalasis/documents/thesis/data_multiclass'
OUTPUT_PATH = './xai_results/gradcam'
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

MODEL_CONFIG = {
    'Binary_CNN': {
        'path':         './best_binary_model.h5',
        'target_layer': 'conv2d_7',
        'classes':      CLASS_NAMES_4,
        'type':         'binary'
    },
    '4Class_CNN': {
        'path':         './multiclass_4class_experiment_20251217_141321/best_multiclass_4class_model.h5',
        'target_layer': 'conv4_2',
        'classes':      CLASS_NAMES_4,
        'type':         '4class'
    },
    '3Class_CNN': {
        'path':         './multiclass_3class_experiment_20260201_161822/best_multiclass_3class_model.h5',
        'target_layer': 'conv4_2',
        'classes':      CLASS_NAMES_3,
        'type':         '3class'
    },
    'ResNet_Dual': {
        'path':         './dual_system_experiment_20260227_114800/models/best_dual_system_model.h5',
        'target_layer': 'custom_conv4_2',
        'classes':      CLASS_NAMES_4,
        'type':         'dual'
    },
    'EfficientNet_Dual': {
        'path':         './efnet_dual_system_experiment_20260305_194201/models/efnet_dual_phase3.h5',
        'target_layer': 'custom_conv4_2',
        'classes':      CLASS_NAMES_4,
        'type':         'dual'
    },
    'Multi_Dual': {
        'path':         './multi_dual_system_experiment_20260310_100816/best_multi_dual_model.h5',
        'target_layer': 'conv4_2',
        'classes':      CLASS_NAMES_4,
        'type':         'multi_dual',
        'custom_objects': True
    },
}


# ══════════════════════════════════════════════════════════
# MODEL LOADING
# ══════════════════════════════════════════════════════════

def load_model(model_name: str, config: dict):
    """Load model and build it if Sequential."""
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
            print(f"   Warning: {e}. Loading without custom objects.")
            model = tf.keras.models.load_model(
                config['path'], compile=False)
    else:
        model = tf.keras.models.load_model(
            config['path'], compile=False)

    # For Sequential models: build with dummy input
    if isinstance(model, tf.keras.Sequential):
        dummy = tf.zeros((1, 224, 224, 3))
        _ = model(dummy)
        print(f"   Built Sequential model with dummy input")

    print(f"   Loaded: {config['path']}")
    return model


# ══════════════════════════════════════════════════════════
# IMAGE LOADING
# ══════════════════════════════════════════════════════════

def load_image(img_path: str, size: int = 224) -> np.ndarray:
    """Load and preprocess image for model input."""
    img = Image.open(img_path).convert('RGB').resize((size, size))
    img_array = np.array(img, dtype=np.float32) / 255.0
    return img_array


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
# GRAD-CAM IMPLEMENTATION
# ══════════════════════════════════════════════════════════

def compute_gradcam_sequential(model, img_array: np.ndarray,
                                target_layer_name: str,
                                class_idx: int = None) -> np.ndarray:
    """
    Grad-CAM for Sequential models.
    Splits model into two sub-models at the target layer.
    """
    layer_names = [l.name for l in model.layers]
    if target_layer_name not in layer_names:
        raise ValueError(
            f"Layer '{target_layer_name}' not found. "
            f"Available: {layer_names}"
        )

    target_idx = layer_names.index(target_layer_name)

    # Sub-model 1: input → target conv layer (inclusive)
    feature_extractor = tf.keras.Sequential(
        model.layers[:target_idx + 1]
    )
    feature_extractor.build((None, 224, 224, 3))

    # Sub-model 2: layers after target conv → output
    classifier = tf.keras.Sequential(
        model.layers[target_idx + 1:]
    )

    img_batch = tf.cast(
        np.expand_dims(img_array, axis=0), tf.float32)

    with tf.GradientTape() as tape:
        conv_output = feature_extractor(img_batch)
        tape.watch(conv_output)
        predictions = classifier(conv_output)

        # Handle binary sigmoid output
        if predictions.shape[-1] == 1:
            loss = predictions[0][0]
        else:
            if class_idx is None:
                class_idx = int(tf.argmax(predictions[0]))
            loss = predictions[0][class_idx]

    grads        = tape.gradient(loss, conv_output)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    heatmap      = conv_output[0] @ pooled_grads[..., tf.newaxis]
    heatmap      = tf.squeeze(heatmap)
    heatmap      = tf.maximum(heatmap, 0) / (
        tf.math.reduce_max(heatmap) + 1e-8)

    return heatmap.numpy()


def compute_gradcam_functional(model, img_array: np.ndarray,
                                target_layer_name: str,
                                class_idx: int = None) -> np.ndarray:
    """
    Grad-CAM for Functional models.
    """
    # Get first output for multi-output models
    outputs = model.output
    if isinstance(outputs, dict):
        first_output = list(outputs.values())[0]
    elif isinstance(outputs, list):
        first_output = outputs[0]
    else:
        first_output = outputs

    grad_model = tf.keras.models.Model(
        inputs=model.inputs,
        outputs=[
            model.get_layer(target_layer_name).output,
            first_output
        ]
    )

    img_batch = tf.cast(
        np.expand_dims(img_array, axis=0), tf.float32)

    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_batch)

        if isinstance(predictions, dict):
            predictions = list(predictions.values())[0]
        elif isinstance(predictions, list):
            predictions = predictions[0]

        if predictions.shape[-1] == 1:
            loss = predictions[0][0]
        else:
            if class_idx is None:
                class_idx = int(tf.argmax(predictions[0]))
            loss = predictions[0][class_idx]

    grads        = tape.gradient(loss, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    heatmap      = conv_outputs[0] @ pooled_grads[..., tf.newaxis]
    heatmap      = tf.squeeze(heatmap)
    heatmap      = tf.maximum(heatmap, 0) / (
        tf.math.reduce_max(heatmap) + 1e-8)

    return heatmap.numpy()


def compute_gradcam(model, img_array: np.ndarray,
                    target_layer_name: str,
                    class_idx: int = None) -> np.ndarray:
    """
    Grad-CAM dispatcher — routes to correct implementation
    based on model type (Sequential vs Functional).
    """
    if isinstance(model, tf.keras.Sequential):
        return compute_gradcam_sequential(
            model, img_array, target_layer_name, class_idx)
    else:
        return compute_gradcam_functional(
            model, img_array, target_layer_name, class_idx)


def overlay_heatmap(img_array: np.ndarray,
                    heatmap: np.ndarray,
                    alpha: float = 0.4) -> np.ndarray:
    """
    Overlay Grad-CAM heatmap on original image.
    """
    # Resize heatmap to image size
    heatmap_resized = np.array(
        Image.fromarray(np.uint8(heatmap * 255))
        .resize((img_array.shape[1], img_array.shape[0]),
                Image.LANCZOS)
    ) / 255.0

    # Apply colormap
    colormap    = matplotlib.colormaps.get_cmap('jet')
    heatmap_rgb = colormap(heatmap_resized)[:, :, :3]

    # Overlay
    overlaid = (1 - alpha) * img_array + alpha * heatmap_rgb
    overlaid = np.clip(overlaid, 0, 1)

    return (overlaid * 255).astype(np.uint8)


# ══════════════════════════════════════════════════════════
# VISUALIZATION
# ══════════════════════════════════════════════════════════

def plot_gradcam_for_model(model_name: str, config: dict,
                            samples: dict, output_dir: str):
    """
    Creates Grad-CAM visualization for one model.
    Layout per class: [Original | Heatmap | Overlay]
    """
    classes   = config['classes']
    n_classes = len(classes)

    fig, axes = plt.subplots(
        n_classes, 3,
        figsize=(13, n_classes * 3.5)
    )
    if n_classes == 1:
        axes = axes[np.newaxis, :]

    fig.suptitle(
        f'Grad-CAM Visualization — {model_name.replace("_", " ")}',
        fontsize=14, fontweight='bold', y=1.01
    )

    col_titles = ['Πρωτότυπη Εικόνα', 'Grad-CAM Heatmap', 'Επικάλυψη']
    for col, title in enumerate(col_titles):
        axes[0, col].set_title(
            title, fontsize=11, fontweight='bold', pad=8)

    model = load_model(model_name, config)

    for row, cls in enumerate(classes):
        img_path  = samples[cls]
        img_array = load_image(img_path)

        try:
            heatmap  = compute_gradcam(
                model, img_array, config['target_layer'])
            overlaid = overlay_heatmap(img_array, heatmap)
        except Exception as e:
            print(f"   Warning Grad-CAM failed for {cls}: {e}")
            heatmap  = np.zeros((7, 7))
            overlaid = (img_array * 255).astype(np.uint8)

        heatmap_display = np.array(
            Image.fromarray(np.uint8(heatmap * 255))
            .resize((224, 224), Image.LANCZOS)
        )

        # Row label
        axes[row, 0].set_ylabel(
            CLASS_LABELS.get(cls, cls),
            fontsize=11, fontweight='bold',
            rotation=90, labelpad=10
        )

        # Col 1: Original
        axes[row, 0].imshow(img_array)
        axes[row, 0].axis('off')

        # Col 2: Heatmap
        axes[row, 1].imshow(heatmap_display, cmap='jet',
                             vmin=0, vmax=255)
        axes[row, 1].axis('off')

        # Col 3: Overlay
        axes[row, 2].imshow(overlaid)
        axes[row, 2].axis('off')

    # Colorbar — outside the grid on the right
    fig.subplots_adjust(right=0.88)
    cbar_ax = fig.add_axes([0.91, 0.15, 0.02, 0.7])
    sm = plt.cm.ScalarMappable(
        cmap='jet',
        norm=plt.Normalize(vmin=0, vmax=1)
    )
    sm.set_array([])
    cbar = fig.colorbar(sm, cax=cbar_ax)
    cbar.set_label('Grad-CAM Intensity', fontsize=10)
    cbar.set_ticks([0, 0.5, 1])
    cbar.set_ticklabels(['Low', 'Medium', 'High'])

    save_path = os.path.join(
        output_dir, f'gradcam_{model_name}.png')
    plt.savefig(save_path, dpi=200,
                bbox_inches='tight', facecolor='white')
    plt.close()
    del model
    print(f"   Saved: {save_path}")


def plot_gradcam_comparison(all_results: dict,
                             output_dir: str):
    """
    Comparison figure across all models for one class.
    Shows overlay only.
    """
    classes     = CLASS_NAMES_4
    model_names = list(all_results.keys())
    n_models    = len(model_names)

    for cls in classes:
        fig, axes = plt.subplots(
            1, n_models,
            figsize=(n_models * 3, 4)
        )
        fig.suptitle(
            f'Grad-CAM Comparison — {CLASS_LABELS.get(cls, cls)}',
            fontsize=13, fontweight='bold'
        )

        for col, model_name in enumerate(model_names):
            if cls in all_results[model_name]:
                overlay = all_results[model_name][cls]
                axes[col].imshow(overlay)
            else:
                axes[col].set_facecolor('#1a1a2e')
                axes[col].text(
                    0.5, 0.5, 'N/A',
                    ha='center', va='center',
                    transform=axes[col].transAxes,
                    fontsize=12, color='white'
                )
            axes[col].set_title(
                model_name.replace('_', '\n'),
                fontsize=9, fontweight='bold'
            )
            axes[col].axis('off')

        plt.tight_layout()
        save_path = os.path.join(
            output_dir,
            f'gradcam_comparison_{cls}.png'
        )
        plt.savefig(save_path, dpi=200,
                    bbox_inches='tight', facecolor='white')
        plt.close()
        print(f"   Saved comparison: {save_path}")


# ══════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════

def main():
    print('=' * 60)
    print('GRAD-CAM VISUALIZATION')
    print('=' * 60)

    samples_4class = get_sample_images(
        DATA_PATH, CLASS_NAMES_4, seed=SEED)
    samples_3class = get_sample_images(
        DATA_PATH, CLASS_NAMES_3, seed=SEED)

    all_results = {name: {} for name in MODEL_CONFIG}

    for model_name, config in MODEL_CONFIG.items():
        print(f"\n[{model_name}]")

        samples = samples_3class \
            if config['type'] == '3class' \
            else samples_4class

        # Per-model figure
        plot_gradcam_for_model(
            model_name, config, samples, OUTPUT_PATH)

        # Store overlays for comparison figure
        try:
            model = load_model(model_name, config)
            for cls in config['classes']:
                img_array = load_image(samples[cls])
                heatmap   = compute_gradcam(
                    model, img_array, config['target_layer'])
                overlaid  = overlay_heatmap(img_array, heatmap)
                all_results[model_name][cls] = overlaid
            del model
        except Exception as e:
            print(f"   Warning: {e}")

    # Comparison figures per class
    print("\nGenerating comparison figures...")
    plot_gradcam_comparison(all_results, OUTPUT_PATH)

    print('\nGrad-CAM completed!')
    print(f'Results saved to: {OUTPUT_PATH}')
    print('\nFiles generated:')
    for name in MODEL_CONFIG:
        print(f'  - gradcam_{name}.png')
    for cls in CLASS_NAMES_4:
        print(f'  - gradcam_comparison_{cls}.png')


if __name__ == '__main__':
    main()