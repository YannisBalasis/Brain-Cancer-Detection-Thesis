"""
MedViT V2 — GradCAM XAI
=========================
Grad-CAM visualizations for MedViT V2 targeting the last spatial feature
stage (before GlobalAveragePooling), matching the paper's Figure 4 style.

Usage:
    python medvit_v2_xai.py \\
        --model medvit_v2_experiment/best_medvit_v2_tiny.keras \\
        --data  /path/to/dataset \\
        --output medvit_v2_xai_results \\
        --n_samples 20 \\
        --target_layer auto
"""

import os
import sys
import json
import argparse
import warnings
import numpy as np
from pathlib import Path

warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.cm as cm_lib

import cv2
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing.image import ImageDataGenerator

from medvit_v2_architecture import (
    build_medvit_v2, LayerNorm2D, KANLayer, LFFNLayer, DiNALayer,
    LFPBlock, EMHSALayer, MHCALayer, GFPBlock, StemLayer, PatchEmbedding
)

# ══════════════════════════════════════════════════════════════════════
# PATHS  — edit these to match your local setup
# ══════════════════════════════════════════════════════════════════════
DATA_PATH = '/users/yannisbalasis/documents/thesis/data_multiclass'

# ══════════════════════════════════════════════════════════════════════
CLASS_NAMES = ['glioma', 'meningioma', 'no_tumor', 'pituitary']
IMG_SIZE    = (224, 224)
NUM_CLASSES = 4

CUSTOM_OBJECTS = {
    'LayerNorm2D':    LayerNorm2D,
    'KANLayer':       KANLayer,
    'LFFNLayer':      LFFNLayer,
    'DiNALayer':      DiNALayer,
    'LFPBlock':       LFPBlock,
    'EMHSALayer':     EMHSALayer,
    'MHCALayer':      MHCALayer,
    'GFPBlock':       GFPBlock,
    'StemLayer':      StemLayer,
    'PatchEmbedding': PatchEmbedding,
}

CLASS_COLORS = {
    'glioma':      '#e41a1c',
    'meningioma':  '#377eb8',
    'no_tumor':    '#4daf4a',
    'pituitary':   '#984ea3',
}


# ══════════════════════════════════════════════════════════════════════
# LAYER DISCOVERY
# ══════════════════════════════════════════════════════════════════════

def find_last_spatial_layer(model):
    """
    Find the last layer that outputs a 4-D spatial tensor (B, H, W, C).
    GradCAM requires a spatial feature map, so we skip 1-D and 2-D outputs.
    Preference order: GFPBlock output > LFPBlock output > any Conv output.
    """
    gfp_candidates, lfp_candidates, other_candidates = [], [], []

    for layer in model.layers:
        out = layer.output
        if not hasattr(out, 'shape'):
            continue
        shape = out.shape
        if len(shape) == 4 and shape[1] is not None and shape[1] > 1:
            name = layer.name.lower()
            if 'gfp' in name:
                gfp_candidates.append(layer)
            elif 'lfp' in name:
                lfp_candidates.append(layer)
            elif 'norm' not in name and 'drop' not in name:
                other_candidates.append(layer)

    if gfp_candidates:
        chosen = gfp_candidates[-1]
    elif lfp_candidates:
        chosen = lfp_candidates[-1]
    elif other_candidates:
        chosen = other_candidates[-1]
    else:
        raise RuntimeError('No suitable spatial layer found for GradCAM.')

    print(f'Auto-selected GradCAM target layer: {chosen.name} '
          f'{chosen.output.shape}')
    return chosen.name


def resolve_target_layer(model, target_layer: str) -> str:
    if target_layer == 'auto':
        return find_last_spatial_layer(model)
    # Validate that the named layer exists and is spatial
    layer = model.get_layer(target_layer)
    shape = layer.output.shape
    if len(shape) != 4:
        raise ValueError(
            f'Layer {target_layer} has shape {shape}, expected 4-D spatial tensor.')
    return target_layer


# ══════════════════════════════════════════════════════════════════════
# GRAD-CAM
# ══════════════════════════════════════════════════════════════════════

def make_gradcam_model(model, target_layer_name: str):
    """Build a sub-model that outputs both (feature_map, logits)."""
    feat_layer = model.get_layer(target_layer_name)
    return keras.Model(
        inputs=model.inputs,
        outputs=[feat_layer.output, model.output]
    )


def compute_gradcam(gradcam_model, img_array: np.ndarray, class_idx: int):
    """
    Compute Grad-CAM heatmap for a single image (1, H, W, 3).
    Returns a 2-D float heatmap in [0, 1], same spatial resolution as
    the target feature map.
    """
    img_tensor = tf.cast(img_array, tf.float32)

    with tf.GradientTape() as tape:
        tape.watch(img_tensor)
        feat_maps, logits = gradcam_model(img_tensor, training=False)
        tape.watch(feat_maps)
        class_score = logits[:, class_idx]

    # Gradients w.r.t. the feature maps (not the input)
    grads = tape.gradient(class_score, feat_maps)          # (1,H,W,C)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))   # (C,)

    feat_maps = feat_maps[0]                               # (H,W,C)
    heatmap = feat_maps @ pooled_grads[..., tf.newaxis]    # (H,W,1)
    heatmap = tf.squeeze(heatmap)                          # (H,W)
    heatmap = tf.nn.relu(heatmap)

    # Normalise
    heatmap = heatmap.numpy()
    max_val = heatmap.max()
    if max_val > 0:
        heatmap /= max_val
    return heatmap


def overlay_heatmap(img_rgb: np.ndarray, heatmap: np.ndarray,
                    alpha: float = 0.45, colormap: int = cv2.COLORMAP_JET):
    """
    Resize heatmap to image size and blend with the original image.
    img_rgb: (H, W, 3) float in [0, 1]
    Returns: (H, W, 3) uint8 blended image.
    """
    h, w = img_rgb.shape[:2]
    hm_uint8 = np.uint8(255 * heatmap)
    hm_resized = cv2.resize(hm_uint8, (w, h))
    hm_color = cv2.applyColorMap(hm_resized, colormap)
    hm_color = cv2.cvtColor(hm_color, cv2.COLOR_BGR2RGB)

    img_uint8 = np.uint8(255 * np.clip(img_rgb, 0, 1))
    blended = cv2.addWeighted(img_uint8, 1 - alpha, hm_color, alpha, 0)
    return blended


# ══════════════════════════════════════════════════════════════════════
# DATA LOADING
# ══════════════════════════════════════════════════════════════════════

def load_samples(data_path: str, n_samples: int, seed: int = 42):
    """
    Load n_samples images from the validation split with their true labels.
    Returns (images_raw, y_true, filenames) where images_raw are in [0,1].
    """
    datagen = ImageDataGenerator(rescale=1.0 / 255, validation_split=0.2)
    gen = datagen.flow_from_directory(
        data_path,
        target_size=IMG_SIZE,
        batch_size=1,
        classes=CLASS_NAMES,
        class_mode='sparse',
        subset='validation',
        shuffle=True,
        seed=seed,
    )

    images, labels, fnames = [], [], []
    total = min(n_samples, gen.samples)
    for _ in range(total):
        x, y = next(gen)
        images.append(x[0])
        labels.append(int(y[0]))
        fnames.append(gen.filenames[gen.batch_index - 1])

    print(f'Loaded {len(images)} samples for XAI.')
    return np.array(images), np.array(labels), fnames


# ══════════════════════════════════════════════════════════════════════
# VISUALISATION HELPERS
# ══════════════════════════════════════════════════════════════════════

def plot_single_gradcam(img_rgb, heatmap, true_cls, pred_cls, prob,
                        idx, output_dir: Path):
    """Save a 3-panel figure: original | heatmap | overlay."""
    overlay = overlay_heatmap(img_rgb, heatmap)

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))

    axes[0].imshow(img_rgb)
    axes[0].set_title('Original', fontsize=12)
    axes[0].axis('off')

    axes[1].imshow(heatmap, cmap='jet', vmin=0, vmax=1)
    axes[1].set_title('Grad-CAM Heatmap', fontsize=12)
    axes[1].axis('off')

    axes[2].imshow(overlay)
    axes[2].set_title('Overlay', fontsize=12)
    axes[2].axis('off')

    correct = true_cls == pred_cls
    color   = 'green' if correct else 'red'
    fig.suptitle(
        f'True: {CLASS_NAMES[true_cls]}  |  '
        f'Pred: {CLASS_NAMES[pred_cls]} ({prob:.3f})',
        fontsize=13, fontweight='bold', color=color
    )
    plt.tight_layout()
    path = output_dir / f'gradcam_{idx:04d}_{CLASS_NAMES[true_cls]}.png'
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)


def plot_class_montage(images_rgb, heatmaps, true_labels, pred_labels,
                       probs, output_dir: Path, n_per_row: int = 4):
    """
    Save a per-class montage: top row original, bottom row overlay.
    One figure per class.
    """
    for cls_idx, cls_name in enumerate(CLASS_NAMES):
        mask = [i for i, t in enumerate(true_labels) if t == cls_idx]
        if not mask:
            continue

        n = min(len(mask), n_per_row * 2)
        selected = mask[:n]

        fig, axes = plt.subplots(2, n, figsize=(3 * n, 7))
        if n == 1:
            axes = axes.reshape(2, 1)

        for col, i in enumerate(selected):
            axes[0, col].imshow(images_rgb[i])
            axes[0, col].axis('off')
            correct = true_labels[i] == pred_labels[i]
            edge_col = 'green' if correct else 'red'
            for spine in axes[0, col].spines.values():
                spine.set_edgecolor(edge_col)
                spine.set_linewidth(3)
                spine.set_visible(True)

            overlay = overlay_heatmap(images_rgb[i], heatmaps[i])
            axes[1, col].imshow(overlay)
            axes[1, col].set_title(
                f'Pred: {CLASS_NAMES[pred_labels[i]]}\n({probs[i]:.2f})',
                fontsize=8)
            axes[1, col].axis('off')

        fig.suptitle(
            f'MedViT V2 — Grad-CAM — {cls_name.capitalize()}',
            fontsize=14, fontweight='bold',
            color=CLASS_COLORS.get(cls_name, 'black')
        )
        axes[0, 0].set_ylabel('Original', fontsize=10, rotation=90, labelpad=4)
        axes[1, 0].set_ylabel('Grad-CAM', fontsize=10, rotation=90, labelpad=4)

        plt.tight_layout()
        path = output_dir / f'montage_{cls_name}.png'
        fig.savefig(path, dpi=200, bbox_inches='tight')
        plt.close(fig)
        print(f'Montage saved: {path}')


def plot_summary_grid(images_rgb, heatmaps, true_labels, pred_labels, probs,
                      output_dir: Path, max_show: int = 16):
    """Compact grid: original + overlay for up to max_show samples."""
    n = min(len(images_rgb), max_show)
    ncols = 8
    nrows = (n * 2 + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 2, nrows * 2.2))
    axes = axes.flatten()

    for ax in axes:
        ax.axis('off')

    for i in range(n):
        ax_orig    = axes[i * 2]
        ax_overlay = axes[i * 2 + 1]

        ax_orig.imshow(images_rgb[i])
        ax_orig.axis('off')

        overlay = overlay_heatmap(images_rgb[i], heatmaps[i])
        correct = true_labels[i] == pred_labels[i]
        ax_overlay.imshow(overlay)
        ax_overlay.set_title(
            f'{"✓" if correct else "✗"} {CLASS_NAMES[pred_labels[i]][:4]}',
            fontsize=7, color='green' if correct else 'red'
        )
        ax_overlay.axis('off')

    fig.suptitle('MedViT V2 — Grad-CAM Summary Grid\n'
                 '(left=original, right=overlay)',
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    path = output_dir / 'gradcam_summary_grid.png'
    fig.savefig(path, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f'Summary grid saved: {path}')


# ══════════════════════════════════════════════════════════════════════
# STATS ON XAI RESULTS
# ══════════════════════════════════════════════════════════════════════

def compute_xai_stats(true_labels, pred_labels, probs, heatmaps):
    """
    Compute basic statistics on the GradCAM session:
    - per-class accuracy, mean confidence, mean heatmap activation
    """
    stats = {'per_class': {}, 'overall': {}}
    preds = np.array(pred_labels)
    trues = np.array(true_labels)
    correct = (preds == trues).astype(float)

    stats['overall']['n_samples']  = len(trues)
    stats['overall']['accuracy']   = float(correct.mean())
    stats['overall']['mean_conf']  = float(np.array(probs).mean())
    stats['overall']['mean_heatmap_activation'] = float(
        np.mean([h.mean() for h in heatmaps]))

    for cls_idx, cls_name in enumerate(CLASS_NAMES):
        mask = trues == cls_idx
        if mask.sum() == 0:
            continue
        cls_hm = [heatmaps[i] for i in range(len(trues)) if mask[i]]
        stats['per_class'][cls_name] = {
            'n': int(mask.sum()),
            'accuracy': float(correct[mask].mean()),
            'mean_confidence': float(np.array(probs)[mask].mean()),
            'mean_heatmap_activation': float(np.mean([h.mean() for h in cls_hm])),
            'mean_heatmap_peak':       float(np.mean([h.max()  for h in cls_hm])),
        }

    return stats


def plot_heatmap_activation_bars(stats, output_dir: Path):
    """Bar chart of mean heatmap activation per class."""
    cls_names = list(stats['per_class'].keys())
    activations = [stats['per_class'][c]['mean_heatmap_activation']
                   for c in cls_names]
    peaks = [stats['per_class'][c]['mean_heatmap_peak'] for c in cls_names]
    accs  = [stats['per_class'][c]['accuracy'] for c in cls_names]

    x     = np.arange(len(cls_names))
    width = 0.28
    colors = [CLASS_COLORS.get(c, '#888888') for c in cls_names]

    fig, ax = plt.subplots(figsize=(10, 5))
    bars1 = ax.bar(x - width, activations, width, label='Mean Activation', alpha=0.85)
    bars2 = ax.bar(x,         peaks,       width, label='Mean Peak',        alpha=0.85)
    bars3 = ax.bar(x + width, accs,        width, label='Accuracy',         alpha=0.85,
                   color=[c + '88' for c in colors])

    for bar, col in zip(bars1, colors):
        bar.set_color(col)

    ax.set_xticks(x)
    ax.set_xticklabels([c.replace('_', '\n') for c in cls_names], fontsize=11)
    ax.set_ylabel('Value', fontsize=12)
    ax.set_title('MedViT V2 — Grad-CAM Activation Statistics per Class',
                 fontsize=13, fontweight='bold')
    ax.legend(fontsize=10)
    ax.set_ylim(0, 1.15)
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    path = output_dir / 'heatmap_activation_stats.png'
    fig.savefig(path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f'Activation stats saved: {path}')


# ══════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════

def run_xai(args):
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── 1. Load model ───────────────────────────────────────────────
    print(f'Loading model: {args.model}')
    model = keras.models.load_model(
        args.model,
        custom_objects=CUSTOM_OBJECTS,
        compile=False,
    )
    print('Model loaded.')

    # ── 2. Resolve target layer ─────────────────────────────────────
    target_layer = resolve_target_layer(model, args.target_layer)

    # ── 3. Build GradCAM sub-model ──────────────────────────────────
    gradcam_model = make_gradcam_model(model, target_layer)

    # ── 4. Load images ──────────────────────────────────────────────
    images_rgb, true_labels, fnames = load_samples(
        args.data, n_samples=args.n_samples, seed=args.seed)

    # ── 5. Compute GradCAMs ─────────────────────────────────────────
    heatmaps, pred_labels, confidences = [], [], []

    print(f'\nGenerating Grad-CAM for {len(images_rgb)} samples ...')
    for i, (img, true_cls) in enumerate(zip(images_rgb, true_labels)):
        img_batch = img[np.newaxis]                            # (1,H,W,3)
        logits    = model(img_batch, training=False).numpy()   # (1,4)
        pred_cls  = int(np.argmax(logits[0]))
        conf      = float(logits[0, pred_cls])

        heatmap = compute_gradcam(gradcam_model, img_batch, pred_cls)
        heatmaps.append(heatmap)
        pred_labels.append(pred_cls)
        confidences.append(conf)

        if (i + 1) % 10 == 0 or i == 0:
            print(f'  [{i+1}/{len(images_rgb)}] '
                  f'true={CLASS_NAMES[true_cls]} '
                  f'pred={CLASS_NAMES[pred_cls]} '
                  f'conf={conf:.3f}')

        # Save individual figure if requested
        if args.save_individual:
            plot_single_gradcam(img, heatmap, true_cls, pred_cls, conf,
                                i, output_dir)

    # ── 6. Visualisations ───────────────────────────────────────────
    print('\nGenerating visualisations ...')
    plot_class_montage(images_rgb, heatmaps, true_labels, pred_labels,
                       confidences, output_dir, n_per_row=args.n_per_row)
    plot_summary_grid(images_rgb, heatmaps, true_labels, pred_labels,
                      confidences, output_dir)

    # ── 7. XAI stats ────────────────────────────────────────────────
    xai_stats = compute_xai_stats(true_labels, pred_labels,
                                   confidences, heatmaps)
    plot_heatmap_activation_bars(xai_stats, output_dir)

    acc = xai_stats['overall']['accuracy']
    print(f'\nXAI session accuracy on {len(images_rgb)} samples: '
          f'{acc:.4f} ({acc*100:.2f}%)')

    # Save JSON stats
    xai_stats['target_layer'] = target_layer
    xai_stats['model_path']   = str(args.model)
    with open(output_dir / 'xai_stats.json', 'w') as f:
        json.dump(xai_stats, f, indent=2)

    print(f'\nAll XAI results saved to: {output_dir}')
    return xai_stats


def parse_args():
    p = argparse.ArgumentParser(description='MedViT V2 — GradCAM XAI')
    p.add_argument('--model',      required=True,
                   help='Path to .keras or .h5 model file')
    p.add_argument('--data',       default=DATA_PATH,
                   help='Path to 4-class dataset directory')
    p.add_argument('--output',     default='medvit_v2_xai_results')
    p.add_argument('--n_samples',  type=int, default=40,
                   help='Number of images to explain')
    p.add_argument('--target_layer', default='auto',
                   help='Layer name for GradCAM (or "auto")')
    p.add_argument('--n_per_row',  type=int, default=4,
                   help='Columns in per-class montage')
    p.add_argument('--save_individual', action='store_true',
                   help='Save one PNG per image (can be many files)')
    p.add_argument('--seed',       type=int, default=42)
    return p.parse_args()


if __name__ == '__main__':
    run_xai(parse_args())
