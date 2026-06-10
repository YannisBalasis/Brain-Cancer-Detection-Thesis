"""
MedViT V2 — Full XAI Suite
===========================
Grad-CAM + SHAP (GradientExplainer) + Integrated Gradients + Occlusion Analysis

SHAP note: GradientExplainer is used (not DeepExplainer). If DiNA custom ops
are not differentiable through the SHAP graph, SHAP falls back to zeros with
a warning — the other three methods are unaffected.

Usage:
    python medvit_v2_xai_full.py \\
        --model /path/to/best_medvit_v2_base.keras \\
        --data  /path/to/dataset \\
        --output medvit_v2_xai_full_results \\
        --n_samples 40 \\
        --target_layer auto \\
        --n_background 20 \\
        --ig_steps 50 \\
        --occlusion_window 28 \\
        --occlusion_stride 7
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
import cv2
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import train_test_split

from medvit_v2_architecture import (
    build_medvit_v2, LayerNorm2D, KANLayer, LFFNLayer, DiNALayer,
    LFPBlock, EMHSALayer, MHCALayer, GFPBlock, StemLayer, PatchEmbedding,
)

# ══════════════════════════════════════════════════════════════════════
# CONSTANTS
# ══════════════════════════════════════════════════════════════════════

DATA_PATH   = '/storage/data4/up1084631/data_multiclass'
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
    'glioma':     '#e41a1c',
    'meningioma': '#377eb8',
    'no_tumor':   '#4daf4a',
    'pituitary':  '#984ea3',
}

CLASS_LABELS = {
    'glioma':     'Glioma',
    'meningioma': 'Meningioma',
    'no_tumor':   'No Tumor',
    'pituitary':  'Pituitary',
}


# ══════════════════════════════════════════════════════════════════════
# DATA UTILITIES
# ══════════════════════════════════════════════════════════════════════

def make_file_splits(data_path: str, seed: int = 42):
    rows = []
    for cls in CLASS_NAMES:
        for ext in ('*.jpg', '*.jpeg', '*.png', '*.JPG', '*.JPEG', '*.PNG'):
            for fp in Path(data_path).joinpath(cls).glob(ext):
                rows.append({'filename': str(fp), 'class': cls})
    df = pd.DataFrame(rows)
    df_tv, df_test = train_test_split(
        df, test_size=0.10, stratify=df['class'], random_state=seed)
    df_train, df_val = train_test_split(
        df_tv, test_size=2/9, stratify=df_tv['class'], random_state=seed)
    return (df_train.reset_index(drop=True),
            df_val.reset_index(drop=True),
            df_test.reset_index(drop=True))


def load_image_from_path(path: str) -> np.ndarray:
    """Load and normalise a single image to float32 [0,1]."""
    from PIL import Image as PILImage
    img = PILImage.open(path).convert('RGB').resize(IMG_SIZE)
    return np.array(img, dtype=np.float32) / 255.0


def load_test_samples(data_path: str, n_samples: int, seed: int = 42):
    """Return (images, labels, filenames) from the 10% test split."""
    from tensorflow.keras.preprocessing.image import ImageDataGenerator
    _, _, df_test = make_file_splits(data_path, seed)
    df_sample = df_test.sample(min(n_samples, len(df_test)), random_state=seed)

    datagen = ImageDataGenerator(rescale=1.0 / 255)
    gen = datagen.flow_from_dataframe(
        df_sample,
        x_col='filename', y_col='class',
        target_size=IMG_SIZE, batch_size=1,
        classes=CLASS_NAMES, class_mode='sparse',
        shuffle=False,
    )
    images, labels = [], []
    for _ in range(len(gen)):
        x, y = next(gen)
        images.append(x[0])
        labels.append(int(y[0]))

    print(f'Loaded {len(images)} test samples.')
    return np.array(images), np.array(labels), list(df_sample['filename'].values)


def load_background_samples(data_path: str, n: int = 20, seed: int = 42):
    """Load n background images (from training split) for SHAP baseline."""
    import random as rnd
    rnd.seed(seed)
    np.random.seed(seed)
    _, _, df_test = make_file_splits(data_path, seed)
    df_train, _, _ = make_file_splits(data_path, seed)

    all_bg = []
    per_class = max(1, n // NUM_CLASSES)
    for cls in CLASS_NAMES:
        subset = df_train[df_train['class'] == cls]
        chosen = subset.sample(min(per_class, len(subset)), random_state=seed)
        for fp in chosen['filename']:
            all_bg.append(load_image_from_path(fp))

    bg = np.array(all_bg[:n], dtype=np.float32)
    print(f'Background samples: {len(bg)}')
    return bg


def pick_representative_samples(images, labels, model, n_per_class: int = 1):
    """
    For each class, pick the n_per_class correctly-classified images
    with highest confidence. Returns (rep_images, rep_labels, rep_indices).
    """
    rep_imgs, rep_lbls, rep_idx = [], [], []
    for cls_idx in range(NUM_CLASSES):
        cls_mask = np.where(labels == cls_idx)[0]
        if len(cls_mask) == 0:
            continue
        cls_images = images[cls_mask]
        preds = model(cls_images, training=False).numpy()
        correct_mask = np.argmax(preds, axis=1) == cls_idx
        confs = preds[:, cls_idx]

        correct_indices = cls_mask[correct_mask]
        correct_confs   = confs[correct_mask]

        if len(correct_indices) == 0:
            correct_indices = cls_mask
            correct_confs   = confs

        order   = np.argsort(-correct_confs)
        chosen  = correct_indices[order[:n_per_class]]
        for i in chosen:
            rep_imgs.append(images[i])
            rep_lbls.append(cls_idx)
            rep_idx.append(i)

    return np.array(rep_imgs), np.array(rep_lbls), rep_idx


# ══════════════════════════════════════════════════════════════════════
# GRAD-CAM
# ══════════════════════════════════════════════════════════════════════

def find_last_spatial_layer(model):
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
    chosen = (gfp_candidates or lfp_candidates or other_candidates)[-1]
    print(f'GradCAM target: {chosen.name}  {chosen.output.shape}')
    return chosen.name


def build_gradcam_model(model, target_layer_name: str):
    feat = model.get_layer(target_layer_name)
    return keras.Model(inputs=model.inputs,
                       outputs=[feat.output, model.output])


def compute_gradcam(gradcam_model, img_batch: np.ndarray, class_idx: int):
    img_t = tf.cast(img_batch, tf.float32)
    with tf.GradientTape() as tape:
        tape.watch(img_t)
        feat_maps, logits = gradcam_model(img_t, training=False)
        tape.watch(feat_maps)
        score = logits[:, class_idx]
    grads       = tape.gradient(score, feat_maps)
    pooled      = tf.reduce_mean(grads, axis=(0, 1, 2))
    heatmap     = feat_maps[0] @ pooled[..., tf.newaxis]
    heatmap     = tf.squeeze(tf.nn.relu(heatmap)).numpy()
    mx = heatmap.max()
    if mx > 0:
        heatmap /= mx
    return heatmap


# ══════════════════════════════════════════════════════════════════════
# SHAP  (GradientExplainer)
# ══════════════════════════════════════════════════════════════════════

def compute_shap(model, img_array: np.ndarray,
                 background: np.ndarray, class_idx: int):
    """
    Returns (H, W, 3) SHAP values or None if GradientExplainer fails.
    """
    try:
        import shap
        img_batch = img_array[np.newaxis].astype(np.float32)
        explainer  = shap.GradientExplainer(model, background)
        shap_vals  = explainer.shap_values(img_batch)

        if isinstance(shap_vals, list):
            sv = shap_vals[min(class_idx, len(shap_vals) - 1)][0]
        else:
            sv = np.array(shap_vals)
            if sv.ndim == 5:
                sv = sv[0, :, :, :, min(class_idx, sv.shape[-1] - 1)]
            elif sv.ndim == 4:
                sv = sv[0]

        sv = np.squeeze(sv)
        if sv.ndim == 2:
            sv = np.stack([sv] * 3, axis=-1)
        return sv.astype(np.float32)

    except Exception as e:
        print(f'  [SHAP] GradientExplainer failed: {e}')
        return None


def shap_to_heatmap(shap_values: np.ndarray) -> np.ndarray:
    return np.mean(np.abs(shap_values), axis=2)


# ══════════════════════════════════════════════════════════════════════
# INTEGRATED GRADIENTS
# ══════════════════════════════════════════════════════════════════════

def compute_integrated_gradients(model, img_array: np.ndarray,
                                  class_idx: int, n_steps: int = 50,
                                  baseline: np.ndarray = None):
    """
    Integrated Gradients with black-image baseline.
    Returns (H, W, 3) attribution map.
    """
    if baseline is None:
        baseline = np.zeros_like(img_array)

    alphas      = np.linspace(0.0, 1.0, n_steps + 1, dtype=np.float32)
    interpolated = np.array([
        baseline + alpha * (img_array - baseline) for alpha in alphas
    ], dtype=np.float32)

    batch_size = 10
    grads_list = []
    for start in range(0, len(interpolated), batch_size):
        batch = tf.constant(interpolated[start:start + batch_size])
        with tf.GradientTape() as tape:
            tape.watch(batch)
            preds = model(batch, training=False)
            score = preds[:, class_idx]
        grads = tape.gradient(score, batch)
        grads_list.append(grads.numpy())

    all_grads = np.concatenate(grads_list, axis=0)
    avg_grads = (all_grads[:-1] + all_grads[1:]) / 2.0
    mean_grads = np.mean(avg_grads, axis=0)
    ig = (img_array - baseline) * mean_grads
    return ig.astype(np.float32)


def ig_to_heatmap(ig: np.ndarray) -> np.ndarray:
    hm = np.sum(np.abs(ig), axis=2)
    mx = hm.max()
    if mx > 0:
        hm /= mx
    return hm


# ══════════════════════════════════════════════════════════════════════
# OCCLUSION ANALYSIS
# ══════════════════════════════════════════════════════════════════════

def compute_occlusion(model, img_array: np.ndarray,
                       class_idx: int,
                       window: int = 28, stride: int = 7,
                       occlude_val: float = 0.0):
    """
    Sliding-window occlusion map. Returns (H, W) sensitivity map.
    """
    H, W = img_array.shape[:2]
    baseline_pred = model(img_array[np.newaxis], training=False).numpy()[0, class_idx]

    n_y = (H - window) // stride + 1
    n_x = (W - window) // stride + 1

    # Batch all occluded images for efficiency
    occluded_imgs = []
    positions     = []
    for iy in range(n_y):
        for ix in range(n_x):
            y0, y1 = iy * stride, iy * stride + window
            x0, x1 = ix * stride, ix * stride + window
            occ = img_array.copy()
            occ[y0:y1, x0:x1] = occlude_val
            occluded_imgs.append(occ)
            positions.append((y0, y1, x0, x1))

    occluded_imgs = np.array(occluded_imgs, dtype=np.float32)
    batch_size    = 32
    drops = []
    for start in range(0, len(occluded_imgs), batch_size):
        batch = occluded_imgs[start:start + batch_size]
        preds = model(batch, training=False).numpy()[:, class_idx]
        drops.extend(baseline_pred - preds)

    # Build sensitivity map
    counts = np.zeros((H, W), dtype=np.float32)
    smap   = np.zeros((H, W), dtype=np.float32)
    for drop, (y0, y1, x0, x1) in zip(drops, positions):
        smap[y0:y1, x0:x1]   += drop
        counts[y0:y1, x0:x1] += 1.0

    counts = np.where(counts == 0, 1.0, counts)
    smap  /= counts
    smap   = np.clip(smap, 0, None)
    mx = smap.max()
    if mx > 0:
        smap /= mx
    return smap.astype(np.float32)


# ══════════════════════════════════════════════════════════════════════
# OVERLAY HELPERS
# ══════════════════════════════════════════════════════════════════════

def overlay_heatmap(img_rgb: np.ndarray, heatmap: np.ndarray,
                    alpha: float = 0.45):
    h, w    = img_rgb.shape[:2]
    hm_u8   = np.uint8(255 * np.clip(heatmap, 0, 1))
    hm_r    = cv2.resize(hm_u8, (w, h))
    hm_col  = cv2.applyColorMap(hm_r, cv2.COLORMAP_JET)
    hm_col  = cv2.cvtColor(hm_col, cv2.COLOR_BGR2RGB)
    img_u8  = np.uint8(255 * np.clip(img_rgb, 0, 1))
    return cv2.addWeighted(img_u8, 1 - alpha, hm_col, alpha, 0)


# ══════════════════════════════════════════════════════════════════════
# COMPARISON FIGURE  (1 row per class, 5 columns)
# ══════════════════════════════════════════════════════════════════════

def plot_comparison(rep_images, rep_labels, pred_labels, confs,
                    gradcam_hms, shap_vals_list,
                    ig_attrs, occ_maps, output_dir: Path):
    """
    Grid: rows = classes, cols = [Original | GradCAM | SHAP | IG | Occlusion]
    SHAP column shows 'N/A' if shap_vals is None.
    """
    n_rows   = len(rep_images)
    col_titles = ['Original', 'Grad-CAM', 'SHAP', 'Integrated\nGradients', 'Occlusion\nAnalysis']
    n_cols   = len(col_titles)

    fig, axes = plt.subplots(n_rows, n_cols,
                              figsize=(n_cols * 3, n_rows * 3.2))
    if n_rows == 1:
        axes = axes[np.newaxis, :]

    for col, title in enumerate(col_titles):
        axes[0, col].set_title(title, fontsize=10, fontweight='bold', pad=6)

    for row in range(n_rows):
        img      = rep_images[row]
        cls_idx  = int(rep_labels[row])
        cls_name = CLASS_NAMES[cls_idx]
        pred     = CLASS_NAMES[int(pred_labels[row])]
        conf     = confs[row]
        correct  = cls_idx == int(pred_labels[row])
        col_edge = CLASS_COLORS.get(cls_name, 'black')

        # Row label
        axes[row, 0].set_ylabel(
            CLASS_LABELS[cls_name], fontsize=10, fontweight='bold',
            color=col_edge, rotation=90, labelpad=6)

        # Col 0: Original
        axes[row, 0].imshow(img)
        axes[row, 0].axis('off')
        mark = '✓' if correct else '✗'
        axes[row, 0].set_xlabel(
            f'{mark} pred:{pred}\n({conf:.2f})',
            fontsize=7, color='green' if correct else 'red')

        # Col 1: GradCAM overlay
        hm = cv2.resize(gradcam_hms[row],
                        (img.shape[1], img.shape[0]))
        axes[row, 1].imshow(overlay_heatmap(img, hm))
        axes[row, 1].axis('off')

        # Col 2: SHAP
        sv = shap_vals_list[row]
        if sv is not None:
            shap_hm = shap_to_heatmap(sv)
            shap_hm_r = cv2.resize(shap_hm,
                                    (img.shape[1], img.shape[0]))
            mx = shap_hm_r.max()
            if mx > 0:
                shap_hm_r /= mx
            axes[row, 2].imshow(overlay_heatmap(img, shap_hm_r))
        else:
            axes[row, 2].set_facecolor('#f0f0f0')
            axes[row, 2].text(
                0.5, 0.5, 'N/A\n(DiNA incompatible)',
                ha='center', va='center',
                transform=axes[row, 2].transAxes,
                fontsize=9, color='#666666')
        axes[row, 2].axis('off')

        # Col 3: Integrated Gradients
        ig_hm   = ig_to_heatmap(ig_attrs[row])
        ig_hm_r = cv2.resize(ig_hm, (img.shape[1], img.shape[0]))
        axes[row, 3].imshow(overlay_heatmap(img, ig_hm_r))
        axes[row, 3].axis('off')

        # Col 4: Occlusion
        occ_r = cv2.resize(occ_maps[row],
                            (img.shape[1], img.shape[0]))
        axes[row, 4].imshow(overlay_heatmap(img, occ_r))
        axes[row, 4].axis('off')

    fig.suptitle('MedViT V2 — XAI Comparison\n'
                 'Grad-CAM | SHAP | Integrated Gradients | Occlusion',
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    path = output_dir / 'xai_comparison_grid.png'
    fig.savefig(path, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f'Comparison grid saved: {path}')


def plot_per_method_montage(rep_images, rep_labels, pred_labels, confs,
                             heatmaps, method_name: str,
                             output_dir: Path):
    """4-panel montage: [Original | Heatmap | Overlay] for each class."""
    n     = len(rep_images)
    fig, axes = plt.subplots(n, 3, figsize=(10, n * 3.2))
    if n == 1:
        axes = axes[np.newaxis, :]

    for row in range(n):
        img      = rep_images[row]
        cls_idx  = int(rep_labels[row])
        cls_name = CLASS_NAMES[cls_idx]
        pred     = CLASS_NAMES[int(pred_labels[row])]
        conf     = confs[row]
        correct  = cls_idx == int(pred_labels[row])

        hm = heatmaps[row]
        if hm is None:
            hm = np.zeros(img.shape[:2], dtype=np.float32)
        hm_r = cv2.resize(hm, (img.shape[1], img.shape[0]))

        axes[row, 0].imshow(img)
        axes[row, 0].axis('off')
        axes[row, 0].set_title(CLASS_LABELS[cls_name], fontsize=9,
                                fontweight='bold',
                                color=CLASS_COLORS.get(cls_name, 'black'))

        axes[row, 1].imshow(hm_r, cmap='jet', vmin=0, vmax=1)
        axes[row, 1].axis('off')
        axes[row, 1].set_title('Heatmap', fontsize=9)

        axes[row, 2].imshow(overlay_heatmap(img, hm_r))
        axes[row, 2].axis('off')
        mark = '✓' if correct else '✗'
        axes[row, 2].set_title(
            f'{mark} {pred} ({conf:.2f})', fontsize=9,
            color='green' if correct else 'red')

    fig.suptitle(f'MedViT V2 — {method_name}',
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    path = output_dir / f'{method_name.lower().replace(" ", "_")}_montage.png'
    fig.savefig(path, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f'{method_name} montage saved: {path}')


# ══════════════════════════════════════════════════════════════════════
# FULL XAI PIPELINE
# ══════════════════════════════════════════════════════════════════════

def run_full_xai(args):
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    print('=' * 65)
    print('  MedViT V2 — Full XAI Suite')
    print('  Methods: Grad-CAM | SHAP | Integrated Gradients | Occlusion')
    print('=' * 65)

    # ── 1. Load model ───────────────────────────────────────────────
    print(f'\nLoading: {args.model}')
    model = keras.models.load_model(
        args.model, custom_objects=CUSTOM_OBJECTS, compile=False)
    print('Model loaded.')

    # ── 2. Load test samples ────────────────────────────────────────
    images, labels, fnames = load_test_samples(
        args.data, n_samples=args.n_samples, seed=args.seed)

    # ── 3. Pick representative images (1 per class, best confidence) ─
    print('\nSelecting representative images per class ...')
    rep_imgs, rep_lbls, rep_idx = pick_representative_samples(
        images, labels, model, n_per_class=1)

    preds_rep  = model(rep_imgs, training=False).numpy()
    pred_lbls  = np.argmax(preds_rep, axis=1)
    confs      = [float(preds_rep[i, pred_lbls[i]])
                  for i in range(len(rep_imgs))]

    for i, (cls, pred, conf) in enumerate(
            zip(rep_lbls, pred_lbls, confs)):
        print(f'  Class {CLASS_NAMES[cls]:12s} → '
              f'pred {CLASS_NAMES[pred]:12s}  conf={conf:.3f}')

    # ── 4. Background for SHAP ──────────────────────────────────────
    print('\nLoading background images for SHAP ...')
    background = load_background_samples(
        args.data, n=args.n_background, seed=args.seed)

    # ── 5. Grad-CAM ─────────────────────────────────────────────────
    print('\n[1/4] Grad-CAM ...')
    target_layer = (find_last_spatial_layer(model)
                    if args.target_layer == 'auto'
                    else args.target_layer)
    gradcam_model = build_gradcam_model(model, target_layer)

    gradcam_hms = []
    for img, cls in zip(rep_imgs, pred_lbls):
        hm = compute_gradcam(gradcam_model, img[np.newaxis], int(cls))
        gradcam_hms.append(hm)
    print('  Grad-CAM done.')

    # ── 6. SHAP ─────────────────────────────────────────────────────
    print('\n[2/4] SHAP (GradientExplainer) ...')
    shap_succeeded = False
    shap_vals_list = []
    for i, (img, cls) in enumerate(zip(rep_imgs, pred_lbls)):
        print(f'  SHAP [{i+1}/{len(rep_imgs)}] class={CLASS_NAMES[cls]} ...')
        sv = compute_shap(model, img, background, int(cls))
        shap_vals_list.append(sv)
        if sv is not None:
            shap_succeeded = True

    if shap_succeeded:
        print('  SHAP: at least one class succeeded.')
    else:
        print('  SHAP: all classes failed — DiNA layers likely incompatible.')

    # ── 7. Integrated Gradients ─────────────────────────────────────
    print(f'\n[3/4] Integrated Gradients ({args.ig_steps} steps) ...')
    ig_attrs = []
    for i, (img, cls) in enumerate(zip(rep_imgs, pred_lbls)):
        print(f'  IG [{i+1}/{len(rep_imgs)}] class={CLASS_NAMES[cls]} ...')
        ig = compute_integrated_gradients(
            model, img, int(cls), n_steps=args.ig_steps)
        ig_attrs.append(ig)
    print('  Integrated Gradients done.')

    # ── 8. Occlusion Analysis ────────────────────────────────────────
    print(f'\n[4/4] Occlusion (window={args.occlusion_window}, '
          f'stride={args.occlusion_stride}) ...')
    occ_maps = []
    for i, (img, cls) in enumerate(zip(rep_imgs, pred_lbls)):
        print(f'  Occlusion [{i+1}/{len(rep_imgs)}] '
              f'class={CLASS_NAMES[cls]} ...')
        occ = compute_occlusion(
            model, img, int(cls),
            window=args.occlusion_window,
            stride=args.occlusion_stride)
        occ_maps.append(occ)
    print('  Occlusion done.')

    # ── 9. Visualisations ────────────────────────────────────────────
    print('\nGenerating figures ...')

    # Main comparison grid
    plot_comparison(rep_imgs, rep_lbls, pred_lbls, confs,
                    gradcam_hms, shap_vals_list,
                    ig_attrs, occ_maps, output_dir)

    # Per-method montages
    plot_per_method_montage(rep_imgs, rep_lbls, pred_lbls, confs,
                             gradcam_hms, 'Grad-CAM', output_dir)

    ig_hms = [ig_to_heatmap(ig) for ig in ig_attrs]
    plot_per_method_montage(rep_imgs, rep_lbls, pred_lbls, confs,
                             ig_hms, 'Integrated Gradients', output_dir)

    plot_per_method_montage(rep_imgs, rep_lbls, pred_lbls, confs,
                             occ_maps, 'Occlusion Analysis', output_dir)

    if shap_succeeded:
        shap_hms = [shap_to_heatmap(sv) if sv is not None
                    else None for sv in shap_vals_list]
        # Normalise each heatmap to [0,1]
        shap_hms_norm = []
        for hm in shap_hms:
            if hm is not None:
                mx = hm.max()
                shap_hms_norm.append(hm / mx if mx > 0 else hm)
            else:
                shap_hms_norm.append(None)
        plot_per_method_montage(rep_imgs, rep_lbls, pred_lbls, confs,
                                 shap_hms_norm, 'SHAP', output_dir)

    # ── 10. Save JSON summary ────────────────────────────────────────
    summary = {
        'model_path':       str(args.model),
        'target_layer':     target_layer,
        'n_samples':        int(args.n_samples),
        'n_representative': int(len(rep_imgs)),
        'ig_steps':         int(args.ig_steps),
        'occlusion_window': int(args.occlusion_window),
        'occlusion_stride': int(args.occlusion_stride),
        'n_background':     int(args.n_background),
        'shap_succeeded':   shap_succeeded,
        'per_class': {},
    }

    for i, cls_idx in enumerate(rep_lbls):
        cls_name = CLASS_NAMES[int(cls_idx)]
        ig_hm    = ig_to_heatmap(ig_attrs[i])
        occ_hm   = occ_maps[i]
        gc_hm    = gradcam_hms[i]
        sv       = shap_vals_list[i]

        entry = {
            'true_class':   cls_name,
            'pred_class':   CLASS_NAMES[int(pred_lbls[i])],
            'confidence':   confs[i],
            'correct':      bool(int(cls_idx) == int(pred_lbls[i])),
            'gradcam_mean_activation': float(gc_hm.mean()),
            'ig_mean_activation':      float(ig_hm.mean()),
            'occlusion_mean':          float(occ_hm.mean()),
            'occlusion_peak':          float(occ_hm.max()),
            'shap_available':          sv is not None,
        }
        if sv is not None:
            shap_hm = shap_to_heatmap(sv)
            entry['shap_mean_activation'] = float(shap_hm.mean())

        summary['per_class'][cls_name] = entry

    with open(output_dir / 'xai_full_stats.json', 'w') as f:
        json.dump(summary, f, indent=2)

    print(f'\nAll XAI results saved to: {output_dir}')
    print(f"  SHAP status: {'SUCCESS' if shap_succeeded else 'FAILED (DiNA incompatible — expected)'}")
    return summary


# ══════════════════════════════════════════════════════════════════════
# CLI
# ══════════════════════════════════════════════════════════════════════

def parse_args():
    p = argparse.ArgumentParser(
        description='MedViT V2 — Full XAI Suite')
    p.add_argument('--model',      required=True,
                   help='Path to .keras model file')
    p.add_argument('--data',       default=DATA_PATH,
                   help='Path to 4-class dataset directory')
    p.add_argument('--output',     default='medvit_v2_xai_full_results')
    p.add_argument('--n_samples',  type=int, default=40,
                   help='Images to draw from test split')
    p.add_argument('--target_layer', default='auto',
                   help='GradCAM layer name (or "auto")')
    p.add_argument('--n_background', type=int, default=20,
                   help='Background images for SHAP')
    p.add_argument('--ig_steps',   type=int, default=50,
                   help='Interpolation steps for Integrated Gradients')
    p.add_argument('--occlusion_window', type=int, default=28,
                   help='Occlusion patch size in pixels')
    p.add_argument('--occlusion_stride', type=int, default=7,
                   help='Occlusion stride in pixels')
    p.add_argument('--seed',       type=int, default=42)
    return p.parse_args()


if __name__ == '__main__':
    run_full_xai(parse_args())
