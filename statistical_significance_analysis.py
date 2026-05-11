#!/usr/bin/env python3
"""
Unified Statistical Significance Analysis
==========================================

Comprehensive statistical comparison across all brain tumor classification models.

Tests performed:
    1. Friedman Test            - overall comparison across all models
    2. Nemenyi Post-hoc Test    - pairwise comparison after significant Friedman
    3. Wilcoxon Signed-Rank     - pairwise with Bonferroni correction
    4. Critical Difference (CD) Diagram - visualization

Models compared (7 total):
    1. Binary CNN
    2. 4-Class CNN
    3. 3-Class CNN
    4. 1-vs-1 Binary Ensemble
    5. ResNet-50 Dual Branch
    6. EfficientNet-B3 Dual Branch
    7. Multi-Dual System (3-Class + 4-Class branches)

Author: Yannis Balasis
Date: March 2026
"""

import os
import sys
import json
import argparse
import warnings
import itertools
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns

import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from scipy.stats import wilcoxon, friedmanchisquare
from scipy.stats import rankdata
from itertools import combinations

warnings.filterwarnings('ignore')

# ─────────────────────────────────────────────
# Optional: scikit-posthocs for Nemenyi
# ─────────────────────────────────────────────
try:
    import scikit_posthocs as sp
    HAVE_POSTHOCS = True
except ImportError:
    HAVE_POSTHOCS = False
    print("Warning: scikit-posthocs not found. Installing...")
    os.system("pip install scikit-posthocs --break-system-packages -q")
    try:
        import scikit_posthocs as sp
        HAVE_POSTHOCS = True
    except ImportError:
        HAVE_POSTHOCS = False
        print("scikit-posthocs could not be installed. Nemenyi will use manual implementation.")


# ══════════════════════════════════════════════════════════════════════════════
# CONFIGURATION
# ══════════════════════════════════════════════════════════════════════════════

CLASS_NAMES_4 = ['glioma', 'meningioma', 'no_tumor', 'pituitary']
CLASS_NAMES_3 = ['glioma', 'meningioma', 'pituitary']
CLASS_NAMES_BINARY = ['no_tumor', 'tumor']

# 1-vs-1 ensemble pair definitions
ENSEMBLE_PAIRS = [
    ('glioma',      'meningioma', 'best_glioma_vs_meningioma_model.h5'),
    ('glioma',      'pituitary',  'best_glioma_vs_pituitary_model.h5'),
    ('meningioma',  'pituitary',  'best_meningioma_vs_pituitary_model.h5'),
]

ALPHA = 0.05  # significance level


# ══════════════════════════════════════════════════════════════════════════════
# DATA LOADING
# ══════════════════════════════════════════════════════════════════════════════

def load_test_data(data_path: str, seed: int = 42):
    """
    Load the held-out test split (10%) from the 4-class dataset.
    Returns images, true 4-class labels, and filenames.
    """
    datagen = ImageDataGenerator(rescale=1.0 / 255.0, validation_split=0.1)

    generator = datagen.flow_from_directory(
        data_path,
        target_size=(224, 224),
        batch_size=1,
        class_mode='sparse',
        classes=CLASS_NAMES_4,
        subset='validation',
        shuffle=False,
        seed=seed
    )

    images, labels = [], []
    for i in range(len(generator)):
        x, y = generator[i]
        images.append(x[0])
        labels.append(int(y[0]))

    images = np.array(images)
    labels = np.array(labels, dtype=np.int32)

    print(f"Test set loaded: {len(images)} samples")
    unique, counts = np.unique(labels, return_counts=True)
    for cls, cnt in zip(unique, counts):
        print(f"   {CLASS_NAMES_4[cls]}: {cnt}")

    return images, labels


# ══════════════════════════════════════════════════════════════════════════════
# MODEL LOADERS & PREDICTORS
# ══════════════════════════════════════════════════════════════════════════════

def load_standard_model(model_path: str):
    """Load a standard Keras .h5 model."""
    model = tf.keras.models.load_model(model_path, compile=False)
    print(f"Loaded: {model_path}")
    return model


def predict_binary_cnn(model, images, labels):
    """
    Binary CNN: predicts tumor (1) vs no_tumor (0).
    Converts to 4-class correct/incorrect on the full test set.
    Correct = model correctly identifies tumor vs no_tumor.
    """
    probs = model.predict(images, verbose=0)

    # Handle both sigmoid (shape N,1) and softmax (shape N,2) outputs
    if probs.shape[-1] == 1:
        pred_binary = (probs[:, 0] > 0.5).astype(int)
    else:
        pred_binary = np.argmax(probs, axis=1)

    # true_binary: 0=no_tumor (class 2 in 4-class), 1=tumor (all others)
    true_binary = (labels != 2).astype(int)

    correct = (pred_binary == true_binary).astype(int)
    return correct


def predict_4class_cnn(model, images, labels):
    """4-Class CNN: direct 4-class prediction."""
    probs = model.predict(images, verbose=0)
    preds = np.argmax(probs, axis=1)
    correct = (preds == labels).astype(int)
    return correct


def predict_3class_cnn(model, images, labels):
    """
    3-Class CNN: evaluated only on tumor samples (no_tumor excluded).
    Returns correct array aligned to the FULL test set:
    - tumor samples: 1 if correctly classified, 0 otherwise
    - no_tumor samples: marked as correct (1) because the model is not
      responsible for that class; this is noted in the report.

    Scientific note: we use the tumor-only subset for the actual
    Wilcoxon/Friedman computation (see statistical_analysis()).
    """
    # map 4-class labels to 3-class (no_tumor -> -1)
    map_4_to_3 = {0: 0, 1: 1, 2: -1, 3: 2}

    tumor_indices = [i for i, lbl in enumerate(labels) if lbl != 2]
    tumor_images  = images[tumor_indices]
    tumor_labels_3 = np.array([map_4_to_3[labels[i]] for i in tumor_indices])

    probs = model.predict(tumor_images, verbose=0)
    preds = np.argmax(probs, axis=1)

    correct_tumor = (preds == tumor_labels_3).astype(int)

    # Build full-length array; no_tumor positions set to -1 (excluded marker)
    correct_full = np.full(len(labels), -1, dtype=np.int32)
    for idx, orig_idx in enumerate(tumor_indices):
        correct_full[orig_idx] = correct_tumor[idx]

    return correct_full


def predict_ensemble_1vs1(pair_models: dict, images, labels):
    """
    1-vs-1 Binary Ensemble with correct sigmoid voting.
    Each model: sigmoid output where 1.0 = cls_b, 0.0 = cls_a.
    """
    cls_to_idx = {name: i for i, name in enumerate(CLASS_NAMES_4)}

    # class_indices from training: cls_a=0, cls_b=1
    # sigmoid > 0.5 → cls_b wins, sigmoid < 0.5 → cls_a wins
    pair_definitions = [
        ('glioma',     'meningioma'),  # 0=glioma,    1=meningioma
        ('glioma',     'pituitary'),   # 0=glioma,    1=pituitary
        ('meningioma', 'pituitary'),   # 0=meningioma, 1=pituitary
    ]

    vote_matrix = np.zeros((len(images), 4), dtype=np.float32)

    for (cls_a, cls_b, fname), model in pair_models.items():
        probs = model.predict(images, verbose=0)  # shape (N, 1)
        prob_b = probs[:, 0]   # probability of cls_b
        prob_a = 1.0 - prob_b  # probability of cls_a

        idx_a = cls_to_idx[cls_a]
        idx_b = cls_to_idx[cls_b]

        vote_matrix[:, idx_a] += prob_a
        vote_matrix[:, idx_b] += prob_b

    # no_tumor: not covered by any pair → set to 0
    # if all tumor votes are low (< 0.5 each = total < 1.5), predict no_tumor
    tumor_cols   = [cls_to_idx[c] for c in ['glioma', 'meningioma', 'pituitary']]
    no_tumor_col = cls_to_idx['no_tumor']

    max_tumor_vote = vote_matrix[:, tumor_cols].max(axis=1)
    # Each class gets max 2 votes (appears in 2 pairs).
    # If best tumor vote < 0.8, lean towards no_tumor
    vote_matrix[:, no_tumor_col] = np.where(max_tumor_vote < 0.8, 1.5, 0.0)

    preds   = np.argmax(vote_matrix, axis=1)
    correct = (preds == labels).astype(int)
    return correct


def predict_resnet_dual(model, images, labels):
    """ResNet-50 Dual Branch: use fusion_output."""
    predictions = model.predict(images, verbose=0)

    if isinstance(predictions, dict):
        fusion_probs = predictions.get('fusion_output',
                       predictions.get('fusion',
                       list(predictions.values())[-1]))
    else:
        fusion_probs = predictions[-1] if isinstance(predictions, list) else predictions

    preds = np.argmax(fusion_probs, axis=1)
    correct = (preds == labels).astype(int)
    return correct


def predict_efnet_dual(model, images, labels):
    """EfficientNet-B3 Dual Branch: handles both single and multi-output."""
    predictions = model.predict(images, verbose=0)

    if isinstance(predictions, dict):
        # Multi-output: try fusion_output first, fallback to classifier_output
        probs = predictions.get('fusion_output',
                predictions.get('classifier_output',
                list(predictions.values())[-1]))
    elif isinstance(predictions, list):
        probs = predictions[-1]
    else:
        # Single output array
        probs = predictions

    preds = np.argmax(probs, axis=1)
    correct = (preds == labels).astype(int)
    return correct


def predict_multi_dual(model, images, labels):
    """Multi-Dual System (3-class + 4-class): use fusion_output."""
    return predict_resnet_dual(model, images, labels)


# ══════════════════════════════════════════════════════════════════════════════
# STATISTICAL TESTS
# ══════════════════════════════════════════════════════════════════════════════

def prepare_common_subset(correct_vectors: dict):
    """
    For fair comparison, restrict to samples where ALL models provide
    a valid prediction (i.e., exclude -1 markers from 3-class model).
    Returns filtered vectors and the common indices used.
    """
    # Find indices where every model has a valid prediction
    lengths = [len(v) for v in correct_vectors.values()]
    assert len(set(lengths)) == 1, "All correct vectors must have the same length"

    n = lengths[0]
    valid_mask = np.ones(n, dtype=bool)

    for v in correct_vectors.values():
        valid_mask &= (v != -1)

    common_indices = np.where(valid_mask)[0]
    filtered = {name: v[common_indices] for name, v in correct_vectors.items()}

    print(f"\nCommon evaluation subset: {len(common_indices)} samples "
          f"(excluded {n - len(common_indices)} no_tumor samples "
          f"due to 3-class model scope)")

    return filtered, common_indices


def run_friedman_test(correct_vectors: dict):
    """
    Friedman Test: non-parametric test for differences across k related groups.
    Input: dict of {model_name: binary_correct_array}
    """
    model_names = list(correct_vectors.keys())
    data = [correct_vectors[name].astype(float) for name in model_names]

    stat, p_value = friedmanchisquare(*data)

    print(f"\nFriedman Test:")
    print(f"   Chi-squared statistic: {stat:.4f}")
    print(f"   p-value: {p_value:.6f}")
    print(f"   Significant (alpha={ALPHA}): {'YES' % () if p_value < ALPHA else 'NO'}")

    return {
        'statistic': float(stat),
        'p_value': float(p_value),
        'significant': bool(p_value < ALPHA),
        'model_names': model_names
    }


def run_nemenyi_test(correct_vectors: dict):
    """
    Nemenyi Post-hoc Test after significant Friedman result.
    Returns a pairwise p-value matrix.
    """
    model_names = list(correct_vectors.keys())
    n_models = len(model_names)

    if HAVE_POSTHOCS:
        # Build (n_samples x n_models) matrix
        data_matrix = np.column_stack(
            [correct_vectors[name].astype(float) for name in model_names]
        )
        df = pd.DataFrame(data_matrix, columns=model_names)
        p_matrix = sp.posthoc_nemenyi_friedman(df)
    else:
        # Manual Nemenyi: based on average ranks
        p_matrix = _manual_nemenyi(correct_vectors, model_names)

    print(f"\nNemenyi Post-hoc Test (pairwise p-values):")
    print(pd.DataFrame(p_matrix,
                       index=model_names,
                       columns=model_names).round(4).to_string())

    return p_matrix, model_names


def _manual_nemenyi(correct_vectors: dict, model_names: list):
    """
    Manual Nemenyi approximation using average ranks and critical difference.
    Returns symmetric p-value matrix (approximate).
    """
    from scipy.stats import norm

    n_models = len(model_names)
    data_matrix = np.column_stack(
        [correct_vectors[name].astype(float) for name in model_names]
    )
    n_samples = data_matrix.shape[0]

    # Compute average ranks across samples
    ranks = np.array([rankdata(row) for row in data_matrix])
    avg_ranks = ranks.mean(axis=0)

    # Nemenyi critical difference statistic
    # z = (|R_i - R_j|) / sqrt(k(k+1) / (6N))
    denom = np.sqrt(n_models * (n_models + 1) / (6 * n_samples))

    p_matrix = pd.DataFrame(
        np.ones((n_models, n_models)),
        index=model_names, columns=model_names
    )

    for i, j in combinations(range(n_models), 2):
        z = abs(avg_ranks[i] - avg_ranks[j]) / denom
        # Two-tailed p-value approximation
        p = 2 * (1 - norm.cdf(z))
        p = min(p, 1.0)
        p_matrix.iloc[i, j] = p
        p_matrix.iloc[j, i] = p

    return p_matrix


def run_wilcoxon_bonferroni(correct_vectors: dict):
    """
    Pairwise Wilcoxon Signed-Rank Tests with Bonferroni correction.
    """
    model_names = list(correct_vectors.keys())
    pairs = list(combinations(range(len(model_names)), 2))
    n_comparisons = len(pairs)

    results = []

    for i, j in pairs:
        name_i = model_names[i]
        name_j = model_names[j]
        vec_i  = correct_vectors[name_i].astype(float)
        vec_j  = correct_vectors[name_j].astype(float)

        # Wilcoxon requires differences; skip if identical
        diff = vec_i - vec_j
        if np.all(diff == 0):
            stat, p_raw = 0.0, 1.0
        else:
            try:
                stat, p_raw = wilcoxon(vec_i, vec_j, alternative='two-sided',
                                       zero_method='wilcox')
            except ValueError:
                stat, p_raw = 0.0, 1.0

        p_corrected = min(p_raw * n_comparisons, 1.0)  # Bonferroni
        significant  = p_corrected < ALPHA

        acc_i = vec_i.mean()
        acc_j = vec_j.mean()
        better = name_i if acc_i >= acc_j else name_j

        results.append({
            'model_1':       name_i,
            'model_2':       name_j,
            'accuracy_1':    float(acc_i),
            'accuracy_2':    float(acc_j),
            'better_model':  better,
            'statistic':     float(stat),
            'p_raw':         float(p_raw),
            'p_bonferroni':  float(p_corrected),
            'significant':   significant
        })

    print(f"\nWilcoxon Signed-Rank Test with Bonferroni Correction:")
    print(f"   Number of pairwise comparisons: {n_comparisons}")
    print(f"   Bonferroni-adjusted alpha: {ALPHA / n_comparisons:.6f}")
    print(f"   {'Model 1':<30} {'Model 2':<30} {'p_raw':>10} {'p_Bonf':>10} {'Sig':>6}")
    print(f"   {'-'*90}")
    for r in results:
        sig_str = 'YES' if r['significant'] else 'NO'
        print(f"   {r['model_1']:<30} {r['model_2']:<30} "
              f"{r['p_raw']:>10.4f} {r['p_bonferroni']:>10.4f} {sig_str:>6}")

    return results

def compute_cohens_d(correct_vectors: dict):
    """
    Cohen's d effect size for all pairwise model comparisons.
    Interpretation:
        d < 0.2  -> negligible
        d 0.2-0.5 -> small
        d 0.5-0.8 -> medium
        d > 0.8  -> large
    """
    model_names = list(correct_vectors.keys())
    pairs = list(combinations(range(len(model_names)), 2))
    results = []

    for i, j in pairs:
        name_i = model_names[i]
        name_j = model_names[j]
        vec_i  = correct_vectors[name_i].astype(float)
        vec_j  = correct_vectors[name_j].astype(float)

        mean_i, mean_j = vec_i.mean(), vec_j.mean()
        std_i,  std_j  = vec_i.std(ddof=1), vec_j.std(ddof=1)
        n_i,    n_j    = len(vec_i), len(vec_j)

        # Pooled standard deviation
        pooled_std = np.sqrt(
            ((n_i - 1) * std_i**2 + (n_j - 1) * std_j**2) / (n_i + n_j - 2)
        )

        if pooled_std == 0:
            d = 0.0
        else:
            d = abs(mean_i - mean_j) / pooled_std

        # Interpretation
        if d < 0.2:
            interpretation = 'Negligible'
        elif d < 0.5:
            interpretation = 'Small'
        elif d < 0.8:
            interpretation = 'Medium'
        else:
            interpretation = 'Large'

        better = name_i if mean_i >= mean_j else name_j

        results.append({
            'model_1':        name_i,
            'model_2':        name_j,
            'accuracy_1':     float(mean_i),
            'accuracy_2':     float(mean_j),
            'cohens_d':       float(d),
            'interpretation': interpretation,
            'better_model':   better
        })

    print(f"\nCohen's d Effect Size (pairwise):")
    print(f"   {'Model 1':<30} {'Model 2':<30} {'d':>8} {'Effect':>12} {'Better':>20}")
    print(f"   {'-'*105}")
    for r in sorted(results, key=lambda x: x['cohens_d'], reverse=True):
        print(f"   {r['model_1']:<30} {r['model_2']:<30} "
              f"{r['cohens_d']:>8.4f} {r['interpretation']:>12} "
              f"{r['better_model']:>20}")

    return results


def compute_average_ranks(correct_vectors: dict):
    """Compute average ranks across samples for CD diagram."""
    model_names = list(correct_vectors.keys())
    data_matrix = np.column_stack(
        [correct_vectors[name].astype(float) for name in model_names]
    )
    # Lower error = better rank; invert: rank accuracy (higher = better = lower rank number)
    ranks = np.array([rankdata(-row, method='average') for row in data_matrix])
    avg_ranks = ranks.mean(axis=0)

    rank_dict = dict(zip(model_names, avg_ranks))
    print(f"\nAverage Ranks (lower = better):")
    for name, rank in sorted(rank_dict.items(), key=lambda x: x[1]):
        acc = correct_vectors[name].mean()
        print(f"   {name:<35} rank={rank:.3f}  acc={acc:.4f}")

    return rank_dict


def critical_difference(n_models: int, n_samples: int, alpha: float = 0.05):
    """
    Compute Nemenyi Critical Difference (CD).
    CD = q_alpha * sqrt(k(k+1) / 6N)
    q_alpha values (two-tailed) from Demsar 2006:
    """
    # Studentized range statistic / sqrt(2) for k groups
    q_alpha_table = {
        (2,  0.05): 1.960, (3,  0.05): 2.344, (4,  0.05): 2.569,
        (5,  0.05): 2.728, (6,  0.05): 2.850, (7,  0.05): 2.949,
        (8,  0.05): 3.031, (9,  0.05): 3.102, (10, 0.05): 3.164,
        (2,  0.10): 1.645, (3,  0.10): 2.052, (4,  0.10): 2.291,
        (5,  0.10): 2.459, (6,  0.10): 2.589, (7,  0.10): 2.693,
        (8,  0.10): 2.780, (9,  0.10): 2.855, (10, 0.10): 2.920,
    }
    q = q_alpha_table.get((n_models, alpha),
        q_alpha_table.get((n_models, 0.05), 2.850))
    cd = q * np.sqrt(n_models * (n_models + 1) / (6 * n_samples))
    return cd


# ══════════════════════════════════════════════════════════════════════════════
# VISUALIZATIONS
# ══════════════════════════════════════════════════════════════════════════════

def plot_cd_diagram(avg_ranks: dict, cd: float, n_samples: int,
                    friedman_result: dict, output_dir: Path):
    """
    Critical Difference Diagram (Demsar 2006 style).
    Models within CD of each other are connected by a horizontal bar.
    """
    model_names = list(avg_ranks.keys())
    ranks = [avg_ranks[m] for m in model_names]

    # Sort by rank (best = lowest rank = leftmost)
    sorted_pairs = sorted(zip(ranks, model_names))
    sorted_ranks  = [r for r, _ in sorted_pairs]
    sorted_names  = [n for _, n in sorted_pairs]

    n_models = len(model_names)
    fig_height = max(4, n_models * 0.6 + 2)
    fig, ax = plt.subplots(figsize=(12, fig_height))

    # Rank axis
    min_rank = max(1, min(sorted_ranks) - 0.5)
    max_rank = min(n_models, max(sorted_ranks) + 0.5)

    ax.set_xlim(min_rank - 0.3, max_rank + 0.3)
    ax.set_ylim(-0.5, n_models + 1.5)

    # Draw axis line
    ax.axhline(y=n_models + 0.8, color='black', linewidth=2,
               xmin=0.05, xmax=0.95)
    ax.set_xticks(np.arange(np.ceil(min_rank), np.floor(max_rank) + 1))
    ax.tick_params(axis='x', top=True, bottom=False,
                   labeltop=True, labelbottom=False, labelsize=11)
    ax.set_xlabel('Average Rank', fontsize=12, labelpad=2)
    ax.xaxis.set_label_position('top')
    ax.set_yticks([])

    # Draw model lines and labels
    colors = plt.cm.Set2(np.linspace(0, 1, n_models))
    y_positions = {}

    for idx, (rank, name) in enumerate(zip(sorted_ranks, sorted_names)):
        y = n_models - idx - 1
        y_positions[name] = y

        # Vertical line from axis to label
        ax.plot([rank, rank], [y, n_models + 0.8], color=colors[idx],
                linewidth=1.5, linestyle='--', alpha=0.6)

        # Dot on axis
        ax.plot(rank, n_models + 0.8, 'o', color=colors[idx],
                markersize=8, zorder=5)

        # Label
        ha = 'left' if rank <= (min_rank + max_rank) / 2 else 'right'
        x_label = rank - 0.05 if ha == 'right' else rank + 0.05
        ax.text(x_label, y, f'{name}\n(rank={rank:.2f})',
                ha=ha, va='center', fontsize=9,
                bbox=dict(boxstyle='round,pad=0.3', facecolor=colors[idx],
                          alpha=0.15, edgecolor='none'))

    # Draw CD bar (top-left)
    cd_x_start = min_rank
    cd_x_end   = min_rank + cd
    cd_y       = n_models + 0.3

    ax.annotate('', xy=(cd_x_end, cd_y), xytext=(cd_x_start, cd_y),
                arrowprops=dict(arrowstyle='<->', color='red', lw=2))
    ax.text((cd_x_start + cd_x_end) / 2, cd_y + 0.15,
            f'CD = {cd:.3f}', ha='center', va='bottom',
            fontsize=10, color='red', fontweight='bold')

    # Draw cliques: connect models NOT significantly different (|rank_i - rank_j| < CD)
    drawn_cliques = set()
    for i in range(n_models):
        for j in range(i + 1, n_models):
            diff = abs(sorted_ranks[i] - sorted_ranks[j])
            if diff < cd:
                key = (i, j)
                if key not in drawn_cliques:
                    # Draw horizontal bar on the right side
                    y_i = y_positions[sorted_names[i]]
                    y_j = y_positions[sorted_names[j]]
                    x_bar = max(sorted_ranks[i], sorted_ranks[j]) + 0.1
                    ax.plot([x_bar, x_bar], [y_i, y_j],
                            color='navy', linewidth=4, alpha=0.5,
                            solid_capstyle='round')
                    drawn_cliques.add(key)

    # Title and info
    p_str = f"{friedman_result['p_value']:.4f}"
    sig_str = 'significant' if friedman_result['significant'] else 'not significant'
    ax.set_title(
        f'Critical Difference Diagram\n'
        f'Friedman χ²={friedman_result["statistic"]:.3f}, '
        f'p={p_str} ({sig_str}) | n={n_samples} samples',
        fontsize=13, fontweight='bold', pad=15
    )

    ax.spines['left'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)

    plt.tight_layout()
    path = output_dir / 'cd_diagram.png'
    plt.savefig(path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"CD Diagram saved: {path}")


def plot_pvalue_heatmap(wilcoxon_results: list, model_names: list,
                        output_dir: Path):
    """Heatmap of Bonferroni-corrected p-values from Wilcoxon tests."""
    n = len(model_names)
    p_matrix = np.ones((n, n))
    idx_map = {name: i for i, name in enumerate(model_names)}

    for r in wilcoxon_results:
        i = idx_map[r['model_1']]
        j = idx_map[r['model_2']]
        p_matrix[i, j] = r['p_bonferroni']
        p_matrix[j, i] = r['p_bonferroni']

    # Mask diagonal
    mask = np.eye(n, dtype=bool)

    fig, ax = plt.subplots(figsize=(10, 8))
    cmap = sns.diverging_palette(10, 130, as_cmap=True)

    sns.heatmap(
        p_matrix, annot=True, fmt='.3f',
        xticklabels=model_names, yticklabels=model_names,
        cmap='RdYlGn', vmin=0, vmax=1,
        linewidths=0.5, ax=ax,
        mask=mask,
        annot_kws={'size': 9}
    )

    # Highlight significant cells
    for i in range(n):
        for j in range(n):
            if i != j and p_matrix[i, j] < ALPHA:
                ax.add_patch(plt.Rectangle((j, i), 1, 1,
                             fill=False, edgecolor='red', lw=2.5))

    ax.set_title(
        'Wilcoxon Signed-Rank Test\nBonferroni-Corrected p-values '
        '(red border = significant)',
        fontsize=13, fontweight='bold'
    )
    ax.set_xticklabels(ax.get_xticklabels(), rotation=30, ha='right', fontsize=9)
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0, fontsize=9)

    plt.tight_layout()
    path = output_dir / 'wilcoxon_heatmap.png'
    plt.savefig(path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Wilcoxon heatmap saved: {path}")


def plot_nemenyi_heatmap(p_matrix, model_names: list, output_dir: Path):
    """Heatmap of Nemenyi post-hoc p-values."""
    if isinstance(p_matrix, pd.DataFrame):
        data = p_matrix.values
        labels = list(p_matrix.columns)
    else:
        data = np.array(p_matrix)
        labels = model_names

    n = len(labels)
    mask = np.eye(n, dtype=bool)

    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(
        data, annot=True, fmt='.3f',
        xticklabels=labels, yticklabels=labels,
        cmap='RdYlGn', vmin=0, vmax=1,
        linewidths=0.5, ax=ax, mask=mask,
        annot_kws={'size': 9}
    )

    for i in range(n):
        for j in range(n):
            if i != j and data[i, j] < ALPHA:
                ax.add_patch(plt.Rectangle((j, i), 1, 1,
                             fill=False, edgecolor='red', lw=2.5))

    ax.set_title(
        'Nemenyi Post-hoc Test\np-values (red border = significant)',
        fontsize=13, fontweight='bold'
    )
    ax.set_xticklabels(ax.get_xticklabels(), rotation=30, ha='right', fontsize=9)
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0, fontsize=9)

    plt.tight_layout()
    path = output_dir / 'nemenyi_heatmap.png'
    plt.savefig(path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Nemenyi heatmap saved: {path}")


def plot_accuracy_comparison(correct_vectors: dict, output_dir: Path):
    """Bar chart of model accuracies with confidence intervals."""
    model_names = list(correct_vectors.keys())
    accs = [correct_vectors[m].mean() for m in model_names]
    ns   = [len(correct_vectors[m]) for m in model_names]

    # Wilson confidence interval
    cis = []
    for acc, n in zip(accs, ns):
        z = 1.96
        ci = z * np.sqrt(acc * (1 - acc) / n)
        cis.append(ci)

    sorted_data = sorted(zip(accs, cis, model_names), reverse=True)
    s_accs  = [a for a, _, _ in sorted_data]
    s_cis   = [c for _, c, _ in sorted_data]
    s_names = [n for _, _, n in sorted_data]

    colors = plt.cm.RdYlGn(np.linspace(0.3, 0.9, len(s_names)))

    fig, ax = plt.subplots(figsize=(12, 6))
    bars = ax.barh(range(len(s_names)), s_accs, xerr=s_cis,
                   color=colors, edgecolor='grey', linewidth=0.8,
                   error_kw=dict(ecolor='black', capsize=4, lw=1.5))

    ax.set_yticks(range(len(s_names)))
    ax.set_yticklabels(s_names, fontsize=11)
    ax.set_xlabel('Accuracy', fontsize=12)
    ax.set_title('Model Accuracy Comparison\n(95% CI, evaluated on common test subset)',
                 fontsize=13, fontweight='bold')
    ax.set_xlim(0.5, 1.05)
    ax.axvline(x=0.95, color='red', linestyle='--', alpha=0.5, label='95% threshold')
    ax.legend(fontsize=10)

    for i, (acc, ci) in enumerate(zip(s_accs, s_cis)):
        ax.text(acc + ci + 0.005, i, f'{acc:.4f}',
                va='center', fontsize=10, fontweight='bold')

    plt.tight_layout()
    path = output_dir / 'accuracy_comparison.png'
    plt.savefig(path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Accuracy comparison saved: {path}")

def plot_cohens_d_heatmap(cohens_d_results: list, model_names: list,
                           output_dir: Path):
    """Heatmap of Cohen's d effect sizes."""
    n = len(model_names)
    d_matrix   = np.zeros((n, n))
    idx_map    = {name: i for i, name in enumerate(model_names)}

    for r in cohens_d_results:
        i = idx_map[r['model_1']]
        j = idx_map[r['model_2']]
        d_matrix[i, j] = r['cohens_d']
        d_matrix[j, i] = r['cohens_d']

    mask = np.eye(n, dtype=bool)

    fig, ax = plt.subplots(figsize=(10, 8))

    # Annotate with d value + interpretation
    annot_matrix = np.empty((n, n), dtype=object)
    for i in range(n):
        for j in range(n):
            if i == j:
                annot_matrix[i, j] = '-'
            else:
                d = d_matrix[i, j]
                if d < 0.2:
                    label = f'{d:.3f}\n(Neg.)'
                elif d < 0.5:
                    label = f'{d:.3f}\n(Small)'
                elif d < 0.8:
                    label = f'{d:.3f}\n(Med.)'
                else:
                    label = f'{d:.3f}\n(Large)'
                annot_matrix[i, j] = label

    sns.heatmap(
        d_matrix, annot=annot_matrix, fmt='',
        xticklabels=model_names, yticklabels=model_names,
        cmap='YlOrRd', vmin=0, vmax=2.0,
        linewidths=0.5, ax=ax, mask=mask,
        annot_kws={'size': 8}
    )

    # Add threshold lines on colorbar
    ax.set_title(
        "Cohen's d Effect Size\n"
        "(Negligible <0.2 | Small 0.2-0.5 | Medium 0.5-0.8 | Large >0.8)",
        fontsize=12, fontweight='bold'
    )
    ax.set_xticklabels(ax.get_xticklabels(), rotation=30, ha='right', fontsize=9)
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0, fontsize=9)

    plt.tight_layout()
    path = output_dir / 'cohens_d_heatmap.png'
    plt.savefig(path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Cohen's d heatmap saved: {path}")


# ══════════════════════════════════════════════════════════════════════════════
# REPORT
# ══════════════════════════════════════════════════════════════════════════════

def generate_report(friedman_result, wilcoxon_results, cohens_d_results,
                    avg_ranks, cd, correct_vectors, n_samples, output_dir):
    """Generate comprehensive text report."""
    lines = []
    lines.append('=' * 80)
    lines.append('STATISTICAL SIGNIFICANCE ANALYSIS REPORT')
    lines.append('Brain Tumor Classification - Model Comparison')
    lines.append('=' * 80)
    lines.append(f'Date: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}')
    lines.append(f'Samples in common evaluation subset: {n_samples}')
    lines.append(f'Models compared: {len(avg_ranks)}')
    lines.append(f'Significance level (alpha): {ALPHA}')
    lines.append('')

    # Accuracies
    lines.append('MODEL ACCURACIES')
    lines.append('-' * 50)
    sorted_models = sorted(avg_ranks.items(), key=lambda x: x[1])
    for name, rank in sorted_models:
        acc = correct_vectors[name].mean()
        lines.append(f'  {name:<35} Accuracy={acc:.4f} ({acc*100:.2f}%)  '
                     f'Avg Rank={rank:.3f}')
    lines.append('')

    # Friedman
    lines.append('1. FRIEDMAN TEST')
    lines.append('-' * 50)
    lines.append(f'  Chi-squared statistic: {friedman_result["statistic"]:.4f}')
    lines.append(f'  p-value:               {friedman_result["p_value"]:.6f}')
    lines.append(f'  Significant:           {"YES" if friedman_result["significant"] else "NO"}')
    if friedman_result['significant']:
        lines.append('  Interpretation: Significant differences exist among models.')
        lines.append('  Proceeding to post-hoc tests.')
    else:
        lines.append('  Interpretation: No significant differences detected.')
    lines.append('')

    # CD
    lines.append('2. CRITICAL DIFFERENCE')
    lines.append('-' * 50)
    lines.append(f'  CD (alpha={ALPHA}): {cd:.4f}')
    lines.append('  Models with |rank_i - rank_j| < CD are NOT significantly different.')
    lines.append('')

    # Wilcoxon
    lines.append('3. WILCOXON SIGNED-RANK TEST WITH BONFERRONI CORRECTION')
    lines.append('-' * 50)
    n_comp = len(wilcoxon_results)
    adj_alpha = ALPHA / n_comp
    lines.append(f'  Number of comparisons: {n_comp}')
    lines.append(f'  Bonferroni-adjusted alpha: {adj_alpha:.6f}')
    lines.append('')
    lines.append(f'  {"Model 1":<30} {"Model 2":<30} {"p_raw":>9} '
                 f'{"p_Bonf":>9} {"Sig":>5} {"Better":>20}')
    lines.append('  ' + '-' * 110)
    for r in sorted(wilcoxon_results, key=lambda x: x['p_bonferroni']):
        sig = 'YES' if r['significant'] else 'NO'
        lines.append(f'  {r["model_1"]:<30} {r["model_2"]:<30} '
                     f'{r["p_raw"]:>9.4f} {r["p_bonferroni"]:>9.4f} '
                     f'{sig:>5} {r["better_model"]:>20}')
    lines.append('')

    # Summary
    sig_pairs = [r for r in wilcoxon_results if r['significant']]
    lines.append('SUMMARY')
    lines.append('-' * 50)
    lines.append(f'  Significant pairwise differences: {len(sig_pairs)} / {n_comp}')
    if sig_pairs:
        lines.append('  Significantly different pairs:')
        for r in sig_pairs:
            lines.append(f'    {r["model_1"]} vs {r["model_2"]} '
                         f'(p_bonf={r["p_bonferroni"]:.4f}, '
                         f'better: {r["better_model"]})')
    lines.append('')

    # Cohen's d
    lines.append('4. COHENS d EFFECT SIZE')
    lines.append('-' * 50)
    lines.append('  Interpretation: d<0.2=Negligible | 0.2-0.5=Small | '
                 '0.5-0.8=Medium | >0.8=Large')
    lines.append('')
    lines.append(f'  {"Model 1":<30} {"Model 2":<30} {"d":>8} {"Effect":>12} {"Better":>20}')
    lines.append('  ' + '-' * 105)
    for r in sorted(cohens_d_results, key=lambda x: x['cohens_d'], reverse=True):
        lines.append(f'  {r["model_1"]:<30} {r["model_2"]:<30} '
                     f'{r["cohens_d"]:>8.4f} {r["interpretation"]:>12} '
                     f'{r["better_model"]:>20}')
    lines.append('')

    # Cohen's d summary
    negligible = [r for r in cohens_d_results if r['interpretation'] == 'Negligible']
    small      = [r for r in cohens_d_results if r['interpretation'] == 'Small']
    medium     = [r for r in cohens_d_results if r['interpretation'] == 'Medium']
    large      = [r for r in cohens_d_results if r['interpretation'] == 'Large']

    lines.append('  Effect Size Distribution:')
    lines.append(f'    Negligible (d<0.2):   {len(negligible)} pairs')
    lines.append(f'    Small (0.2-0.5):      {len(small)} pairs')
    lines.append(f'    Medium (0.5-0.8):     {len(medium)} pairs')
    lines.append(f'    Large (>0.8):         {len(large)} pairs')
    lines.append('')

    lines.append('=' * 80)

    report_text = '\n'.join(lines)
    path = output_dir / 'statistical_analysis_report.txt'
    with open(path, 'w') as f:
        f.write(report_text)
    print(f'\nReport saved: {path}')
    print('\n' + report_text)
    return report_text



def save_json_results(friedman_result, wilcoxon_results, cohens_d_results,
                      avg_ranks, cd, output_dir: Path):
    """Save all results as JSON for future use."""
    results = {
        'friedman': friedman_result,
        'wilcoxon_bonferroni': wilcoxon_results,
        'average_ranks': {k: float(v) for k, v in avg_ranks.items()},
        'critical_difference': float(cd),
    }
    path = output_dir / 'statistical_results.json'
    with open(path, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    print(f'JSON results saved: {path}')


# ══════════════════════════════════════════════════════════════════════════════
# MAIN PIPELINE
# ══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description='Unified Statistical Significance Analysis for all models')

    parser.add_argument('--data', required=True,
                        help='Path to 4-class dataset directory')
    parser.add_argument('--binary_model', required=True,
                        help='Path to Binary CNN .h5 model')
    parser.add_argument('--model_4class', required=True,
                        help='Path to 4-Class CNN .h5 model')
    parser.add_argument('--model_3class', required=True,
                        help='Path to 3-Class CNN .h5 model')
    parser.add_argument('--ensemble_dir', required=True,
                        help='Directory containing the 3 binary ensemble .h5 models')
    parser.add_argument('--resnet_dual', required=True,
                        help='Path to ResNet-50 Dual Branch .h5 model')
    parser.add_argument('--efnet_dual', required=True,
                        help='Path to EfficientNet-B3 Dual Branch .h5 model')
    parser.add_argument('--multi_dual', required=True,
                        help='Path to Multi-Dual System .h5 model')
    parser.add_argument('--output', default='statistical_analysis_results',
                        help='Output directory for results')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for test split')

    args = parser.parse_args()

    # Create output directory
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    print('=' * 60)
    print('UNIFIED STATISTICAL SIGNIFICANCE ANALYSIS')
    print('=' * 60)

    # ── 1. Load test data ──────────────────────────────────────────
    print('\n[1/6] Loading test data...')
    images, labels = load_test_data(args.data, seed=args.seed)

    # ── 2. Load models and get predictions ────────────────────────
    print('\n[2/6] Loading models and generating predictions...')

    correct_vectors = {}

    # Binary CNN
    model = load_standard_model(args.binary_model)
    correct_vectors['Binary CNN'] = predict_binary_cnn(model, images, labels)
    del model

    # 4-Class CNN
    model = load_standard_model(args.model_4class)
    correct_vectors['4-Class CNN'] = predict_4class_cnn(model, images, labels)
    del model

    # 3-Class CNN
    model = load_standard_model(args.model_3class)
    correct_vectors['3-Class CNN'] = predict_3class_cnn(model, images, labels)
    del model

    # 1-vs-1 Ensemble
    ensemble_dir = Path(args.ensemble_dir)
    pair_models = {}
    for cls_a, cls_b, fname in ENSEMBLE_PAIRS:
        fpath = ensemble_dir / fname
        if not fpath.exists():
            # try flat path
            fpath = Path(fname)
        m = load_standard_model(str(fpath))
        pair_models[(cls_a, cls_b, fname)] = m
    correct_vectors['1-vs-1 Ensemble'] = predict_ensemble_1vs1(
        pair_models, images, labels)
    for m in pair_models.values():
        del m

    # ResNet-50 Dual
    model = load_standard_model(args.resnet_dual)
    correct_vectors['ResNet-50 Dual'] = predict_resnet_dual(model, images, labels)
    del model

    # EfficientNet-B3 Dual
    model = load_standard_model(args.efnet_dual)
    correct_vectors['EfficientNet Dual'] = predict_efnet_dual(model, images, labels)
    del model

    # Multi-Dual System
    # Needs custom objects for masked loss
    sys.path.append(str(Path(args.multi_dual).parent.parent))
    try:
        from multi_dual_system_architecture import MultiDualSystemArchitecture
        arch = MultiDualSystemArchitecture()
        custom_objects = {
            '_masked_sparse_crossentropy': arch._masked_sparse_crossentropy,
            '_masked_sparse_accuracy':     arch._masked_sparse_accuracy
        }
        model = tf.keras.models.load_model(
            args.multi_dual, custom_objects=custom_objects, compile=False)
    except Exception:
        model = tf.keras.models.load_model(args.multi_dual, compile=False)
    correct_vectors['Multi-Dual System'] = predict_multi_dual(model, images, labels)
    del model

    # ── 3. Prepare common evaluation subset ───────────────────────
    print('\n[3/6] Preparing common evaluation subset...')
    filtered_vectors, common_indices = prepare_common_subset(correct_vectors)
    n_samples = len(common_indices)

    # ── 4. Statistical tests ───────────────────────────────────────
    print('\n[4/6] Running statistical tests...')

    friedman_result  = run_friedman_test(filtered_vectors)
    wilcoxon_results = run_wilcoxon_bonferroni(filtered_vectors)
    # Cohen's d effect size
    cohens_d_results = compute_cohens_d(filtered_vectors)
    avg_ranks        = compute_average_ranks(filtered_vectors)
    cd               = critical_difference(len(filtered_vectors), n_samples, ALPHA)

    print(f'\nCritical Difference (CD) at alpha={ALPHA}: {cd:.4f}')

    # Nemenyi
    if friedman_result['significant']:
        print('\nRunning Nemenyi post-hoc test...')
        nemenyi_matrix, nemenyi_names = run_nemenyi_test(filtered_vectors)
    else:
        print('\nFriedman not significant — skipping Nemenyi post-hoc.')
        nemenyi_matrix = None
        nemenyi_names  = list(filtered_vectors.keys())

    # ── 5. Visualizations ─────────────────────────────────────────
    print('\n[5/6] Creating visualizations...')

    plot_cd_diagram(avg_ranks, cd, n_samples, friedman_result, output_dir)
    plot_pvalue_heatmap(wilcoxon_results, list(filtered_vectors.keys()), output_dir)
    plot_accuracy_comparison(filtered_vectors, output_dir)
    plot_cohens_d_heatmap(cohens_d_results, list(filtered_vectors.keys()), output_dir)

    if nemenyi_matrix is not None:
        plot_nemenyi_heatmap(nemenyi_matrix, nemenyi_names, output_dir)

    # ── 6. Save results ───────────────────────────────────────────
    print('\n[6/6] Saving results...')

    generate_report(friedman_result, wilcoxon_results, cohens_d_results,
                    avg_ranks, cd, filtered_vectors, n_samples, output_dir)
    save_json_results(friedman_result, wilcoxon_results, cohens_d_results,
                    avg_ranks, cd, output_dir)

    print(f'\nAll results saved to: {output_dir}')
    print('Analysis complete.')


if __name__ == '__main__':
    main()