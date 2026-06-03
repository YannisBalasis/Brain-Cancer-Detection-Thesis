"""
MedViT V2 — Extended Statistical Significance Analysis
========================================================
Extends the existing 7-model SST to include MedViT V2 (model 8) and
optionally MedViT V2 Dual System (model 9).

All statistical procedures are identical to statistical_significance_analysis.py:
    1. Friedman Test
    2. Nemenyi Post-hoc
    3. Wilcoxon Signed-Rank with Bonferroni correction
    4. Cohen's d effect size
    5. Critical Difference Diagram

Usage (all 9 models):
    python medvit_v2_statistical_analysis.py \\
        --data             /path/to/dataset \\
        --binary_model     binary_cnn.h5 \\
        --model_4class     4class_cnn.h5 \\
        --model_3class     3class_cnn.h5 \\
        --ensemble_dir     ensemble_models/ \\
        --resnet_dual      resnet_dual.h5 \\
        --efnet_dual       efnet_dual.h5 \\
        --multi_dual       multi_dual.h5 \\
        --medvit_v2        medvit_v2_experiment/best_medvit_v2_tiny.keras \\
        --medvit_v2_dual   medvit_v2_dual_experiment/best_medvit_v2_dual.keras \\
        --output           extended_statistical_results

Drop --medvit_v2_dual to compare only 8 models.
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
from itertools import combinations

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from scipy.stats import wilcoxon, friedmanchisquare, rankdata

warnings.filterwarnings('ignore')

try:
    import scikit_posthocs as sp
    HAVE_POSTHOCS = True
except ImportError:
    HAVE_POSTHOCS = False
    os.system('pip install scikit-posthocs --break-system-packages -q')
    try:
        import scikit_posthocs as sp
        HAVE_POSTHOCS = True
    except ImportError:
        HAVE_POSTHOCS = False

# ══════════════════════════════════════════════════════════════════════
CLASS_NAMES_4 = ['glioma', 'meningioma', 'no_tumor', 'pituitary']
ENSEMBLE_PAIRS = [
    ('glioma',     'meningioma', 'best_glioma_vs_meningioma_model.h5'),
    ('glioma',     'pituitary',  'best_glioma_vs_pituitary_model.h5'),
    ('meningioma', 'pituitary',  'best_meningioma_vs_pituitary_model.h5'),
]
ALPHA = 0.05

# ══════════════════════════════════════════════════════════════════════
# PATHS  — edit these to match your local setup
# ══════════════════════════════════════════════════════════════════════
DATA_PATH = '/users/yannisbalasis/documents/thesis/data_multiclass'


# ══════════════════════════════════════════════════════════════════════
# CUSTOM OBJECTS for MedViT V2
# ══════════════════════════════════════════════════════════════════════

def _get_medvit_custom_objects():
    from medvit_v2_architecture import (
        LayerNorm2D, KANLayer, LFFNLayer, DiNALayer,
        LFPBlock, EMHSALayer, MHCALayer, GFPBlock,
        StemLayer, PatchEmbedding
    )
    return {
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


def _get_medvit_dual_custom_objects():
    """Custom objects for the MedViT V2 Dual System model."""
    base = _get_medvit_custom_objects()
    try:
        from medvit_v2_dual_architecture import MedViTDualFusion
        base['MedViTDualFusion'] = MedViTDualFusion
    except ImportError:
        pass
    return base


# ══════════════════════════════════════════════════════════════════════
# DATA LOADING
# ══════════════════════════════════════════════════════════════════════

def load_test_data(data_path: str, seed: int = 42):
    datagen = ImageDataGenerator(rescale=1.0 / 255.0, validation_split=0.1)
    gen = datagen.flow_from_directory(
        data_path,
        target_size=(224, 224),
        batch_size=1,
        class_mode='sparse',
        classes=CLASS_NAMES_4,
        subset='validation',
        shuffle=False,
        seed=seed,
    )
    images, labels = [], []
    for i in range(len(gen)):
        x, y = gen[i]
        images.append(x[0])
        labels.append(int(y[0]))

    images = np.array(images)
    labels = np.array(labels, dtype=np.int32)
    print(f'Test set: {len(images)} samples')
    for c, cnt in zip(*np.unique(labels, return_counts=True)):
        print(f'  {CLASS_NAMES_4[c]}: {cnt}')
    return images, labels


# ══════════════════════════════════════════════════════════════════════
# MODEL LOADERS
# ══════════════════════════════════════════════════════════════════════

def load_standard_model(path: str):
    model = tf.keras.models.load_model(path, compile=False)
    print(f'Loaded: {path}')
    return model


def load_medvit_model(path: str):
    custom = _get_medvit_custom_objects()
    model  = tf.keras.models.load_model(path, custom_objects=custom, compile=False)
    print(f'Loaded MedViT V2: {path}')
    return model


def load_medvit_dual_model(path: str):
    custom = _get_medvit_dual_custom_objects()
    model  = tf.keras.models.load_model(path, custom_objects=custom, compile=False)
    print(f'Loaded MedViT V2 Dual: {path}')
    return model


# ══════════════════════════════════════════════════════════════════════
# PREDICTORS
# ══════════════════════════════════════════════════════════════════════

def predict_binary_cnn(model, images, labels):
    probs = model.predict(images, verbose=0)
    if probs.shape[-1] == 1:
        pred_binary = (probs[:, 0] > 0.5).astype(int)
    else:
        pred_binary = np.argmax(probs, axis=1)
    true_binary = (labels != 2).astype(int)
    return (pred_binary == true_binary).astype(int)


def predict_4class(model, images, labels):
    probs  = model.predict(images, verbose=0)
    preds  = np.argmax(probs, axis=1)
    return (preds == labels).astype(int)


def predict_3class_cnn(model, images, labels):
    map_4_to_3    = {0: 0, 1: 1, 2: -1, 3: 2}
    tumor_indices = [i for i, l in enumerate(labels) if l != 2]
    tumor_images  = images[tumor_indices]
    tumor_labels  = np.array([map_4_to_3[labels[i]] for i in tumor_indices])

    probs  = model.predict(tumor_images, verbose=0)
    preds  = np.argmax(probs, axis=1)
    correct_tumor = (preds == tumor_labels).astype(int)

    correct_full = np.full(len(labels), -1, dtype=np.int32)
    for idx, orig in enumerate(tumor_indices):
        correct_full[orig] = correct_tumor[idx]
    return correct_full


def predict_ensemble_1vs1(pair_models: dict, images, labels):
    cls_to_idx   = {n: i for i, n in enumerate(CLASS_NAMES_4)}
    vote_matrix  = np.zeros((len(images), 4), dtype=np.float32)
    for (cls_a, cls_b, _), model in pair_models.items():
        probs  = model.predict(images, verbose=0)
        prob_b = probs[:, 0]
        prob_a = 1.0 - prob_b
        vote_matrix[:, cls_to_idx[cls_a]] += prob_a
        vote_matrix[:, cls_to_idx[cls_b]] += prob_b
    tumor_cols = [cls_to_idx[c] for c in ['glioma', 'meningioma', 'pituitary']]
    max_tumor  = vote_matrix[:, tumor_cols].max(axis=1)
    vote_matrix[:, cls_to_idx['no_tumor']] = np.where(max_tumor < 0.8, 1.5, 0.0)
    preds = np.argmax(vote_matrix, axis=1)
    return (preds == labels).astype(int)


def predict_dual_output(model, images, labels):
    """Handle both dict/list/single-array multi-output dual systems."""
    preds = model.predict(images, verbose=0)
    if isinstance(preds, dict):
        probs = preds.get('fusion_output',
                preds.get('fusion', list(preds.values())[-1]))
    elif isinstance(preds, list):
        probs = preds[-1]
    else:
        probs = preds
    return (np.argmax(probs, axis=1) == labels).astype(int)


# ══════════════════════════════════════════════════════════════════════
# STATISTICAL TESTS  (identical logic to original file)
# ══════════════════════════════════════════════════════════════════════

def prepare_common_subset(correct_vectors: dict):
    lengths = [len(v) for v in correct_vectors.values()]
    assert len(set(lengths)) == 1
    n = lengths[0]
    valid = np.ones(n, dtype=bool)
    for v in correct_vectors.values():
        valid &= (v != -1)
    idx = np.where(valid)[0]
    filtered = {name: v[idx] for name, v in correct_vectors.items()}
    print(f'\nCommon subset: {len(idx)} samples '
          f'(excluded {n - len(idx)} due to 3-class model scope)')
    return filtered, idx


def run_friedman_test(correct_vectors: dict):
    model_names = list(correct_vectors.keys())
    data = [correct_vectors[n].astype(float) for n in model_names]
    stat, p = friedmanchisquare(*data)
    sig = p < ALPHA
    print(f'\nFriedman Test: χ²={stat:.4f}, p={p:.6f}, '
          f'significant={"YES" if sig else "NO"}')
    return {'statistic': float(stat), 'p_value': float(p),
            'significant': bool(sig), 'model_names': model_names}


def _manual_nemenyi(correct_vectors: dict, model_names: list):
    from scipy.stats import norm
    k = len(model_names)
    data_matrix = np.column_stack(
        [correct_vectors[n].astype(float) for n in model_names])
    N = data_matrix.shape[0]
    ranks = np.array([rankdata(row) for row in data_matrix])
    avg_r = ranks.mean(axis=0)
    denom = np.sqrt(k * (k + 1) / (6 * N))
    pm = pd.DataFrame(np.ones((k, k)), index=model_names, columns=model_names)
    for i, j in combinations(range(k), 2):
        z = abs(avg_r[i] - avg_r[j]) / denom
        p = min(2 * (1 - norm.cdf(z)), 1.0)
        pm.iloc[i, j] = pm.iloc[j, i] = p
    return pm


def run_nemenyi_test(correct_vectors: dict):
    model_names = list(correct_vectors.keys())
    if HAVE_POSTHOCS:
        mat = np.column_stack([correct_vectors[n].astype(float)
                               for n in model_names])
        df  = pd.DataFrame(mat, columns=model_names)
        pm  = sp.posthoc_nemenyi_friedman(df)
    else:
        pm = _manual_nemenyi(correct_vectors, model_names)
    print('\nNemenyi Post-hoc p-values:')
    print(pd.DataFrame(pm, index=model_names,
                       columns=model_names).round(4).to_string())
    return pm, model_names


def run_wilcoxon_bonferroni(correct_vectors: dict):
    model_names = list(correct_vectors.keys())
    pairs       = list(combinations(range(len(model_names)), 2))
    n_comp      = len(pairs)
    results = []
    for i, j in pairs:
        ni, nj  = model_names[i], model_names[j]
        vi, vj  = correct_vectors[ni].astype(float), correct_vectors[nj].astype(float)
        diff    = vi - vj
        if np.all(diff == 0):
            stat, p_raw = 0.0, 1.0
        else:
            try:
                stat, p_raw = wilcoxon(vi, vj, alternative='two-sided',
                                       zero_method='wilcox')
            except ValueError:
                stat, p_raw = 0.0, 1.0
        p_corr = min(p_raw * n_comp, 1.0)
        results.append({
            'model_1': ni, 'model_2': nj,
            'accuracy_1': float(vi.mean()), 'accuracy_2': float(vj.mean()),
            'better_model': ni if vi.mean() >= vj.mean() else nj,
            'statistic': float(stat),
            'p_raw': float(p_raw), 'p_bonferroni': float(p_corr),
            'significant': bool(p_corr < ALPHA),
        })
    print(f'\nWilcoxon + Bonferroni ({n_comp} pairs):')
    for r in results:
        print(f'  {r["model_1"]:<28} vs {r["model_2"]:<28} '
              f'p_bonf={r["p_bonferroni"]:.4f} '
              f'{"*" if r["significant"] else ""}')
    return results


def compute_cohens_d(correct_vectors: dict):
    model_names = list(correct_vectors.keys())
    pairs = list(combinations(range(len(model_names)), 2))
    results = []
    for i, j in pairs:
        ni, nj  = model_names[i], model_names[j]
        vi, vj  = correct_vectors[ni].astype(float), correct_vectors[nj].astype(float)
        mi, mj  = vi.mean(), vj.mean()
        si, sj  = vi.std(ddof=1), vj.std(ddof=1)
        ni_, nj_ = len(vi), len(vj)
        pooled  = np.sqrt(((ni_ - 1) * si**2 + (nj_ - 1) * sj**2) / (ni_ + nj_ - 2))
        d = abs(mi - mj) / pooled if pooled > 0 else 0.0
        interp = ('Negligible' if d < 0.2 else 'Small' if d < 0.5
                  else 'Medium' if d < 0.8 else 'Large')
        results.append({'model_1': ni, 'model_2': nj,
                        'accuracy_1': float(mi), 'accuracy_2': float(mj),
                        'cohens_d': float(d), 'interpretation': interp,
                        'better_model': ni if mi >= mj else nj})
    return results


def compute_average_ranks(correct_vectors: dict):
    model_names = list(correct_vectors.keys())
    mat  = np.column_stack([correct_vectors[n].astype(float) for n in model_names])
    rnks = np.array([rankdata(-row, method='average') for row in mat])
    avg  = rnks.mean(axis=0)
    rank_dict = dict(zip(model_names, avg))
    print('\nAverage Ranks (lower = better):')
    for n, r in sorted(rank_dict.items(), key=lambda x: x[1]):
        print(f'  {n:<35} rank={r:.3f}  acc={correct_vectors[n].mean():.4f}')
    return rank_dict


def critical_difference(n_models: int, n_samples: int, alpha: float = 0.05):
    q_table = {
        (2, 0.05): 1.960, (3, 0.05): 2.344, (4, 0.05): 2.569,
        (5, 0.05): 2.728, (6, 0.05): 2.850, (7, 0.05): 2.949,
        (8, 0.05): 3.031, (9, 0.05): 3.102, (10,0.05): 3.164,
    }
    q = q_table.get((n_models, alpha), q_table.get((min(n_models, 10), alpha), 3.102))
    return q * np.sqrt(n_models * (n_models + 1) / (6 * n_samples))


# ══════════════════════════════════════════════════════════════════════
# VISUALISATIONS
# ══════════════════════════════════════════════════════════════════════

def plot_cd_diagram(avg_ranks, cd, n_samples, friedman_result, output_dir):
    model_names  = list(avg_ranks.keys())
    sorted_pairs = sorted(zip([avg_ranks[m] for m in model_names], model_names))
    s_ranks, s_names = zip(*sorted_pairs)
    s_ranks = list(s_ranks); s_names = list(s_names)

    n = len(s_names)
    fig, ax = plt.subplots(figsize=(14, max(5, n * 0.65 + 2)))

    min_r, max_r = max(1, min(s_ranks) - 0.5), min(n, max(s_ranks) + 0.5)
    ax.set_xlim(min_r - 0.3, max_r + 0.3)
    ax.set_ylim(-0.5, n + 1.5)
    ax.axhline(y=n + 0.8, color='black', linewidth=2, xmin=0.05, xmax=0.95)
    ax.set_xticks(np.arange(np.ceil(min_r), np.floor(max_r) + 1))
    ax.tick_params(axis='x', top=True, bottom=False,
                   labeltop=True, labelbottom=False, labelsize=11)
    ax.set_xlabel('Average Rank', fontsize=12, labelpad=2)
    ax.xaxis.set_label_position('top')
    ax.set_yticks([])

    colors = plt.cm.Set3(np.linspace(0, 1, n))
    y_pos  = {}
    for idx, (rank, name) in enumerate(zip(s_ranks, s_names)):
        y = n - idx - 1
        y_pos[name] = y
        ax.plot([rank, rank], [y, n + 0.8], color=colors[idx],
                lw=1.5, ls='--', alpha=0.7)
        ax.plot(rank, n + 0.8, 'o', color=colors[idx], ms=8, zorder=5)
        ha = 'left' if rank <= (min_r + max_r) / 2 else 'right'
        ax.text(rank + (0.05 if ha == 'left' else -0.05), y,
                f'{name}\n(rank={rank:.2f})',
                ha=ha, va='center', fontsize=8,
                bbox=dict(boxstyle='round,pad=0.3', facecolor=colors[idx],
                          alpha=0.2, edgecolor='none'))

    ax.annotate('', xy=(min_r + cd, n + 0.3), xytext=(min_r, n + 0.3),
                arrowprops=dict(arrowstyle='<->', color='red', lw=2))
    ax.text(min_r + cd / 2, n + 0.45, f'CD={cd:.3f}',
            ha='center', va='bottom', fontsize=10, color='red', fontweight='bold')

    drawn = set()
    for i in range(n):
        for j in range(i + 1, n):
            if abs(s_ranks[i] - s_ranks[j]) < cd and (i, j) not in drawn:
                yi, yj = y_pos[s_names[i]], y_pos[s_names[j]]
                xb = max(s_ranks[i], s_ranks[j]) + 0.1
                ax.plot([xb, xb], [yi, yj], color='navy', lw=4,
                        alpha=0.5, solid_capstyle='round')
                drawn.add((i, j))

    p_str  = f"{friedman_result['p_value']:.4f}"
    sig    = 'significant' if friedman_result['significant'] else 'not significant'
    ax.set_title(
        f'Critical Difference Diagram (Extended — {n} Models)\n'
        f'Friedman χ²={friedman_result["statistic"]:.3f}, p={p_str} ({sig})'
        f' | n={n_samples}',
        fontsize=12, fontweight='bold', pad=15)
    for spine in ['left', 'right', 'bottom']:
        ax.spines[spine].set_visible(False)

    plt.tight_layout()
    path = output_dir / 'cd_diagram_extended.png'
    plt.savefig(path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f'CD Diagram saved: {path}')


def plot_pvalue_heatmap(wilcoxon_results, model_names, output_dir, suffix=''):
    n     = len(model_names)
    pm    = np.ones((n, n))
    idx_m = {n_: i for i, n_ in enumerate(model_names)}
    for r in wilcoxon_results:
        i, j = idx_m[r['model_1']], idx_m[r['model_2']]
        pm[i, j] = pm[j, i] = r['p_bonferroni']
    mask = np.eye(n, dtype=bool)
    fig, ax = plt.subplots(figsize=(max(10, n * 1.1), max(8, n * 0.9)))
    sns.heatmap(pm, annot=True, fmt='.3f',
                xticklabels=model_names, yticklabels=model_names,
                cmap='RdYlGn', vmin=0, vmax=1,
                linewidths=0.5, ax=ax, mask=mask, annot_kws={'size': 8})
    for i in range(n):
        for j in range(n):
            if i != j and pm[i, j] < ALPHA:
                ax.add_patch(plt.Rectangle((j, i), 1, 1,
                             fill=False, edgecolor='red', lw=2))
    ax.set_title('Wilcoxon + Bonferroni p-values (red=significant)',
                 fontsize=13, fontweight='bold')
    ax.set_xticklabels(ax.get_xticklabels(), rotation=30, ha='right', fontsize=8)
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0, fontsize=8)
    plt.tight_layout()
    path = output_dir / f'wilcoxon_heatmap{suffix}.png'
    plt.savefig(path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f'Wilcoxon heatmap saved: {path}')


def plot_accuracy_comparison(correct_vectors, output_dir, suffix=''):
    model_names = list(correct_vectors.keys())
    accs = [correct_vectors[m].mean() for m in model_names]
    ns   = [len(correct_vectors[m]) for m in model_names]
    cis  = [1.96 * np.sqrt(a * (1 - a) / n) for a, n in zip(accs, ns)]

    data     = sorted(zip(accs, cis, model_names), reverse=True)
    s_accs   = [a for a, _, _ in data]
    s_cis    = [c for _, c, _ in data]
    s_names  = [n for _, _, n in data]
    colors   = plt.cm.RdYlGn(np.linspace(0.25, 0.9, len(s_names)))

    fig, ax = plt.subplots(figsize=(13, max(5, len(s_names) * 0.55 + 2)))
    ax.barh(range(len(s_names)), s_accs, xerr=s_cis, color=colors,
            edgecolor='grey', lw=0.8,
            error_kw=dict(ecolor='black', capsize=4, lw=1.5))
    ax.set_yticks(range(len(s_names)))
    ax.set_yticklabels(s_names, fontsize=10)
    ax.set_xlabel('Accuracy', fontsize=12)
    ax.set_title('Model Accuracy Comparison (Extended)\n95% CI on common test subset',
                 fontsize=13, fontweight='bold')
    ax.set_xlim(0.45, 1.08)
    ax.axvline(x=0.95, color='red', ls='--', alpha=0.5, label='95% threshold')
    ax.legend(fontsize=10)
    for i, (a, ci) in enumerate(zip(s_accs, s_cis)):
        ax.text(a + ci + 0.005, i, f'{a:.4f}', va='center',
                fontsize=9, fontweight='bold')
    plt.tight_layout()
    path = output_dir / f'accuracy_comparison{suffix}.png'
    plt.savefig(path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f'Accuracy comparison saved: {path}')


def plot_cohens_d_heatmap(cohens_d_results, model_names, output_dir, suffix=''):
    n      = len(model_names)
    dm     = np.zeros((n, n))
    idx_m  = {n_: i for i, n_ in enumerate(model_names)}
    for r in cohens_d_results:
        i, j = idx_m[r['model_1']], idx_m[r['model_2']]
        dm[i, j] = dm[j, i] = r['cohens_d']
    mask   = np.eye(n, dtype=bool)
    annot  = np.empty((n, n), dtype=object)
    for i in range(n):
        for j in range(n):
            if i == j:
                annot[i, j] = '-'
            else:
                d = dm[i, j]
                label = (f'{d:.3f}\n(Neg.)'   if d < 0.2 else
                         f'{d:.3f}\n(Small)'  if d < 0.5 else
                         f'{d:.3f}\n(Med.)'   if d < 0.8 else
                         f'{d:.3f}\n(Large)')
                annot[i, j] = label
    fig, ax = plt.subplots(figsize=(max(10, n * 1.1), max(8, n * 0.9)))
    sns.heatmap(dm, annot=annot, fmt='',
                xticklabels=model_names, yticklabels=model_names,
                cmap='YlOrRd', vmin=0, vmax=2.0,
                linewidths=0.5, ax=ax, mask=mask, annot_kws={'size': 7})
    ax.set_title("Cohen's d Effect Size (Extended)\n"
                 "Negligible<0.2 | Small 0.2-0.5 | Medium 0.5-0.8 | Large>0.8",
                 fontsize=12, fontweight='bold')
    ax.set_xticklabels(ax.get_xticklabels(), rotation=30, ha='right', fontsize=8)
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0, fontsize=8)
    plt.tight_layout()
    path = output_dir / f'cohens_d_heatmap{suffix}.png'
    plt.savefig(path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Cohen's d heatmap saved: {path}")


def plot_nemenyi_heatmap(p_matrix, model_names, output_dir):
    data   = p_matrix.values if isinstance(p_matrix, pd.DataFrame) else np.array(p_matrix)
    labels = list(p_matrix.columns) if isinstance(p_matrix, pd.DataFrame) else model_names
    n      = len(labels)
    mask   = np.eye(n, dtype=bool)
    fig, ax = plt.subplots(figsize=(max(10, n * 1.1), max(8, n * 0.9)))
    sns.heatmap(data, annot=True, fmt='.3f',
                xticklabels=labels, yticklabels=labels,
                cmap='RdYlGn', vmin=0, vmax=1,
                linewidths=0.5, ax=ax, mask=mask, annot_kws={'size': 8})
    for i in range(n):
        for j in range(n):
            if i != j and data[i, j] < ALPHA:
                ax.add_patch(plt.Rectangle((j, i), 1, 1,
                             fill=False, edgecolor='red', lw=2))
    ax.set_title('Nemenyi Post-hoc p-values (extended)',
                 fontsize=13, fontweight='bold')
    ax.set_xticklabels(ax.get_xticklabels(), rotation=30, ha='right', fontsize=8)
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0, fontsize=8)
    plt.tight_layout()
    path = output_dir / 'nemenyi_heatmap_extended.png'
    plt.savefig(path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f'Nemenyi heatmap saved: {path}')


# ══════════════════════════════════════════════════════════════════════
# REPORT
# ══════════════════════════════════════════════════════════════════════

def generate_report(friedman_result, wilcoxon_results, cohens_d_results,
                    avg_ranks, cd, correct_vectors, n_samples, output_dir):
    lines = ['=' * 80,
             'EXTENDED STATISTICAL SIGNIFICANCE ANALYSIS REPORT',
             'Brain Tumor Classification — MedViT V2 vs All Models',
             '=' * 80,
             f'Date: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}',
             f'Common evaluation subset: {n_samples} samples',
             f'Models compared: {len(avg_ranks)}',
             f'Significance level (alpha): {ALPHA}',
             '']

    lines += ['MODEL ACCURACIES (sorted by avg rank)', '-' * 60]
    for name, rank in sorted(avg_ranks.items(), key=lambda x: x[1]):
        acc = correct_vectors[name].mean()
        lines.append(f'  {name:<38} acc={acc:.4f} ({acc*100:.2f}%)  '
                     f'rank={rank:.3f}')
    lines += ['', '1. FRIEDMAN TEST', '-' * 60,
              f'  χ²={friedman_result["statistic"]:.4f}',
              f'  p={friedman_result["p_value"]:.6f}',
              f'  Significant: {"YES" if friedman_result["significant"] else "NO"}',
              '']
    lines += ['2. CRITICAL DIFFERENCE', '-' * 60,
              f'  CD (alpha={ALPHA}): {cd:.4f}', '']

    lines += ['3. WILCOXON SIGNED-RANK + BONFERRONI', '-' * 60]
    n_comp  = len(wilcoxon_results)
    adj_alp = ALPHA / n_comp
    lines.append(f'  Comparisons: {n_comp}, adj. alpha: {adj_alp:.6f}')
    for r in sorted(wilcoxon_results, key=lambda x: x['p_bonferroni']):
        sig = 'YES' if r['significant'] else 'NO'
        lines.append(f'  {r["model_1"]:<30} vs {r["model_2"]:<30} '
                     f'p_bonf={r["p_bonferroni"]:.4f} sig={sig}')

    sig_pairs = [r for r in wilcoxon_results if r['significant']]
    lines += ['', 'SUMMARY', '-' * 60,
              f'  Significant pairs: {len(sig_pairs)}/{n_comp}']
    for r in sig_pairs:
        lines.append(f'    {r["model_1"]} vs {r["model_2"]} '
                     f'(better: {r["better_model"]}, '
                     f'p_bonf={r["p_bonferroni"]:.4f})')

    lines += ['', '4. COHENS d EFFECT SIZE', '-' * 60]
    for r in sorted(cohens_d_results, key=lambda x: x['cohens_d'], reverse=True):
        lines.append(f'  {r["model_1"]:<30} vs {r["model_2"]:<30} '
                     f'd={r["cohens_d"]:.4f} ({r["interpretation"]}) '
                     f'better={r["better_model"]}')
    lines.append('=' * 80)

    text = '\n'.join(lines)
    path = output_dir / 'extended_statistical_report.txt'
    with open(path, 'w') as f:
        f.write(text)
    print(f'\nReport saved: {path}')
    print('\n' + text)


def save_json(friedman_result, wilcoxon_results, cohens_d_results,
              avg_ranks, cd, output_dir):
    data = {
        'friedman': friedman_result,
        'wilcoxon_bonferroni': wilcoxon_results,
        'cohens_d': cohens_d_results,
        'average_ranks': {k: float(v) for k, v in avg_ranks.items()},
        'critical_difference': float(cd),
    }
    path = output_dir / 'extended_statistical_results.json'
    with open(path, 'w') as f:
        json.dump(data, f, indent=2, default=str)
    print(f'JSON saved: {path}')


# ══════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════

def main():
    p = argparse.ArgumentParser(
        description='Extended Statistical Analysis — adds MedViT V2 to comparison')
    p.add_argument('--data',          default=DATA_PATH)
    p.add_argument('--binary_model',  required=True)
    p.add_argument('--model_4class',  required=True)
    p.add_argument('--model_3class',  required=True)
    p.add_argument('--ensemble_dir',  required=True)
    p.add_argument('--resnet_dual',   required=True)
    p.add_argument('--efnet_dual',    required=True)
    p.add_argument('--multi_dual',    required=True)
    p.add_argument('--medvit_v2',     required=True,
                   help='Path to trained MedViT V2 .keras model')
    p.add_argument('--medvit_v2_dual', default=None,
                   help='Path to MedViT V2 Dual .keras model (optional)')
    p.add_argument('--output',        default='extended_statistical_results')
    p.add_argument('--seed',          type=int, default=42)
    args = p.parse_args()

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    print('=' * 65)
    print('  EXTENDED STATISTICAL SIGNIFICANCE ANALYSIS')
    print('  (Original 7 Models + MedViT V2)')
    print('=' * 65)

    # ── 1. Load test data ──────────────────────────────────────────
    print('\n[1] Loading test data...')
    images, labels = load_test_data(args.data, seed=args.seed)

    # ── 2. Predictions ─────────────────────────────────────────────
    print('\n[2] Generating predictions...')
    cv = {}

    model = load_standard_model(args.binary_model)
    cv['Binary CNN'] = predict_binary_cnn(model, images, labels); del model

    model = load_standard_model(args.model_4class)
    cv['4-Class CNN'] = predict_4class(model, images, labels); del model

    model = load_standard_model(args.model_3class)
    cv['3-Class CNN'] = predict_3class_cnn(model, images, labels); del model

    ensemble_dir = Path(args.ensemble_dir)
    pair_models  = {}
    for cls_a, cls_b, fname in ENSEMBLE_PAIRS:
        fpath = ensemble_dir / fname
        if not fpath.exists():
            fpath = Path(fname)
        pair_models[(cls_a, cls_b, fname)] = load_standard_model(str(fpath))
    cv['1-vs-1 Ensemble'] = predict_ensemble_1vs1(pair_models, images, labels)
    for m in pair_models.values():
        del m

    model = load_standard_model(args.resnet_dual)
    cv['ResNet-50 Dual'] = predict_dual_output(model, images, labels); del model

    model = load_standard_model(args.efnet_dual)
    cv['EfficientNet Dual'] = predict_dual_output(model, images, labels); del model

    sys.path.append(str(Path(args.multi_dual).parent.parent))
    try:
        from multi_dual_system_architecture import MultiDualSystemArchitecture
        arch = MultiDualSystemArchitecture()
        custom = {'_masked_sparse_crossentropy': arch._masked_sparse_crossentropy,
                  '_masked_sparse_accuracy':     arch._masked_sparse_accuracy}
        model = tf.keras.models.load_model(args.multi_dual,
                                           custom_objects=custom, compile=False)
    except Exception:
        model = tf.keras.models.load_model(args.multi_dual, compile=False)
    cv['Multi-Dual System'] = predict_dual_output(model, images, labels); del model

    # MedViT V2
    model = load_medvit_model(args.medvit_v2)
    cv['MedViT V2'] = predict_4class(model, images, labels); del model

    # MedViT V2 Dual (optional)
    if args.medvit_v2_dual:
        model = load_medvit_dual_model(args.medvit_v2_dual)
        cv['MedViT V2 Dual'] = predict_dual_output(model, images, labels)
        del model

    # ── 3. Common subset ───────────────────────────────────────────
    print('\n[3] Preparing common evaluation subset...')
    filtered, common_idx = prepare_common_subset(cv)
    n_samples = len(common_idx)

    # ── 4. Statistical tests ───────────────────────────────────────
    print('\n[4] Running statistical tests...')
    model_names      = list(filtered.keys())
    friedman_result  = run_friedman_test(filtered)
    wilcoxon_results = run_wilcoxon_bonferroni(filtered)
    cohens_d_results = compute_cohens_d(filtered)
    avg_ranks        = compute_average_ranks(filtered)
    cd               = critical_difference(len(filtered), n_samples, ALPHA)
    print(f'\nCritical Difference at alpha={ALPHA}: {cd:.4f}')

    nemenyi_matrix = None
    if friedman_result['significant']:
        print('\nRunning Nemenyi post-hoc...')
        nemenyi_matrix, _ = run_nemenyi_test(filtered)

    # ── 5. Visualisations ─────────────────────────────────────────
    print('\n[5] Creating visualisations...')
    plot_cd_diagram(avg_ranks, cd, n_samples, friedman_result, output_dir)
    plot_pvalue_heatmap(wilcoxon_results, model_names, output_dir)
    plot_accuracy_comparison(filtered, output_dir)
    plot_cohens_d_heatmap(cohens_d_results, model_names, output_dir)
    if nemenyi_matrix is not None:
        plot_nemenyi_heatmap(nemenyi_matrix, model_names, output_dir)

    # ── 6. Save results ───────────────────────────────────────────
    print('\n[6] Saving results...')
    generate_report(friedman_result, wilcoxon_results, cohens_d_results,
                    avg_ranks, cd, filtered, n_samples, output_dir)
    save_json(friedman_result, wilcoxon_results, cohens_d_results,
              avg_ranks, cd, output_dir)

    print(f'\nAll results saved to: {output_dir}')
    print('Extended analysis complete.')


if __name__ == '__main__':
    main()
