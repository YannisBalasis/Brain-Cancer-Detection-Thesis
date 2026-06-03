"""
MedViT V2 Dual System — Evaluation
=====================================
Evaluates a trained MedViT V2 Dual model on the 4-class brain tumor dataset.
Produces the same output artefacts as medvit_v2_evaluation.py plus a
branch-contribution analysis (individual branch predictions vs fusion).

Usage:
    python medvit_v2_dual_evaluation.py \\
        --model      medvit_v2_dual_experiment/best_medvit_v2_dual.keras \\
        --medvit_v2  medvit_v2_experiment/best_medvit_v2_tiny.keras \\
        --cnn_model  multiclass_4class_experiment/best_model.h5 \\
        --data       /path/to/dataset \\
        --output     medvit_v2_dual_eval_results
"""

import os
import sys
import json
import argparse
import warnings
import numpy as np
from pathlib import Path
from datetime import datetime

warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import (
    classification_report, confusion_matrix, roc_auc_score,
    roc_curve, precision_recall_fscore_support, accuracy_score
)
from sklearn.preprocessing import label_binarize

from medvit_v2_architecture import (
    LayerNorm2D, KANLayer, LFFNLayer, DiNALayer,
    LFPBlock, EMHSALayer, MHCALayer, GFPBlock, StemLayer, PatchEmbedding,
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


# ══════════════════════════════════════════════════════════════════════
# DATA
# ══════════════════════════════════════════════════════════════════════

def load_test_generator(data_path: str, batch_size: int = 32, seed: int = 42):
    datagen = ImageDataGenerator(rescale=1.0 / 255, validation_split=0.2)
    gen = datagen.flow_from_directory(
        data_path,
        target_size=IMG_SIZE,
        batch_size=batch_size,
        classes=CLASS_NAMES,
        class_mode='categorical',
        subset='validation',
        shuffle=False,
        seed=seed,
    )
    print(f'Test generator: {gen.samples} samples in {len(gen)} batches')
    return gen


def collect_predictions(model, generator):
    generator.reset()
    y_prob = model.predict(generator, verbose=1)
    y_true = generator.classes[:len(y_prob)]
    y_pred = np.argmax(y_prob, axis=1)
    return y_true, y_pred, y_prob


# ══════════════════════════════════════════════════════════════════════
# METRICS
# ══════════════════════════════════════════════════════════════════════

def compute_all_metrics(y_true, y_pred, y_prob, model_label: str = 'Model'):
    acc = accuracy_score(y_true, y_pred)
    prec, rec, f1, sup = precision_recall_fscore_support(
        y_true, y_pred, labels=list(range(NUM_CLASSES)), zero_division=0)

    cm = confusion_matrix(y_true, y_pred, labels=list(range(NUM_CLASSES)))
    spec = []
    for i in range(NUM_CLASSES):
        tp = cm[i, i]; fn = cm[i, :].sum() - tp
        fp = cm[:, i].sum() - tp; tn = cm.sum() - tp - fn - fp
        spec.append(tn / (tn + fp + 1e-9))

    y_bin = label_binarize(y_true, classes=list(range(NUM_CLASSES)))
    auc_per_class = []
    for i in range(NUM_CLASSES):
        try:
            auc_per_class.append(roc_auc_score(y_bin[:, i], y_prob[:, i]))
        except ValueError:
            auc_per_class.append(float('nan'))
    macro_auc = float(np.nanmean(auc_per_class))

    results = {
        'model_label':       model_label,
        'overall_accuracy':  float(acc),
        'macro_auc':         macro_auc,
        'macro_precision':   float(np.nanmean(prec)),
        'macro_recall':      float(np.nanmean(rec)),
        'macro_f1':          float(np.nanmean(f1)),
        'macro_specificity': float(np.nanmean(spec)),
        'per_class':         {}
    }
    for i, name in enumerate(CLASS_NAMES):
        results['per_class'][name] = {
            'precision':   float(prec[i]),
            'recall':      float(rec[i]),
            'f1':          float(f1[i]),
            'specificity': float(spec[i]),
            'auc':         float(auc_per_class[i]),
            'support':     int(sup[i]),
        }
    return results, cm


def print_metrics(results, label: str = ''):
    header = label or results.get('model_label', 'Model')
    print('\n' + '=' * 65)
    print(f'  {header} — Evaluation Results')
    print('=' * 65)
    print(f'  Overall Accuracy  : {results["overall_accuracy"]:.4f} '
          f'({results["overall_accuracy"]*100:.2f}%)')
    print(f'  Macro AUC         : {results["macro_auc"]:.4f}')
    print(f'  Macro Precision   : {results["macro_precision"]:.4f}')
    print(f'  Macro Recall      : {results["macro_recall"]:.4f}')
    print(f'  Macro F1          : {results["macro_f1"]:.4f}')
    print(f'  Macro Specificity : {results["macro_specificity"]:.4f}')
    print()
    print(f'  {"Class":<14} {"Prec":>7} {"Rec":>7} {"F1":>7} '
          f'{"Spec":>7} {"AUC":>7} {"N":>6}')
    print('  ' + '-' * 57)
    for name, m in results['per_class'].items():
        print(f'  {name:<14} {m["precision"]:>7.4f} {m["recall"]:>7.4f} '
              f'{m["f1"]:>7.4f} {m["specificity"]:>7.4f} '
              f'{m["auc"]:>7.4f} {m["support"]:>6}')
    print('=' * 65)


# ══════════════════════════════════════════════════════════════════════
# VISUALISATIONS
# ══════════════════════════════════════════════════════════════════════

def plot_confusion_matrix(cm, output_dir: Path, filename: str = 'confusion_matrix.png',
                          title: str = 'MedViT V2 Dual — Confusion Matrix'):
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    labels = [c.replace('_', '\n') for c in CLASS_NAMES]

    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=labels, yticklabels=labels, ax=axes[0])
    axes[0].set_title('Counts', fontweight='bold')
    axes[0].set_xlabel('Predicted'); axes[0].set_ylabel('True')

    cm_pct = cm.astype(float) / cm.sum(axis=1, keepdims=True) * 100
    sns.heatmap(cm_pct, annot=True, fmt='.1f', cmap='Blues',
                xticklabels=labels, yticklabels=labels, ax=axes[1])
    axes[1].set_title('Percentage (%)', fontweight='bold')
    axes[1].set_xlabel('Predicted'); axes[1].set_ylabel('True')

    plt.suptitle(title, fontsize=14, fontweight='bold')
    plt.tight_layout()
    fig.savefig(output_dir / filename, dpi=300, bbox_inches='tight')
    plt.close(fig)


def plot_roc_curves(y_true, y_prob, output_dir: Path,
                    filename: str = 'roc_curves.png',
                    title: str = 'MedViT V2 Dual — ROC Curves'):
    y_bin  = label_binarize(y_true, classes=list(range(NUM_CLASSES)))
    colors = ['#e41a1c', '#377eb8', '#4daf4a', '#984ea3']
    fig, ax = plt.subplots(figsize=(8, 7))
    for i, (name, col) in enumerate(zip(CLASS_NAMES, colors)):
        fpr, tpr, _ = roc_curve(y_bin[:, i], y_prob[:, i])
        auc = roc_auc_score(y_bin[:, i], y_prob[:, i])
        ax.plot(fpr, tpr, color=col, lw=2, label=f'{name} (AUC={auc:.3f})')
    ax.plot([0, 1], [0, 1], 'k--', lw=1)
    ax.set_xlim([0, 1]); ax.set_ylim([0, 1.02])
    ax.set_xlabel('FPR', fontsize=12); ax.set_ylabel('TPR', fontsize=12)
    ax.set_title(title, fontsize=13, fontweight='bold')
    ax.legend(loc='lower right', fontsize=10); ax.grid(True, alpha=0.3)
    fig.savefig(output_dir / filename, dpi=300, bbox_inches='tight')
    plt.close(fig)


def plot_class_metrics(results, output_dir: Path,
                       filename: str = 'per_class_metrics.png',
                       title: str = 'MedViT V2 Dual — Per-Class Metrics'):
    names   = CLASS_NAMES
    metrics = ['precision', 'recall', 'f1', 'specificity', 'auc']
    labels  = ['Precision', 'Recall', 'F1', 'Specificity', 'AUC']
    colors  = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
    x, w    = np.arange(len(names)), 0.15

    fig, ax = plt.subplots(figsize=(11, 6))
    for j, (m, lbl, col) in enumerate(zip(metrics, labels, colors)):
        vals = [results['per_class'][n][m] for n in names]
        ax.bar(x + j * w, vals, w, label=lbl, color=col, alpha=0.85)
    ax.set_xticks(x + 2 * w)
    ax.set_xticklabels([c.replace('_', '\n') for c in names], fontsize=11)
    ax.set_ylim(0, 1.05); ax.set_ylabel('Score', fontsize=12)
    ax.set_title(title, fontsize=13, fontweight='bold')
    ax.legend(fontsize=10); ax.grid(True, alpha=0.3, axis='y')
    fig.savefig(output_dir / filename, dpi=300, bbox_inches='tight')
    plt.close(fig)


def plot_branch_comparison(fusion_results, medvit_results, cnn_results,
                           output_dir: Path):
    """Bar chart comparing fusion vs individual branches on key metrics."""
    metrics = ['overall_accuracy', 'macro_auc', 'macro_f1', 'macro_precision',
               'macro_recall', 'macro_specificity']
    labels  = ['Accuracy', 'AUC', 'F1', 'Precision', 'Recall', 'Specificity']
    x       = np.arange(len(metrics))
    width   = 0.25

    fv = [fusion_results[m]  for m in metrics]
    mv = [medvit_results[m]  for m in metrics]
    cv = [cnn_results[m]     for m in metrics]

    fig, ax = plt.subplots(figsize=(13, 6))
    ax.bar(x - width, fv, width, label='Fusion (Dual)',  color='#2ca02c', alpha=0.9)
    ax.bar(x,         mv, width, label='MedViT V2 only', color='#1f77b4', alpha=0.9)
    ax.bar(x + width, cv, width, label='CNN only',        color='#ff7f0e', alpha=0.9)

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=11)
    ax.set_ylim(0, 1.1)
    ax.set_ylabel('Score', fontsize=12)
    ax.set_title('Branch Contribution Analysis\n'
                 '(Fusion vs Individual Branches)',
                 fontsize=13, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3, axis='y')

    for bars in [ax.containers[0], ax.containers[1], ax.containers[2]]:
        for bar in bars:
            h = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2, h + 0.005,
                    f'{h:.3f}', ha='center', va='bottom', fontsize=7)

    plt.tight_layout()
    fig.savefig(output_dir / 'branch_contribution.png', dpi=300, bbox_inches='tight')
    plt.close(fig)
    print('Branch contribution chart saved.')


# ══════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════

def evaluate(args):
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── Load dual model ──────────────────────────────────────────────
    print(f'Loading dual model: {args.model}')
    dual_model = keras.models.load_model(
        args.model, custom_objects=CUSTOM_OBJECTS, compile=False)
    print('Dual model loaded.')

    gen = load_test_generator(args.data, batch_size=args.batch_size, seed=args.seed)
    y_true, y_pred_dual, y_prob_dual = collect_predictions(dual_model, gen)

    # ── Dual metrics ─────────────────────────────────────────────────
    results_dual, cm_dual = compute_all_metrics(
        y_true, y_pred_dual, y_prob_dual, 'MedViT V2 Dual')
    print_metrics(results_dual, 'MedViT V2 Dual System')

    # ── Optional individual branch evaluation ────────────────────────
    results_medvit, results_cnn = None, None

    if args.medvit_v2:
        print(f'\nLoading MedViT V2 branch: {args.medvit_v2}')
        medvit_model = keras.models.load_model(
            args.medvit_v2, custom_objects=CUSTOM_OBJECTS, compile=False)
        gen.reset()
        y_prob_mv = medvit_model.predict(gen, verbose=1)
        y_pred_mv = np.argmax(y_prob_mv, axis=1)
        results_medvit, _ = compute_all_metrics(
            y_true[:len(y_pred_mv)], y_pred_mv, y_prob_mv, 'MedViT V2 Branch')
        print_metrics(results_medvit, 'MedViT V2 (standalone)')

    if args.cnn_model:
        print(f'\nLoading CNN branch: {args.cnn_model}')
        cnn_model = keras.models.load_model(args.cnn_model, compile=False)
        gen.reset()
        y_prob_cnn = cnn_model.predict(gen, verbose=1)
        y_pred_cnn = np.argmax(y_prob_cnn, axis=1)
        results_cnn, _ = compute_all_metrics(
            y_true[:len(y_pred_cnn)], y_pred_cnn, y_prob_cnn, 'CNN Branch')
        print_metrics(results_cnn, '4-Class CNN (standalone)')

    # ── Visualisations ───────────────────────────────────────────────
    plot_confusion_matrix(cm_dual, output_dir)
    plot_roc_curves(y_true, y_prob_dual, output_dir)
    plot_class_metrics(results_dual, output_dir)

    if results_medvit and results_cnn:
        plot_branch_comparison(results_dual, results_medvit, results_cnn, output_dir)

    # ── Classification report ────────────────────────────────────────
    report = classification_report(
        y_true, y_pred_dual, target_names=CLASS_NAMES, digits=4)
    with open(output_dir / 'classification_report.txt', 'w') as f:
        f.write('MedViT V2 Dual System — Classification Report\n')
        f.write('=' * 50 + '\n')
        f.write(report)
    print('\nClassification report:')
    print(report)

    # ── Confusion matrix text ────────────────────────────────────────
    with open(output_dir / 'confusion_matrix.txt', 'w') as f:
        f.write('Confusion Matrix\n')
        header = '         ' + '  '.join(f'{c[:8]:>8}' for c in CLASS_NAMES)
        f.write(header + '\n')
        for i, row in enumerate(cm_dual):
            f.write(f'{CLASS_NAMES[i][:8]:>8}  ' +
                    '  '.join(f'{v:>8}' for v in row) + '\n')

    # ── Save JSON ────────────────────────────────────────────────────
    all_results = {
        'timestamp':   datetime.now().isoformat(),
        'model_path':  str(args.model),
        'dual':        results_dual,
    }
    if results_medvit:
        all_results['medvit_v2_branch'] = results_medvit
    if results_cnn:
        all_results['cnn_branch'] = results_cnn

    with open(output_dir / 'evaluation_results.json', 'w') as f:
        json.dump(all_results, f, indent=2)

    print(f'\nAll evaluation results saved to: {output_dir}')
    return all_results


# ══════════════════════════════════════════════════════════════════════
# CLI
# ══════════════════════════════════════════════════════════════════════

def parse_args():
    p = argparse.ArgumentParser(description='Evaluate MedViT V2 Dual System')
    p.add_argument('--model',      required=True,
                   help='Path to trained dual .keras model')
    p.add_argument('--data',       default=DATA_PATH,
                   help='Path to 4-class dataset directory')
    p.add_argument('--output',     default='medvit_v2_dual_eval_results')
    p.add_argument('--medvit_v2',  default=None,
                   help='MedViT V2 branch model for standalone comparison (optional)')
    p.add_argument('--cnn_model',  default=None,
                   help='4-Class CNN branch model for standalone comparison (optional)')
    p.add_argument('--batch_size', type=int, default=32)
    p.add_argument('--seed',       type=int, default=42)
    return p.parse_args()


if __name__ == '__main__':
    evaluate(parse_args())
