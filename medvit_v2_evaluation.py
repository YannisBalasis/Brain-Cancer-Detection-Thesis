"""
MedViT V2 — Comprehensive Evaluation
======================================
Evaluates a trained MedViT V2 model on the 4-class brain tumor dataset.
Produces:
    - Classification report (per-class precision/recall/F1)
    - Confusion matrix (raw + %)
    - AUC-ROC curves (per class + macro)
    - Accuracy, Sensitivity, Specificity, F1, AUC per class
    - JSON results file

Usage:
    python medvit_v2_evaluation.py \\
        --model medvit_v2_experiment/best_medvit_v2_tiny.keras \\
        --data  /path/to/dataset \\
        --output medvit_v2_eval_results
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

import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    classification_report, confusion_matrix, roc_auc_score,
    roc_curve, precision_recall_fscore_support, accuracy_score
)

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


# ══════════════════════════════════════════════════════════════════════
# CUSTOM OBJECTS for model loading
# ══════════════════════════════════════════════════════════════════════

CUSTOM_OBJECTS = {
    'LayerNorm2D': LayerNorm2D,
    'KANLayer':    KANLayer,
    'LFFNLayer':   LFFNLayer,
    'DiNALayer':   DiNALayer,
    'LFPBlock':    LFPBlock,
    'EMHSALayer':  EMHSALayer,
    'MHCALayer':   MHCALayer,
    'GFPBlock':    GFPBlock,
    'StemLayer':   StemLayer,
    'PatchEmbedding': PatchEmbedding,
}


# ══════════════════════════════════════════════════════════════════════
# DATA
# ══════════════════════════════════════════════════════════════════════

def make_file_splits(data_path: str, seed: int = 42):
    """Stratified 70 / 20 / 10 split. Returns (df_train, df_val, df_test)."""
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


def load_test_generator(data_path: str, batch_size: int = 32, seed: int = 42):
    _, _, df_test = make_file_splits(data_path, seed)
    datagen = ImageDataGenerator(rescale=1.0 / 255)
    gen = datagen.flow_from_dataframe(
        df_test,
        x_col='filename', y_col='class',
        target_size=IMG_SIZE, batch_size=batch_size,
        classes=CLASS_NAMES, class_mode='categorical',
        shuffle=False,
    )
    print(f'Test generator: {gen.samples} samples in {len(gen)} batches')
    return gen


def collect_predictions(model, generator):
    """Run inference and collect true labels + predicted probabilities."""
    generator.reset()
    y_prob = model.predict(generator, verbose=1)           # (N, 4)
    y_true = generator.classes[:len(y_prob)]               # (N,)
    y_pred = np.argmax(y_prob, axis=1)                     # (N,)
    return y_true, y_pred, y_prob


# ══════════════════════════════════════════════════════════════════════
# METRICS
# ══════════════════════════════════════════════════════════════════════

def compute_all_metrics(y_true, y_pred, y_prob):
    """Compute comprehensive metrics for all classes."""
    acc = accuracy_score(y_true, y_pred)

    # Per-class precision / recall / F1 / support
    prec, rec, f1, sup = precision_recall_fscore_support(
        y_true, y_pred, labels=list(range(NUM_CLASSES)), zero_division=0)

    # Specificity per class = TN / (TN + FP)
    cm   = confusion_matrix(y_true, y_pred, labels=list(range(NUM_CLASSES)))
    spec = []
    for i in range(NUM_CLASSES):
        tp = cm[i, i]
        fn = cm[i, :].sum() - tp
        fp = cm[:, i].sum() - tp
        tn = cm.sum() - tp - fn - fp
        spec.append(tn / (tn + fp + 1e-9))

    # AUC per class (one-vs-rest)
    from sklearn.preprocessing import label_binarize
    y_bin = label_binarize(y_true, classes=list(range(NUM_CLASSES)))
    auc_per_class = []
    for i in range(NUM_CLASSES):
        try:
            auc_per_class.append(roc_auc_score(y_bin[:, i], y_prob[:, i]))
        except ValueError:
            auc_per_class.append(float('nan'))
    macro_auc = np.nanmean(auc_per_class)

    # Macro averages
    macro_prec = float(np.nanmean(prec))
    macro_rec  = float(np.nanmean(rec))
    macro_f1   = float(np.nanmean(f1))
    macro_spec = float(np.nanmean(spec))

    results = {
        'overall_accuracy':  float(acc),
        'macro_auc':         float(macro_auc),
        'macro_precision':   macro_prec,
        'macro_recall':      macro_rec,
        'macro_f1':          macro_f1,
        'macro_specificity': macro_spec,
        'per_class': {}
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


def print_metrics(results):
    print('\n' + '=' * 65)
    print('  MedViT V2 — Evaluation Results')
    print('=' * 65)
    print(f'  Overall Accuracy  : {results["overall_accuracy"]:.4f} '
          f'({results["overall_accuracy"]*100:.2f}%)')
    print(f'  Macro AUC         : {results["macro_auc"]:.4f}')
    print(f'  Macro Precision   : {results["macro_precision"]:.4f}')
    print(f'  Macro Recall      : {results["macro_recall"]:.4f}')
    print(f'  Macro F1          : {results["macro_f1"]:.4f}')
    print(f'  Macro Specificity : {results["macro_specificity"]:.4f}')
    print()
    header = f'  {"Class":<14} {"Prec":>7} {"Rec":>7} {"F1":>7} {"Spec":>7} {"AUC":>7} {"N":>6}'
    print(header)
    print('  ' + '-' * 57)
    for name, m in results['per_class'].items():
        print(f'  {name:<14} {m["precision"]:>7.4f} {m["recall"]:>7.4f} '
              f'{m["f1"]:>7.4f} {m["specificity"]:>7.4f} '
              f'{m["auc"]:>7.4f} {m["support"]:>6}')
    print('=' * 65)


# ══════════════════════════════════════════════════════════════════════
# VISUALISATIONS
# ══════════════════════════════════════════════════════════════════════

def plot_confusion_matrix(cm, output_dir: Path):
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    labels = [c.replace('_', '\n') for c in CLASS_NAMES]

    # Raw counts
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=labels, yticklabels=labels, ax=axes[0])
    axes[0].set_title('Confusion Matrix (counts)', fontweight='bold')
    axes[0].set_xlabel('Predicted'); axes[0].set_ylabel('True')

    # Percentages
    cm_pct = cm.astype(float) / cm.sum(axis=1, keepdims=True) * 100
    sns.heatmap(cm_pct, annot=True, fmt='.1f', cmap='Blues',
                xticklabels=labels, yticklabels=labels, ax=axes[1])
    axes[1].set_title('Confusion Matrix (%)', fontweight='bold')
    axes[1].set_xlabel('Predicted'); axes[1].set_ylabel('True')

    plt.suptitle('MedViT V2 — Confusion Matrix', fontsize=14, fontweight='bold')
    plt.tight_layout()
    fig.savefig(output_dir / 'confusion_matrix.png', dpi=300, bbox_inches='tight')
    plt.close(fig)
    print('Confusion matrix saved.')


def plot_roc_curves(y_true, y_prob, output_dir: Path):
    from sklearn.preprocessing import label_binarize
    y_bin = label_binarize(y_true, classes=list(range(NUM_CLASSES)))
    colors = ['#e41a1c', '#377eb8', '#4daf4a', '#984ea3']

    fig, ax = plt.subplots(figsize=(8, 7))
    for i, (name, col) in enumerate(zip(CLASS_NAMES, colors)):
        fpr, tpr, _ = roc_curve(y_bin[:, i], y_prob[:, i])
        auc = roc_auc_score(y_bin[:, i], y_prob[:, i])
        ax.plot(fpr, tpr, color=col, lw=2,
                label=f'{name} (AUC={auc:.3f})')

    ax.plot([0, 1], [0, 1], 'k--', lw=1)
    ax.set_xlim([0.0, 1.0]); ax.set_ylim([0.0, 1.02])
    ax.set_xlabel('False Positive Rate', fontsize=12)
    ax.set_ylabel('True Positive Rate',  fontsize=12)
    ax.set_title('MedViT V2 — ROC Curves (One-vs-Rest)', fontsize=13,
                 fontweight='bold')
    ax.legend(loc='lower right', fontsize=10)
    ax.grid(True, alpha=0.3)

    fig.savefig(output_dir / 'roc_curves.png', dpi=300, bbox_inches='tight')
    plt.close(fig)
    print('ROC curves saved.')


def plot_class_metrics(results, output_dir: Path):
    names   = CLASS_NAMES
    metrics = ['precision', 'recall', 'f1', 'specificity', 'auc']
    labels  = ['Precision', 'Recall', 'F1', 'Specificity', 'AUC']
    colors  = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']

    fig, ax = plt.subplots(figsize=(11, 6))
    x       = np.arange(len(names))
    width   = 0.15

    for j, (metric, label, col) in enumerate(zip(metrics, labels, colors)):
        vals = [results['per_class'][n][metric] for n in names]
        ax.bar(x + j * width, vals, width, label=label, color=col, alpha=0.85)

    ax.set_xticks(x + 2 * width)
    ax.set_xticklabels([c.replace('_', '\n') for c in names], fontsize=11)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel('Score', fontsize=12)
    ax.set_title('MedViT V2 — Per-Class Metrics', fontsize=13, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3, axis='y')

    fig.savefig(output_dir / 'per_class_metrics.png', dpi=300, bbox_inches='tight')
    plt.close(fig)
    print('Per-class metrics bar chart saved.')


# ══════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════

def evaluate(args):
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f'Loading model: {args.model}')
    model = keras.models.load_model(
        args.model,
        custom_objects=CUSTOM_OBJECTS,
        compile=False,
    )
    print('Model loaded.')

    gen = load_test_generator(args.data, batch_size=args.batch_size,
                               seed=args.seed)
    y_true, y_pred, y_prob = collect_predictions(model, gen)

    results, cm = compute_all_metrics(y_true, y_pred, y_prob)
    print_metrics(results)

    # Save JSON
    results['timestamp'] = datetime.now().isoformat()
    results['model_path'] = str(args.model)
    with open(output_dir / 'evaluation_results.json', 'w') as f:
        json.dump(results, f, indent=2)

    # Save classification report text
    report = classification_report(
        y_true, y_pred, target_names=CLASS_NAMES, digits=4)
    with open(output_dir / 'classification_report.txt', 'w') as f:
        f.write('MedViT V2 — Classification Report\n')
        f.write('=' * 50 + '\n')
        f.write(report)
    print('\nClassification report:')
    print(report)

    # Save confusion matrix as text
    with open(output_dir / 'confusion_matrix.txt', 'w') as f:
        f.write('Confusion Matrix\n')
        header = '         ' + '  '.join(f'{c[:8]:>8}' for c in CLASS_NAMES)
        f.write(header + '\n')
        for i, row in enumerate(cm):
            f.write(f'{CLASS_NAMES[i][:8]:>8}  ' +
                    '  '.join(f'{v:>8}' for v in row) + '\n')

    # Plots
    plot_confusion_matrix(cm, output_dir)
    plot_roc_curves(y_true, y_prob, output_dir)
    plot_class_metrics(results, output_dir)

    print(f'\nAll evaluation results saved to: {output_dir}')
    return results


def parse_args():
    p = argparse.ArgumentParser(description='Evaluate MedViT V2')
    p.add_argument('--model',      required=True,
                   help='Path to .keras or .h5 model file')
    p.add_argument('--data',       default=DATA_PATH,
                   help='Path to 4-class dataset directory')
    p.add_argument('--output',     default='medvit_v2_eval_results',
                   help='Output directory')
    p.add_argument('--batch_size', type=int, default=32)
    p.add_argument('--seed',       type=int, default=42)
    return p.parse_args()


if __name__ == '__main__':
    evaluate(parse_args())
