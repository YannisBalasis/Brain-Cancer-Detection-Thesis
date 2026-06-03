"""
MedViT V2 Dual System — Training Script
=========================================
Two-phase training for the MedViT V2 + 4-Class CNN dual-branch fusion model.

Phase 1 — Fusion head only  (branches frozen):
    AdamW lr=1e-3, up to 40 epochs, early stopping patience=10

Phase 2 — Full fine-tuning  (branches unfrozen):
    AdamW lr=1e-4, up to 60 epochs, early stopping patience=15

Usage:
    python medvit_v2_dual_train.py \\
        --data         /path/to/dataset \\
        --medvit_model medvit_v2_experiment/best_medvit_v2_tiny.keras \\
        --cnn_model    multiclass_4class_experiment/best_model.h5 \\
        --output       medvit_v2_dual_experiment
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

import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight

from medvit_v2_architecture import (
    build_medvit_v2, LayerNorm2D, KANLayer, LFFNLayer, DiNALayer,
    LFPBlock, EMHSALayer, MHCALayer, GFPBlock, StemLayer, PatchEmbedding,
)
from medvit_v2_dual_architecture import (
    build_dual_system, set_branches_trainable, CUSTOM_OBJECTS
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


def make_generators(data_path: str, batch_size: int, seed: int = 42):
    """Train (70%) and val (20%) generators; test (10%) held out for evaluation."""
    df_train, df_val, df_test = make_file_splits(data_path, seed)

    aug = ImageDataGenerator(
        rescale=1.0 / 255,
        rotation_range=15,
        width_shift_range=0.1,
        height_shift_range=0.1,
        horizontal_flip=True,
        zoom_range=0.1,
        brightness_range=[0.9, 1.1],
        fill_mode='nearest',
    )
    val_datagen = ImageDataGenerator(rescale=1.0 / 255)

    common = dict(x_col='filename', y_col='class', target_size=IMG_SIZE,
                  batch_size=batch_size, classes=CLASS_NAMES, class_mode='categorical')

    train_gen = aug.flow_from_dataframe(df_train, shuffle=True,  seed=seed, **common)
    val_gen   = val_datagen.flow_from_dataframe(df_val, shuffle=False, **common)

    print(f'Train: {len(df_train)} samples  '
          f'Val: {len(df_val)} samples  '
          f'Test (held out): {len(df_test)} samples')
    return train_gen, val_gen


def compute_weights(generator):
    labels  = generator.classes
    classes = np.unique(labels)
    weights = compute_class_weight('balanced', classes=classes, y=labels)
    return dict(enumerate(weights))


# ══════════════════════════════════════════════════════════════════════
# CALLBACKS
# ══════════════════════════════════════════════════════════════════════

def make_callbacks(output_dir: Path, phase: int, patience: int,
                   monitor: str = 'val_accuracy'):
    suffix = f'_phase{phase}'
    return [
        keras.callbacks.ModelCheckpoint(
            filepath=str(output_dir / f'best_medvit_v2_dual{suffix}.keras'),
            monitor=monitor,
            save_best_only=True,
            save_weights_only=False,
            verbose=1,
        ),
        keras.callbacks.EarlyStopping(
            monitor=monitor,
            patience=patience,
            restore_best_weights=True,
            verbose=1,
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=max(patience // 3, 5),
            min_lr=1e-7,
            verbose=1,
        ),
        keras.callbacks.CSVLogger(
            str(output_dir / f'dual_training_log{suffix}.csv')),
        keras.callbacks.TensorBoard(
            log_dir=str(output_dir / f'logs/phase{phase}'),
            histogram_freq=0,
            write_graph=False,
        ),
    ]


# ══════════════════════════════════════════════════════════════════════
# COMPILE
# ══════════════════════════════════════════════════════════════════════

def compile_model(model: keras.Model, lr: float, weight_decay: float = 1e-4):
    optimizer = keras.optimizers.AdamW(
        learning_rate=lr, weight_decay=weight_decay,
        beta_1=0.9, beta_2=0.999,
    )
    model.compile(
        optimizer=optimizer,
        loss='categorical_crossentropy',
        metrics=[
            'accuracy',
            keras.metrics.AUC(name='auc', multi_label=False),
            keras.metrics.Precision(name='precision'),
            keras.metrics.Recall(name='recall'),
        ],
    )


# ══════════════════════════════════════════════════════════════════════
# TRAINING
# ══════════════════════════════════════════════════════════════════════

def train(args):
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    print('=' * 65)
    print('  MedViT V2 Dual System — Training')
    print('=' * 65)
    print(f'  Data         : {args.data}')
    print(f'  MedViT V2    : {args.medvit_model}')
    print(f'  CNN Model    : {args.cnn_model}')
    print(f'  Output       : {output_dir}')
    print(f'  P1 epochs    : {args.epochs_p1}')
    print(f'  P2 epochs    : {args.epochs_p2}')
    print(f'  Batch        : {args.batch_size}')
    print(f'  Seed         : {args.seed}')

    tf.random.set_seed(args.seed)
    np.random.seed(args.seed)

    # ── Data ─────────────────────────────────────────────────────────
    train_gen, val_gen = make_generators(
        args.data, batch_size=args.batch_size, seed=args.seed)
    class_weights = compute_weights(train_gen)
    print(f'Class weights: {class_weights}')

    # ── Load pre-trained branches ────────────────────────────────────
    print('\nLoading pre-trained MedViT V2 ...')
    medvit_model = keras.models.load_model(
        args.medvit_model, custom_objects=CUSTOM_OBJECTS, compile=False)

    print('Loading pre-trained 4-Class CNN ...')
    cnn_model = keras.models.load_model(args.cnn_model, compile=False)

    # ── Build dual system ────────────────────────────────────────────
    print('\nBuilding dual system ...')
    dual_model, medvit_ext, cnn_ext = build_dual_system(
        medvit_model, cnn_model,
        input_shape=IMG_SIZE + (3,),
        num_classes=NUM_CLASSES,
        dropout_rate=args.dropout,
    )
    dual_model.summary(line_length=100)
    print(f'Total parameters: {dual_model.count_params():,}')

    # ════════════════════════════════════════════════════════════════
    # PHASE 1: Train fusion head only
    # ════════════════════════════════════════════════════════════════
    print('\n' + '─' * 55)
    print('  PHASE 1: Training fusion head (branches frozen)')
    print('─' * 55)

    set_branches_trainable(dual_model, False, medvit_ext, cnn_ext)
    compile_model(dual_model, lr=args.lr_p1)

    t_start = datetime.now()
    history_p1 = dual_model.fit(
        train_gen,
        epochs=args.epochs_p1,
        validation_data=val_gen,
        class_weight=class_weights,
        callbacks=make_callbacks(output_dir, phase=1, patience=args.patience_p1),
        verbose=1,
    )

    best_p1 = max(history_p1.history['val_accuracy'])
    print(f'\nPhase 1 best val accuracy: {best_p1:.4f}')

    # ════════════════════════════════════════════════════════════════
    # PHASE 2: Fine-tune all layers
    # ════════════════════════════════════════════════════════════════
    print('\n' + '─' * 55)
    print('  PHASE 2: Full fine-tuning (all layers unfrozen)')
    print('─' * 55)

    set_branches_trainable(dual_model, True, medvit_ext, cnn_ext)
    compile_model(dual_model, lr=args.lr_p2)

    history_p2 = dual_model.fit(
        train_gen,
        epochs=args.epochs_p2,
        validation_data=val_gen,
        class_weight=class_weights,
        callbacks=make_callbacks(output_dir, phase=2, patience=args.patience_p2),
        verbose=1,
    )

    elapsed = datetime.now() - t_start
    best_p2 = max(history_p2.history['val_accuracy'])
    best_overall = max(best_p1, best_p2)
    print(f'\nPhase 2 best val accuracy : {best_p2:.4f}')
    print(f'Overall best val accuracy : {best_overall:.4f}')
    print(f'Total training time       : {elapsed}')

    # ── Save best model (phase 2 checkpoint already saved by callback)
    # Reload the best phase-2 checkpoint and save as final
    best_p2_path = output_dir / 'best_medvit_v2_dual_phase2.keras'
    if best_p2_path.exists():
        best_model = keras.models.load_model(
            str(best_p2_path), custom_objects=CUSTOM_OBJECTS, compile=False)
        final_path = output_dir / 'best_medvit_v2_dual.keras'
        best_model.save(str(final_path))
        print(f'Final model saved: {final_path}')

    # ── Save history ─────────────────────────────────────────────────
    combined_history = {}
    for k, v in history_p1.history.items():
        combined_history[f'p1_{k}'] = [float(x) for x in v]
    for k, v in history_p2.history.items():
        combined_history[f'p2_{k}'] = [float(x) for x in v]

    with open(output_dir / 'dual_training_history.json', 'w') as f:
        json.dump(combined_history, f, indent=2)

    # ── Save config ───────────────────────────────────────────────────
    config = {
        'medvit_model':   args.medvit_model,
        'cnn_model':      args.cnn_model,
        'data':           args.data,
        'output':         str(output_dir),
        'epochs_p1':      args.epochs_p1,
        'epochs_p2':      args.epochs_p2,
        'lr_p1':          args.lr_p1,
        'lr_p2':          args.lr_p2,
        'batch_size':     args.batch_size,
        'dropout':        args.dropout,
        'seed':           args.seed,
        'timestamp':      t_start.isoformat(),
        'training_time_sec': elapsed.total_seconds(),
        'best_val_accuracy_p1': float(best_p1),
        'best_val_accuracy_p2': float(best_p2),
        'total_params':        dual_model.count_params(),
    }
    with open(output_dir / 'dual_experiment_config.json', 'w') as f:
        json.dump(config, f, indent=2)

    # ── Plot training curves ──────────────────────────────────────────
    _plot_dual_history(history_p1, history_p2, output_dir)

    print(f'\nAll outputs saved to: {output_dir}')
    return dual_model, history_p1, history_p2


def _plot_dual_history(history_p1, history_p2, output_dir: Path):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    h1, h2 = history_p1.history, history_p2.history
    e1 = range(1, len(h1['loss']) + 1)
    e2 = range(len(e1) + 1, len(e1) + len(h2['loss']) + 1)

    def _concat(key):
        v1 = h1.get(key, [])
        v2 = h2.get(key, [])
        return list(e1)[:len(v1)] + list(e2)[:len(v2)], v1 + v2

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    for ax, key, label in [
        (axes[0, 0], 'accuracy', 'Accuracy'),
        (axes[0, 1], 'loss',     'Loss'),
        (axes[1, 0], 'auc',      'AUC'),
        (axes[1, 1], 'precision', 'Precision'),
    ]:
        ep, tr = _concat(key)
        ev, vl = _concat(f'val_{key}')
        ax.plot(ep, tr, label='Train')
        ax.plot(ev, vl, label='Val')
        # Vertical line at phase boundary
        ax.axvline(x=len(e1) + 0.5, color='grey', ls='--',
                   alpha=0.6, label='Phase 2 starts')
        ax.set_title(f'MedViT V2 Dual — {label}')
        ax.set_xlabel('Epoch')
        ax.set_ylabel(label)
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)

    plt.suptitle('MedViT V2 Dual System — Training Curves', fontsize=14)
    plt.tight_layout()
    fig.savefig(output_dir / 'dual_training_curves.png', dpi=300, bbox_inches='tight')
    plt.close(fig)
    print('Training curves saved.')


# ══════════════════════════════════════════════════════════════════════
# CLI
# ══════════════════════════════════════════════════════════════════════

def parse_args():
    p = argparse.ArgumentParser(description='Train MedViT V2 Dual System')
    p.add_argument('--data',         default=DATA_PATH)
    p.add_argument('--medvit_model', required=True,
                   help='Pre-trained MedViT V2 .keras file')
    p.add_argument('--cnn_model',    required=True,
                   help='Pre-trained 4-Class CNN .h5 or .keras file')
    p.add_argument('--output',       default=None)
    p.add_argument('--epochs_p1',    type=int,   default=40,
                   help='Phase-1 epochs (fusion head only)')
    p.add_argument('--epochs_p2',    type=int,   default=60,
                   help='Phase-2 epochs (full fine-tune)')
    p.add_argument('--lr_p1',        type=float, default=1e-3)
    p.add_argument('--lr_p2',        type=float, default=1e-4)
    p.add_argument('--batch_size',   type=int,   default=32)
    p.add_argument('--dropout',      type=float, default=0.4)
    p.add_argument('--patience_p1',  type=int,   default=10)
    p.add_argument('--patience_p2',  type=int,   default=15)
    p.add_argument('--seed',         type=int,   default=42)
    return p.parse_args()


if __name__ == '__main__':
    args = parse_args()
    if args.output is None:
        ts = datetime.now().strftime('%Y%m%d_%H%M%S')
        args.output = f'medvit_v2_dual_experiment_{ts}'
    train(args)
