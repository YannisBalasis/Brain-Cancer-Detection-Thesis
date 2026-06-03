"""
MedViT V2 — Training Script
============================
Brain Tumor Classification (4-class): Glioma, Meningioma, No Tumor, Pituitary

Training protocol (from paper Section 4.2):
    - 100 epochs, batch size 128, AdamW lr=0.001
    - LR decayed ×0.1 at epochs 50 and 75
    - Images resized to 224×224

Usage:
    python medvit_v2_train.py \\
        --data /path/to/dataset \\
        --variant tiny \\
        --output medvit_v2_experiment_<timestamp>
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

from medvit_v2_architecture import build_medvit_v2, MEDVIT_V2_CONFIGS

# ══════════════════════════════════════════════════════════════════════
# PATHS  — edit these to match your local setup
# ══════════════════════════════════════════════════════════════════════
DATA_PATH = '/users/yannisbalasis/documents/thesis/data_multiclass'

# ══════════════════════════════════════════════════════════════════════
# CONSTANTS
# ══════════════════════════════════════════════════════════════════════
CLASS_NAMES   = ['glioma', 'meningioma', 'no_tumor', 'pituitary']
IMG_SIZE      = (224, 224)
NUM_CLASSES   = 4


# ══════════════════════════════════════════════════════════════════════
# DATA LOADERS
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
    """Train (70%) and val (20%) generators; test (10%) is held out for evaluation."""
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

    print(f'Train : {len(df_train)} samples  '
          f'Val : {len(df_val)} samples  '
          f'Test (held out) : {len(df_test)} samples')
    return train_gen, val_gen


def compute_weights(generator):
    """Compute balanced class weights from generator labels."""
    labels  = generator.classes
    classes = np.unique(labels)
    weights = compute_class_weight('balanced', classes=classes, y=labels)
    return dict(enumerate(weights))


# ══════════════════════════════════════════════════════════════════════
# LR SCHEDULE — decay ×0.1 at epochs 50 and 75 (paper Section 4.2)
# ══════════════════════════════════════════════════════════════════════

class MedViTLRSchedule(keras.optimizers.schedules.LearningRateSchedule):
    """Step LR: initial_lr × 0.1 at milestone epochs."""
    def __init__(self, initial_lr=1e-3, milestones=(50, 75), decay=0.1):
        super().__init__()
        self.initial_lr = initial_lr
        self.milestones = sorted(milestones)
        self.decay      = decay

    def __call__(self, step):
        # step is global step; we need epoch ≈ step / steps_per_epoch
        # We use a simple epoch-based approach via callback instead (see below)
        return self.initial_lr

    def get_config(self):
        return {'initial_lr': self.initial_lr,
                'milestones':  self.milestones,
                'decay':       self.decay}


class StepLRCallback(keras.callbacks.Callback):
    """Decay LR by ×decay at specified epochs (1-indexed)."""
    def __init__(self, milestones=(50, 75), decay=0.1):
        super().__init__()
        self.milestones = set(milestones)
        self.decay      = decay

    def on_epoch_begin(self, epoch, logs=None):
        if (epoch + 1) in self.milestones:
            old_lr = float(self.model.optimizer.learning_rate)
            new_lr = old_lr * self.decay
            self.model.optimizer.learning_rate.assign(new_lr)
            print(f'\n[StepLR] Epoch {epoch+1}: LR {old_lr:.6f} → {new_lr:.6f}')


# ══════════════════════════════════════════════════════════════════════
# CALLBACKS
# ══════════════════════════════════════════════════════════════════════

def build_callbacks(output_dir: Path, variant: str, monitor='val_accuracy',
                    patience=15):
    return [
        keras.callbacks.ModelCheckpoint(
            filepath=str(output_dir / f'best_medvit_v2_{variant}.keras'),
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
        StepLRCallback(milestones=(50, 75), decay=0.1),
        keras.callbacks.CSVLogger(
            str(output_dir / f'medvit_v2_{variant}_training_log.csv'),
        ),
        keras.callbacks.TensorBoard(
            log_dir=str(output_dir / 'logs'),
            histogram_freq=0,
            write_graph=False,
        ),
    ]


# ══════════════════════════════════════════════════════════════════════
# TRAINING
# ══════════════════════════════════════════════════════════════════════

def train(args):
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    print('=' * 65)
    print(f'  MedViT V2 — {args.variant.upper()} — Brain Tumor Classification')
    print('=' * 65)
    print(f'  Data      : {args.data}')
    print(f'  Output    : {output_dir}')
    print(f'  Epochs    : {args.epochs}')
    print(f'  Batch     : {args.batch_size}')
    print(f'  LR        : {args.lr}')
    print(f'  Variant   : {args.variant}')
    print(f'  Seed      : {args.seed}')

    tf.random.set_seed(args.seed)
    np.random.seed(args.seed)

    # ── Data ────────────────────────────────────────────────────
    train_gen, val_gen = make_generators(
        args.data, batch_size=args.batch_size, seed=args.seed)

    class_weights = compute_weights(train_gen)
    print(f'\nClass weights: {class_weights}')

    # ── Model ───────────────────────────────────────────────────
    model = build_medvit_v2(
        input_shape=IMG_SIZE + (3,),
        num_classes=NUM_CLASSES,
        variant=args.variant,
        dropout_rate=args.dropout,
    )
    model.summary(line_length=100)
    print(f'Parameters: {model.count_params():,}')

    # ── Compile (AdamW as in paper) ──────────────────────────────
    optimizer = keras.optimizers.AdamW(
        learning_rate=args.lr,
        weight_decay=1e-4,
        beta_1=0.9,
        beta_2=0.999,
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

    # ── Train ───────────────────────────────────────────────────
    t0 = datetime.now()
    print(f'\nTraining started: {t0.strftime("%Y-%m-%d %H:%M:%S")}')

    history = model.fit(
        train_gen,
        epochs=args.epochs,
        validation_data=val_gen,
        class_weight=class_weights,
        callbacks=build_callbacks(output_dir, args.variant,
                                  patience=args.patience),
        verbose=1,
    )

    elapsed = datetime.now() - t0
    print(f'\nTraining finished. Elapsed: {elapsed}')

    # ── Save training history ─────────────────────────────────
    history_path = output_dir / 'training_history.json'
    with open(history_path, 'w') as f:
        json.dump({k: [float(v) for v in vals]
                   for k, vals in history.history.items()}, f, indent=2)
    print(f'History saved: {history_path}')

    # ── Save config ───────────────────────────────────────────
    config = {
        'variant':    args.variant,
        'epochs':     args.epochs,
        'batch_size': args.batch_size,
        'lr':         args.lr,
        'dropout':    args.dropout,
        'seed':       args.seed,
        'data':       args.data,
        'output':     str(output_dir),
        'timestamp':  t0.isoformat(),
        'training_time_sec': elapsed.total_seconds(),
        'num_params': model.count_params(),
        'best_val_accuracy': float(max(history.history['val_accuracy'])),
    }
    with open(output_dir / 'experiment_config.json', 'w') as f:
        json.dump(config, f, indent=2)

    # ── Plot learning curves ──────────────────────────────────
    _plot_history(history, output_dir, args.variant)

    print(f'\nBest val accuracy: {config["best_val_accuracy"]:.4f}')
    print(f'All outputs in: {output_dir}')
    return model, history


def _plot_history(history, output_dir: Path, variant: str):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    hist = history.history
    epochs = range(1, len(hist['loss']) + 1)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    axes[0, 0].plot(epochs, hist['accuracy'],     label='Train')
    axes[0, 0].plot(epochs, hist['val_accuracy'], label='Val')
    axes[0, 0].set_title(f'MedViT V2 {variant.upper()} — Accuracy')
    axes[0, 0].set_xlabel('Epoch'); axes[0, 0].set_ylabel('Accuracy')
    axes[0, 0].legend(); axes[0, 0].grid(True, alpha=0.3)

    axes[0, 1].plot(epochs, hist['loss'],     label='Train')
    axes[0, 1].plot(epochs, hist['val_loss'], label='Val')
    axes[0, 1].set_title('Loss')
    axes[0, 1].set_xlabel('Epoch'); axes[0, 1].set_ylabel('Loss')
    axes[0, 1].legend(); axes[0, 1].grid(True, alpha=0.3)

    if 'auc' in hist:
        axes[1, 0].plot(epochs, hist['auc'],     label='Train')
        axes[1, 0].plot(epochs, hist['val_auc'], label='Val')
        axes[1, 0].set_title('AUC')
        axes[1, 0].set_xlabel('Epoch'); axes[1, 0].set_ylabel('AUC')
        axes[1, 0].legend(); axes[1, 0].grid(True, alpha=0.3)

    axes[1, 1].plot(epochs, hist['precision'],     label='Precision-Train')
    axes[1, 1].plot(epochs, hist['val_precision'], label='Precision-Val')
    axes[1, 1].plot(epochs, hist['recall'],        label='Recall-Train',    linestyle='--')
    axes[1, 1].plot(epochs, hist['val_recall'],    label='Recall-Val',      linestyle='--')
    axes[1, 1].set_title('Precision & Recall')
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].legend(fontsize=8); axes[1, 1].grid(True, alpha=0.3)

    plt.suptitle(f'MedViT V2 {variant.upper()} — Training Curves', fontsize=14)
    plt.tight_layout()
    fig.savefig(output_dir / 'training_curves.png', dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f'Training curves saved.')


# ══════════════════════════════════════════════════════════════════════
# CLI
# ══════════════════════════════════════════════════════════════════════

def parse_args():
    p = argparse.ArgumentParser(description='Train MedViT V2 for brain tumor classification')
    p.add_argument('--data',       default=DATA_PATH,
                   help='Path to 4-class dataset directory')
    p.add_argument('--variant',    default='tiny',
                   choices=['tiny', 'small', 'base', 'large'],
                   help='MedViT V2 variant (default: tiny)')
    p.add_argument('--epochs',     type=int,   default=100)
    p.add_argument('--batch_size', type=int,   default=32,
                   help='Batch size (paper uses 128, reduce if GPU limited)')
    p.add_argument('--lr',         type=float, default=1e-3)
    p.add_argument('--dropout',    type=float, default=0.1)
    p.add_argument('--patience',   type=int,   default=20)
    p.add_argument('--seed',       type=int,   default=42)
    p.add_argument('--output',     default=None,
                   help='Output directory (auto-generated if not set)')
    return p.parse_args()


if __name__ == '__main__':
    args = parse_args()
    if args.output is None:
        ts = datetime.now().strftime('%Y%m%d_%H%M%S')
        args.output = f'medvit_v2_{args.variant}_experiment_{ts}'
    train(args)
