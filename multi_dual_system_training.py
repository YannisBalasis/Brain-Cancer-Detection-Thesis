#!/usr/bin/env python3
"""
Multi-Dual System Training Script
==================================

Training pipeline for the Hybrid 3-Class + 4-Class Brain Tumor Classification System.

Author: Yannis Balasis
Date: March 2026
"""

import os
import sys
import argparse
import json
import time
from datetime import datetime
from pathlib import Path
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau, CSVLogger
from sklearn.utils.class_weight import compute_class_weight
import matplotlib.pyplot as plt

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from multi_dual_system_architecture import MultiDualSystemArchitecture


class MultiDualSystemTrainer:
    """Training system for Multi-Dual Branch Architecture."""

    def __init__(
        self,
        data_path: str,
        backbone_type: str = "efficient",
        fusion_strategy: str = "weighted_ensemble",
        seed: int = 789
    ):
        self.data_path = Path(data_path)
        self.backbone_type = backbone_type
        self.fusion_strategy = fusion_strategy
        self.seed = seed

        self.image_size = (224, 224)
        self.batch_size = 32
        self.epochs = 50
        self.validation_split = 0.2
        self.test_split = 0.1

        tf.random.set_seed(seed)
        np.random.seed(seed)

        self.class_names_4 = ['glioma', 'meningioma', 'no_tumor', 'pituitary']
        self.class_names_3 = ['glioma', 'meningioma', 'pituitary']

        # no_tumor (index 2) maps to -1 for branch1; masked loss will ignore it
        self.map_4_to_3 = {0: 0, 1: 1, 2: -1, 3: 2}

        print("Multi-Dual System Trainer Initialized")
        print(f"   Data Path: {self.data_path}")
        print(f"   Backbone: {self.backbone_type}")
        print(f"   Fusion: {self.fusion_strategy}")
        print(f"   Seed: {self.seed}")

    # ------------------------------------------------------------------
    # Data preparation
    # ------------------------------------------------------------------

    def prepare_datasets(self):
        """Prepare training and validation datasets."""
        print("\nPreparing datasets...")

        train_datagen = ImageDataGenerator(
            rescale=1.0 / 255.0,
            rotation_range=20,
            width_shift_range=0.15,
            height_shift_range=0.15,
            horizontal_flip=True,
            zoom_range=0.1,
            shear_range=0.1,
            brightness_range=[0.9, 1.1],
            fill_mode='nearest',
            validation_split=self.validation_split + self.test_split
        )

        val_datagen = ImageDataGenerator(
            rescale=1.0 / 255.0,
            validation_split=self.validation_split + self.test_split
        )

        self.train_generator = train_datagen.flow_from_directory(
            self.data_path,
            target_size=self.image_size,
            batch_size=self.batch_size,
            class_mode='sparse',
            classes=self.class_names_4,
            subset='training',
            seed=self.seed,
            shuffle=True
        )

        self.val_generator = val_datagen.flow_from_directory(
            self.data_path,
            target_size=self.image_size,
            batch_size=self.batch_size,
            class_mode='sparse',
            classes=self.class_names_4,
            subset='validation',
            seed=self.seed,
            shuffle=False
        )

        print(f"   Training samples: {len(self.train_generator.filenames)}")
        print(f"   Validation samples: {len(self.val_generator.filenames)}")

        unique_classes, counts = np.unique(self.train_generator.classes, return_counts=True)
        self.class_distribution = dict(zip(unique_classes.tolist(), counts.tolist()))
        print(f"   Class distribution: {self.class_distribution}")

    def compute_class_weights(self):
        """Compute class weights for balanced training."""
        unique_classes_4 = np.unique(self.train_generator.classes)

        weights_4_present = compute_class_weight(
            'balanced',
            classes=unique_classes_4,
            y=self.train_generator.classes
        )

        self.class_weights_4 = {}
        for i in range(len(self.class_names_4)):
            if i in unique_classes_4:
                idx = np.where(unique_classes_4 == i)[0][0]
                self.class_weights_4[i] = float(weights_4_present[idx])
            else:
                self.class_weights_4[i] = 1.0

        # 3-class weights: exclude no_tumor samples (label 2)
        tumor_mask = self.train_generator.classes != 2
        tumor_labels_4 = self.train_generator.classes[tumor_mask]
        tumor_labels_3 = np.array([self.map_4_to_3[l] for l in tumor_labels_4])

        if len(tumor_labels_3) > 0:
            unique_tumor = np.unique(tumor_labels_3)
            weights_3_present = compute_class_weight(
                'balanced', classes=unique_tumor, y=tumor_labels_3
            )
            self.class_weights_3 = {}
            for i in range(len(self.class_names_3)):
                if i in unique_tumor:
                    idx = np.where(unique_tumor == i)[0][0]
                    self.class_weights_3[i] = float(weights_3_present[idx])
                else:
                    self.class_weights_3[i] = 1.0
        else:
            self.class_weights_3 = {i: 1.0 for i in range(len(self.class_names_3))}

        print(f"\nClass weights computed:")
        print(f"   4-class weights: {self.class_weights_4}")
        print(f"   3-class weights: {self.class_weights_3}")

    # ------------------------------------------------------------------
    # Model
    # ------------------------------------------------------------------

    def create_model(self):
        """Create and compile the multi-dual system model."""
        print("\nCreating Multi-Dual System model...")

        self.model_system = MultiDualSystemArchitecture(
            input_shape=(*self.image_size, 3),
            backbone_type=self.backbone_type,
            fusion_strategy=self.fusion_strategy,
            seed=self.seed
        )

        self.model_system.compile_model(
            learning_rate=0.001,
            branch1_weight=0.3,
            branch2_weight=0.4,
            fusion_weight=0.3
        )

        self.model = self.model_system.model

        print(f"Model created and compiled")
        print(f"   Total parameters: {self.model.count_params():,}")

    # ------------------------------------------------------------------
    # Callbacks
    # ------------------------------------------------------------------

    def setup_callbacks(self, experiment_dir: Path):
        callbacks = []

        callbacks.append(ModelCheckpoint(
            str(experiment_dir / "best_multi_dual_model.h5"),
            monitor='val_fusion_output_fusion_accuracy',
            save_best_only=True,
            save_weights_only=False,
            mode='max',
            verbose=1
        ))

        callbacks.append(EarlyStopping(
            monitor='val_fusion_output_fusion_accuracy',
            patience=10,
            restore_best_weights=True,
            mode='max',
            verbose=1
        ))

        callbacks.append(ReduceLROnPlateau(
            monitor='val_fusion_output_fusion_accuracy',
            factor=0.5,
            patience=5,
            min_lr=1e-7,
            mode='max',
            verbose=1
        ))

        callbacks.append(CSVLogger(
            str(experiment_dir / "training_history.csv"),
            append=False
        ))

        return callbacks

    # ------------------------------------------------------------------
    # Multi-task data
    # ------------------------------------------------------------------

    def create_multi_task_data(self):
        """
        Load all images into memory and build label dictionaries.

        Branch 1 labels: -1 for no_tumor samples (masked loss ignores them).
        Branch 2 and fusion labels: standard 4-class labels.
        """
        print("\nPreparing multi-task data...")

        def load_generator(generator):
            images, labels_4 = [], []
            steps = len(generator)
            for i in range(steps):
                batch_x, batch_y = generator[i]
                images.extend(batch_x)
                labels_4.extend(batch_y.astype(int).tolist())
            return np.array(images), np.array(labels_4, dtype=np.int32)

        train_x, train_labels_4 = load_generator(self.train_generator)
        val_x, val_labels_4 = load_generator(self.val_generator)

        # Branch 1 labels: map no_tumor -> -1
        train_labels_3 = np.array([self.map_4_to_3[l] for l in train_labels_4], dtype=np.int32)
        val_labels_3 = np.array([self.map_4_to_3[l] for l in val_labels_4], dtype=np.int32)

        print(f"   Training samples: {len(train_x)}")
        print(f"   Validation samples: {len(val_x)}")
        print(f"   Branch1 train label distribution: {dict(zip(*np.unique(train_labels_3, return_counts=True)))}")

        train_y = {
            'branch1_3class': train_labels_3,
            'branch2_4class': train_labels_4,
            'fusion_output': train_labels_4
        }
        val_y = {
            'branch1_3class': val_labels_3,
            'branch2_4class': val_labels_4,
            'fusion_output': val_labels_4
        }

        return train_x, train_y, val_x, val_y

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def train_model(self, output_dir: str = None):
        """Train the multi-dual system model."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        experiment_name = f"multi_dual_system_experiment_{timestamp}"

        if output_dir:
            experiment_dir = Path(output_dir) / experiment_name
        else:
            experiment_dir = Path(experiment_name)

        experiment_dir.mkdir(parents=True, exist_ok=True)

        print(f"\nStarting Multi-Dual System training...")
        print(f"Experiment directory: {experiment_dir}")

        self.prepare_datasets()
        self.compute_class_weights()
        self.create_model()

        callbacks = self.setup_callbacks(experiment_dir)
        train_x, train_y, val_x, val_y = self.create_multi_task_data()

        # Save configuration
        config = {
            "experiment_name": experiment_name,
            "backbone_type": self.backbone_type,
            "fusion_strategy": self.fusion_strategy,
            "seed": self.seed,
            "image_size": list(self.image_size),
            "batch_size": self.batch_size,
            "epochs": self.epochs,
            "validation_split": self.validation_split,
            "test_split": self.test_split,
            "class_names_3": self.class_names_3,
            "class_names_4": self.class_names_4,
            "class_weights_3": self.class_weights_3,
            "class_weights_4": self.class_weights_4,
            "class_distribution": {str(k): int(v) for k, v in self.class_distribution.items()},
            "model_summary": self.model_system.get_model_summary()
        }

        with open(experiment_dir / "config.json", 'w') as f:
            json.dump(config, f, indent=2)

        print(f"\nTraining configuration:")
        print(f"   Training samples: {len(train_x)}")
        print(f"   Validation samples: {len(val_x)}")
        print(f"   Batch size: {self.batch_size}")
        print(f"   Epochs: {self.epochs}")

        start_time = time.time()

        try:
            history = self.model.fit(
                train_x, train_y,
                validation_data=(val_x, val_y),
                epochs=self.epochs,
                batch_size=self.batch_size,
                callbacks=callbacks,
                verbose=1
            )
        except Exception as e:
            print(f"Training failed: {e}")
            import traceback
            traceback.print_exc()
            return None, None

        training_time = time.time() - start_time
        print(f"\nTraining completed!")
        print(f"Total training time: {training_time / 3600:.2f} hours")

        # Save final model
        self.model.save(experiment_dir / "multi_dual_final_model.h5")

        # Save training history
        history_dict = {k: [float(x) for x in v] for k, v in history.history.items()}
        with open(experiment_dir / "training_history.json", 'w') as f:
            json.dump(history_dict, f, indent=2)

        self.plot_training_history(history, experiment_dir)

        final_metrics = self.get_final_performance(history)
        print(f"\nFinal Training Performance:")
        for metric, value in final_metrics.items():
            print(f"   {metric}: {value:.4f}")

        return str(experiment_dir), str(experiment_dir / "best_multi_dual_model.h5")

    # ------------------------------------------------------------------
    # Plotting & utilities
    # ------------------------------------------------------------------

    def plot_training_history(self, history, save_dir: Path):
        """Plot training history."""
        plt.style.use('default')
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Multi-Dual System Training History', fontsize=16, fontweight='bold')

        # Detect correct metric key names (may vary between Keras versions)
        h = history.history
        fusion_acc_key = next((k for k in h if 'fusion' in k and 'accuracy' in k and 'val' not in k), None)
        val_fusion_acc_key = next((k for k in h if 'fusion' in k and 'accuracy' in k and 'val' in k), None)
        b1_acc_key = next((k for k in h if 'branch1' in k and 'accuracy' in k and 'val' not in k), None)
        b2_acc_key = next((k for k in h if 'branch2' in k and 'accuracy' in k and 'val' not in k), None)

        if fusion_acc_key and val_fusion_acc_key:
            axes[0, 0].plot(h[fusion_acc_key], label='Train', linewidth=2, color='#1f77b4')
            axes[0, 0].plot(h[val_fusion_acc_key], label='Validation', linewidth=2, color='#ff7f0e')
        axes[0, 0].set_title('Fusion Output Accuracy', fontweight='bold')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Accuracy')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)

        if b1_acc_key and b2_acc_key:
            axes[0, 1].plot(h[b1_acc_key], label='Branch 1 (3-class)', linewidth=2, color='#2ca02c')
            axes[0, 1].plot(h[b2_acc_key], label='Branch 2 (4-class)', linewidth=2, color='#d62728')
        axes[0, 1].set_title('Branch Accuracies', fontweight='bold')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Accuracy')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)

        axes[1, 0].plot(h.get('loss', []), label='Train', linewidth=2, color='#9467bd')
        axes[1, 0].plot(h.get('val_loss', []), label='Validation', linewidth=2, color='#8c564b')
        axes[1, 0].set_title('Combined Loss', fontweight='bold')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('Loss')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)

        if 'lr' in h:
            axes[1, 1].plot(h['lr'], linewidth=2, color='#bcbd22')
            axes[1, 1].set_yscale('log')
        else:
            axes[1, 1].text(0.5, 0.5, 'Learning Rate\nNot Tracked',
                            ha='center', va='center', transform=axes[1, 1].transAxes)
        axes[1, 1].set_title('Learning Rate', fontweight='bold')
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(save_dir / "training_history.png", dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Training history saved to: {save_dir}/training_history.png")

    def get_final_performance(self, history):
        h = history.history
        final_metrics = {}
        for key in h:
            if 'val_' in key and 'accuracy' in key:
                clean = key.replace('val_', 'Best Val ')
                final_metrics[clean] = max(h[key])
        for key in ['loss', 'val_loss']:
            if key in h:
                final_metrics[key] = h[key][-1]
        return final_metrics


# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description='Train Multi-Dual System for Brain Tumor Classification')
    parser.add_argument('--data', type=str, required=True,
                        help='Path to multiclass dataset directory')
    parser.add_argument('--backbone', type=str, default='efficient',
                        choices=['efficient', 'residual', 'custom'])
    parser.add_argument('--fusion', type=str, default='weighted_ensemble',
                        choices=['weighted_ensemble', 'hierarchical', 'attention'])
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--output', type=str, default=None)
    parser.add_argument('--seed', type=int, default=789)

    args = parser.parse_args()

    if not os.path.exists(args.data):
        print(f"Error: Data path {args.data} does not exist")
        return

    print("Multi-Dual System Training")
    print("=" * 50)
    print(f"Data path: {args.data}")
    print(f"Backbone: {args.backbone}")
    print(f"Fusion strategy: {args.fusion}")
    print(f"Epochs: {args.epochs}")
    print(f"Batch size: {args.batch_size}")
    print(f"Random seed: {args.seed}")
    print("=" * 50)

    trainer = MultiDualSystemTrainer(
        data_path=args.data,
        backbone_type=args.backbone,
        fusion_strategy=args.fusion,
        seed=args.seed
    )
    trainer.epochs = args.epochs
    trainer.batch_size = args.batch_size

    try:
        experiment_dir, model_path = trainer.train_model(args.output)
        if experiment_dir and model_path:
            print(f"\nTraining completed successfully!")
            print(f"Experiment directory: {experiment_dir}")
            print(f"Best model: {model_path}")
        else:
            print("Training failed!")
    except Exception as e:
        print(f"Training error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()