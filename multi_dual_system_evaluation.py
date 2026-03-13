#!/usr/bin/env python3
"""
Multi-Dual System Evaluation Script
=====================================

Comprehensive evaluation for the Hybrid 3-Class + 4-Class Brain Tumor Classification.

Author: Yannis Balasis
Date: March 2026
"""

import os
import sys
import argparse
import json
from datetime import datetime
import numpy as np
import tensorflow as tf
from pathlib import Path
from sklearn.metrics import (
    classification_report, confusion_matrix,
    precision_recall_fscore_support
)
import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow.keras.preprocessing.image import ImageDataGenerator

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from multi_dual_system_architecture import MultiDualSystemArchitecture


class MultiDualSystemEvaluator:
    """Comprehensive evaluation system for Multi-Dual Branch Architecture."""

    def __init__(
        self,
        model_path: str,
        data_path: str,
        test_data_path: str = None,
        config_path: str = None
    ):
        self.model_path = Path(model_path)
        self.data_path = Path(data_path)
        self.test_data_path = Path(test_data_path) if test_data_path else None
        self.config_path = Path(config_path) if config_path else None

        self.load_config()

        self.class_names_4 = ['glioma', 'meningioma', 'no_tumor', 'pituitary']
        self.class_names_3 = ['glioma', 'meningioma', 'pituitary']
        self.map_4_to_3 = {0: 0, 1: 1, 2: -1, 3: 2}

        print("Multi-Dual System Evaluator Initialized")
        print(f"   Model: {self.model_path}")
        print(f"   Data: {self.data_path}")

    # ------------------------------------------------------------------
    # Setup
    # ------------------------------------------------------------------

    def load_config(self):
        if self.config_path and self.config_path.exists():
            with open(self.config_path, 'r') as f:
                self.config = json.load(f)
        else:
            config_file = self.model_path.parent / "config.json"
            if config_file.exists():
                with open(config_file, 'r') as f:
                    self.config = json.load(f)
                print(f"Configuration loaded from: {config_file}")
            else:
                self.config = {}
                print("No configuration file found, using defaults")

    def load_model(self):
        print("\nLoading model...")
        try:
            # Custom objects needed for masked loss/accuracy
            custom_objects = {
                '_masked_sparse_crossentropy': None,
                '_masked_sparse_accuracy': None
            }

            # Build a temporary architecture instance to get the custom functions
            arch = MultiDualSystemArchitecture()
            custom_objects = {
                '_masked_sparse_crossentropy': arch._masked_sparse_crossentropy,
                '_masked_sparse_accuracy': arch._masked_sparse_accuracy
            }

            self.model = tf.keras.models.load_model(
                self.model_path,
                custom_objects=custom_objects
            )
            print("Model loaded successfully")
            print(f"   Model outputs: {list(self.model.output.keys())}")
        except Exception as e:
            print(f"Error loading model: {e}")
            import traceback
            traceback.print_exc()
            return False
        return True

    # ------------------------------------------------------------------
    # Data
    # ------------------------------------------------------------------

    def prepare_test_data(self):
        print("\nPreparing test data...")

        test_datagen = ImageDataGenerator(rescale=1.0 / 255.0)

        if self.test_data_path and self.test_data_path.exists():
            generator = test_datagen.flow_from_directory(
                self.test_data_path,
                target_size=(224, 224),
                batch_size=1,
                class_mode='sparse',
                classes=self.class_names_4,
                shuffle=False
            )
        else:
            test_datagen_split = ImageDataGenerator(
                rescale=1.0 / 255.0,
                validation_split=0.1
            )
            generator = test_datagen_split.flow_from_directory(
                self.data_path,
                target_size=(224, 224),
                batch_size=1,
                class_mode='sparse',
                classes=self.class_names_4,
                subset='validation',
                shuffle=False,
                seed=789
            )

        print(f"   Test samples: {len(generator.filenames)}")

        self.test_data, self.test_labels_4class = [], []
        for i in range(len(generator)):
            batch_x, batch_y = generator[i]
            self.test_data.append(batch_x[0])
            self.test_labels_4class.append(int(batch_y[0]))

        self.test_data = np.array(self.test_data)
        self.test_labels_4class = np.array(self.test_labels_4class, dtype=np.int32)

        # Build 3-class subset (exclude no_tumor)
        self.test_indices_3class = [
            i for i, lbl in enumerate(self.test_labels_4class) if lbl != 2
        ]
        self.test_labels_3class = np.array(
            [self.map_4_to_3[self.test_labels_4class[i]] for i in self.test_indices_3class],
            dtype=np.int32
        )
        self.test_data_3class = self.test_data[self.test_indices_3class]

        print(f"   4-class samples: {len(self.test_labels_4class)}")
        print(f"   3-class samples (tumor only): {len(self.test_labels_3class)}")

        unique_4, counts_4 = np.unique(self.test_labels_4class, return_counts=True)
        print(f"   4-class distribution: {dict(zip(unique_4.tolist(), counts_4.tolist()))}")

    # ------------------------------------------------------------------
    # Evaluation
    # ------------------------------------------------------------------

    def evaluate_model(self):
        print("\nRunning predictions...")
        predictions = self.model.predict(self.test_data, verbose=1)

        self.pred_branch1_probs = predictions['branch1_3class']
        self.pred_branch2_probs = predictions['branch2_4class']
        self.pred_fusion_probs = predictions['fusion_output']

        self.pred_branch1_classes = np.argmax(self.pred_branch1_probs, axis=1)
        self.pred_branch2_classes = np.argmax(self.pred_branch2_probs, axis=1)
        self.pred_fusion_classes = np.argmax(self.pred_fusion_probs, axis=1)

        print("Predictions completed")

    def calculate_metrics(self):
        print("\nCalculating metrics...")
        self.metrics = {}

        # --- Branch 1 (3-class, tumor samples only) ---
        if len(self.test_labels_3class) > 0:
            pred_3 = self.pred_branch1_classes[self.test_indices_3class]
            cm_3 = confusion_matrix(self.test_labels_3class, pred_3)
            self.metrics['branch1_3class'] = {
                'accuracy': float(np.mean(pred_3 == self.test_labels_3class)),
                'top2_accuracy': self._top2_accuracy(
                    self.pred_branch1_probs[self.test_indices_3class],
                    self.test_labels_3class
                ),
                'classification_report': classification_report(
                    self.test_labels_3class, pred_3,
                    target_names=self.class_names_3,
                    output_dict=True
                ),
                'confusion_matrix': cm_3,
                'per_class_accuracy': self._per_class_accuracy(cm_3)
            }

        # --- Branch 2 (4-class) ---
        cm_4 = confusion_matrix(self.test_labels_4class, self.pred_branch2_classes)
        self.metrics['branch2_4class'] = {
            'accuracy': float(np.mean(self.pred_branch2_classes == self.test_labels_4class)),
            'top2_accuracy': self._top2_accuracy(self.pred_branch2_probs, self.test_labels_4class),
            'classification_report': classification_report(
                self.test_labels_4class, self.pred_branch2_classes,
                target_names=self.class_names_4,
                output_dict=True
            ),
            'confusion_matrix': cm_4,
            'per_class_accuracy': self._per_class_accuracy(cm_4)
        }

        # --- Fusion (4-class) ---
        cm_f = confusion_matrix(self.test_labels_4class, self.pred_fusion_classes)
        self.metrics['fusion'] = {
            'accuracy': float(np.mean(self.pred_fusion_classes == self.test_labels_4class)),
            'top2_accuracy': self._top2_accuracy(self.pred_fusion_probs, self.test_labels_4class),
            'classification_report': classification_report(
                self.test_labels_4class, self.pred_fusion_classes,
                target_names=self.class_names_4,
                output_dict=True
            ),
            'confusion_matrix': cm_f,
            'per_class_accuracy': self._per_class_accuracy(cm_f)
        }

        # --- Binary tumor detection ---
        binary_true = (self.test_labels_4class != 2).astype(int)
        binary_pred_b2 = (self.pred_branch2_classes != 2).astype(int)
        binary_pred_f = (self.pred_fusion_classes != 2).astype(int)

        def detection_metrics(pred, true):
            tp = np.sum((pred == 1) & (true == 1))
            tn = np.sum((pred == 0) & (true == 0))
            fp = np.sum((pred == 1) & (true == 0))
            fn = np.sum((pred == 0) & (true == 1))
            sensitivity = tp / (tp + fn + 1e-8)
            specificity = tn / (tn + fp + 1e-8)
            accuracy = np.mean(pred == true)
            return float(accuracy), float(sensitivity), float(specificity)

        b2_acc, b2_sens, b2_spec = detection_metrics(binary_pred_b2, binary_true)
        f_acc, f_sens, f_spec = detection_metrics(binary_pred_f, binary_true)

        self.metrics['binary_detection'] = {
            'branch2_accuracy': b2_acc,
            'branch2_sensitivity': b2_sens,
            'branch2_specificity': b2_spec,
            'fusion_accuracy': f_acc,
            'fusion_sensitivity': f_sens,
            'fusion_specificity': f_spec
        }

        print("Metrics calculated")

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _top2_accuracy(self, probs, true_labels):
        correct = 0
        for i, true in enumerate(true_labels):
            top2 = np.argsort(probs[i])[-2:]
            if true in top2:
                correct += 1
        return float(correct / len(true_labels))

    def _per_class_accuracy(self, cm):
        accs = []
        for i in range(cm.shape[0]):
            total = cm[i].sum()
            accs.append(float(cm[i, i] / total) if total > 0 else 0.0)
        return accs

    # ------------------------------------------------------------------
    # Visualizations
    # ------------------------------------------------------------------

    def create_visualizations(self, output_dir: Path):
        print("\nCreating visualizations...")
        plt.style.use('default')

        # 1. Confusion matrices
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))

        if 'branch1_3class' in self.metrics:
            sns.heatmap(self.metrics['branch1_3class']['confusion_matrix'],
                        annot=True, fmt='d', cmap='Blues',
                        xticklabels=self.class_names_3, yticklabels=self.class_names_3,
                        ax=axes[0])
            axes[0].set_title('Branch 1: 3-Class Confusion Matrix', fontweight='bold')
            axes[0].set_xlabel('Predicted')
            axes[0].set_ylabel('True')

        sns.heatmap(self.metrics['branch2_4class']['confusion_matrix'],
                    annot=True, fmt='d', cmap='Greens',
                    xticklabels=self.class_names_4, yticklabels=self.class_names_4,
                    ax=axes[1])
        axes[1].set_title('Branch 2: 4-Class Confusion Matrix', fontweight='bold')
        axes[1].set_xlabel('Predicted')
        axes[1].set_ylabel('True')

        sns.heatmap(self.metrics['fusion']['confusion_matrix'],
                    annot=True, fmt='d', cmap='Purples',
                    xticklabels=self.class_names_4, yticklabels=self.class_names_4,
                    ax=axes[2])
        axes[2].set_title('Fusion: 4-Class Confusion Matrix', fontweight='bold')
        axes[2].set_xlabel('Predicted')
        axes[2].set_ylabel('True')

        plt.tight_layout()
        plt.savefig(output_dir / "confusion_matrices.png", dpi=300, bbox_inches='tight')
        plt.close()

        # 2. Accuracy comparison
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

        accuracies, labels = [], []
        if 'branch1_3class' in self.metrics:
            accuracies.append(self.metrics['branch1_3class']['accuracy'])
            labels.append('Branch 1\n(3-Class)')
        accuracies += [self.metrics['branch2_4class']['accuracy'], self.metrics['fusion']['accuracy']]
        labels += ['Branch 2\n(4-Class)', 'Fusion\n(4-Class)']

        colors = ['#1f77b4', '#2ca02c', '#9467bd'][:len(accuracies)]
        bars = ax1.bar(labels, accuracies, color=colors)
        ax1.set_title('Overall Accuracy Comparison', fontweight='bold')
        ax1.set_ylabel('Accuracy')
        ax1.set_ylim(0, 1)
        for bar, acc in zip(bars, accuracies):
            ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                     f'{acc:.3f}', ha='center', va='bottom', fontweight='bold')

        top2_accuracies = [self.metrics[k]['top2_accuracy']
                           for k in ['branch1_3class', 'branch2_4class', 'fusion']
                           if k in self.metrics]
        top2_labels = [l for l, k in zip(labels, ['branch1_3class', 'branch2_4class', 'fusion'])
                       if k in self.metrics]
        top2_colors = ['#ff7f0e', '#d62728', '#8c564b'][:len(top2_accuracies)]
        bars2 = ax2.bar(top2_labels, top2_accuracies, color=top2_colors)
        ax2.set_title('Top-2 Accuracy Comparison', fontweight='bold')
        ax2.set_ylabel('Top-2 Accuracy')
        ax2.set_ylim(0, 1)
        for bar, acc in zip(bars2, top2_accuracies):
            ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                     f'{acc:.3f}', ha='center', va='bottom', fontweight='bold')

        plt.tight_layout()
        plt.savefig(output_dir / "accuracy_comparison.png", dpi=300, bbox_inches='tight')
        plt.close()

        # 3. Per-class accuracy
        fig, ax = plt.subplots(figsize=(12, 6))
        x = np.arange(len(self.class_names_4))
        width = 0.35
        b2_acc = self.metrics['branch2_4class']['per_class_accuracy']
        f_acc = self.metrics['fusion']['per_class_accuracy']
        ax.bar(x - width / 2, b2_acc, width, label='Branch 2 (4-Class)', color='#2ca02c')
        ax.bar(x + width / 2, f_acc, width, label='Fusion', color='#9467bd')
        ax.set_title('Per-Class Accuracy Comparison', fontweight='bold')
        ax.set_ylabel('Accuracy')
        ax.set_xlabel('Class')
        ax.set_xticks(x)
        ax.set_xticklabels([n.title() for n in self.class_names_4])
        ax.legend()
        ax.set_ylim(0, 1)
        for i, (a2, af) in enumerate(zip(b2_acc, f_acc)):
            ax.text(i - width / 2, a2 + 0.01, f'{a2:.2f}', ha='center', va='bottom')
            ax.text(i + width / 2, af + 0.01, f'{af:.2f}', ha='center', va='bottom')
        plt.tight_layout()
        plt.savefig(output_dir / "per_class_accuracy.png", dpi=300, bbox_inches='tight')
        plt.close()

        # 4. Binary detection
        fig, ax = plt.subplots(figsize=(10, 6))
        metrics_names = ['Accuracy', 'Sensitivity\n(Recall)', 'Specificity']
        b2_vals = [self.metrics['binary_detection']['branch2_accuracy'],
                   self.metrics['binary_detection']['branch2_sensitivity'],
                   self.metrics['binary_detection']['branch2_specificity']]
        f_vals = [self.metrics['binary_detection']['fusion_accuracy'],
                  self.metrics['binary_detection']['fusion_sensitivity'],
                  self.metrics['binary_detection']['fusion_specificity']]
        x = np.arange(len(metrics_names))
        ax.bar(x - width / 2, b2_vals, width, label='Branch 2', color='#2ca02c')
        ax.bar(x + width / 2, f_vals, width, label='Fusion', color='#9467bd')
        ax.set_title('Binary Tumor Detection Performance', fontweight='bold')
        ax.set_ylabel('Score')
        ax.set_xticks(x)
        ax.set_xticklabels(metrics_names)
        ax.legend()
        ax.set_ylim(0, 1)
        for i, (b, f) in enumerate(zip(b2_vals, f_vals)):
            ax.text(i - width / 2, b + 0.01, f'{b:.3f}', ha='center', va='bottom')
            ax.text(i + width / 2, f + 0.01, f'{f:.3f}', ha='center', va='bottom')
        plt.tight_layout()
        plt.savefig(output_dir / "binary_detection_performance.png", dpi=300, bbox_inches='tight')
        plt.close()

        print(f"Visualizations saved to {output_dir}")

    # ------------------------------------------------------------------
    # Report
    # ------------------------------------------------------------------

    def generate_comprehensive_report(self, output_dir: Path):
        print("\nGenerating evaluation report...")

        lines = []
        lines.append("=" * 80)
        lines.append("MULTI-DUAL SYSTEM EVALUATION REPORT")
        lines.append("=" * 80)
        lines.append("")

        lines.append("MODEL INFORMATION")
        lines.append("-" * 50)
        lines.append(f"Model Path: {self.model_path}")
        lines.append(f"Data Path: {self.data_path}")
        if self.config:
            lines.append(f"Backbone Type: {self.config.get('backbone_type', 'N/A')}")
            lines.append(f"Fusion Strategy: {self.config.get('fusion_strategy', 'N/A')}")
            lines.append(f"Seed: {self.config.get('seed', 'N/A')}")
        lines.append("")

        lines.append("DATASET INFORMATION")
        lines.append("-" * 50)
        lines.append(f"Total Test Samples: {len(self.test_labels_4class)}")
        lines.append(f"3-Class Tumor Samples: {len(self.test_labels_3class)}")
        unique_4, counts_4 = np.unique(self.test_labels_4class, return_counts=True)
        lines.append("\n4-Class Distribution:")
        for ci, cnt in zip(unique_4, counts_4):
            pct = cnt / len(self.test_labels_4class) * 100
            lines.append(f"  {self.class_names_4[ci].title()}: {cnt} ({pct:.1f}%)")
        lines.append("")

        lines.append("PERFORMANCE RESULTS")
        lines.append("-" * 50)

        if 'branch1_3class' in self.metrics:
            m = self.metrics['branch1_3class']
            lines.append("Branch 1 (3-Class Tumor Classification):")
            lines.append(f"  Overall Accuracy: {m['accuracy']:.4f} ({m['accuracy']*100:.2f}%)")
            lines.append(f"  Top-2 Accuracy:   {m['top2_accuracy']:.4f} ({m['top2_accuracy']*100:.2f}%)")
            lines.append("  Per-Class:")
            for i, cn in enumerate(self.class_names_3):
                cr = m['classification_report'].get(cn, {})
                lines.append(
                    f"    {cn.title()}: Acc={m['per_class_accuracy'][i]:.3f}, "
                    f"P={cr.get('precision', 0):.3f}, R={cr.get('recall', 0):.3f}, "
                    f"F1={cr.get('f1-score', 0):.3f}"
                )
            lines.append("")

        for branch_key, branch_label, class_names in [
            ('branch2_4class', 'Branch 2 (4-Class Full Classification)', self.class_names_4),
            ('fusion', 'Fusion Output (Primary Classification)', self.class_names_4)
        ]:
            m = self.metrics[branch_key]
            lines.append(f"{branch_label}:")
            lines.append(f"  Overall Accuracy: {m['accuracy']:.4f} ({m['accuracy']*100:.2f}%)")
            lines.append(f"  Top-2 Accuracy:   {m['top2_accuracy']:.4f} ({m['top2_accuracy']*100:.2f}%)")
            lines.append("  Per-Class:")
            for i, cn in enumerate(class_names):
                cr = m['classification_report'].get(cn, {})
                lines.append(
                    f"    {cn.title()}: Acc={m['per_class_accuracy'][i]:.3f}, "
                    f"P={cr.get('precision', 0):.3f}, R={cr.get('recall', 0):.3f}, "
                    f"F1={cr.get('f1-score', 0):.3f}"
                )
            lines.append("")

        bd = self.metrics['binary_detection']
        lines.append("Binary Tumor Detection:")
        lines.append(f"  Branch 2 - Accuracy: {bd['branch2_accuracy']:.4f}, "
                     f"Sensitivity: {bd['branch2_sensitivity']:.4f}, "
                     f"Specificity: {bd['branch2_specificity']:.4f}")
        lines.append(f"  Fusion    - Accuracy: {bd['fusion_accuracy']:.4f}, "
                     f"Sensitivity: {bd['fusion_sensitivity']:.4f}, "
                     f"Specificity: {bd['fusion_specificity']:.4f}")
        lines.append("")

        lines.append("CLINICAL ASSESSMENT")
        lines.append("-" * 50)
        fusion_acc = self.metrics['fusion']['accuracy']
        if fusion_acc >= 0.95:
            grade = "EXCELLENT - Clinical Grade Performance"
        elif fusion_acc >= 0.90:
            grade = "GOOD - Research Grade Performance"
        elif fusion_acc >= 0.85:
            grade = "MODERATE - Needs Improvement"
        else:
            grade = "POOR - Requires Significant Enhancement"
        lines.append(f"Grade: {grade}")

        worst_idx = int(np.argmin(self.metrics['fusion']['per_class_accuracy']))
        lines.append(f"Most Challenging Class: {self.class_names_4[worst_idx].title()} "
                     f"({self.metrics['fusion']['per_class_accuracy'][worst_idx]:.1%})")

        if bd['fusion_sensitivity'] < 0.90:
            lines.append(f"WARNING: Low tumor sensitivity ({bd['fusion_sensitivity']:.1%}) - risk of false negatives")
        if bd['fusion_specificity'] < 0.90:
            lines.append(f"WARNING: Low specificity ({bd['fusion_specificity']:.1%}) - risk of false positives")
        lines.append("")

        lines.append("COMPARISON WITH EXISTING MODELS")
        lines.append("-" * 50)
        lines.append("  1. Binary CNN:            99.57%")
        lines.append("  2. 4-Class CNN:           98.01%")
        lines.append("  3. EfficientNet-B3 Dual:  91.27%")
        lines.append("  4. ResNet-50 Dual:        85.45%")
        lines.append(f"  5. Multi-Dual System:     {fusion_acc:.2%}  <-- CURRENT")
        lines.append("")

        if fusion_acc > 0.9127:
            lines.append("Result: Outperformed all previous dual systems.")
        elif fusion_acc > 0.8545:
            lines.append("Result: Better than ResNet-50 dual system.")
        else:
            lines.append("Result: Below previous dual systems - further tuning needed.")

        lines.append("")
        lines.append("=" * 80)

        report_text = "\n".join(lines)
        with open(output_dir / "evaluation_report.txt", 'w') as f:
            f.write(report_text)

        print(f"Report saved to: {output_dir}/evaluation_report.txt")
        return report_text

    def save_detailed_results(self, output_dir: Path):
        results = {
            'model_info': {
                'model_path': str(self.model_path),
                'data_path': str(self.data_path),
                'config': self.config if hasattr(self, 'config') else {}
            },
            'dataset_info': {
                'total_test_samples': len(self.test_labels_4class),
                'test_samples_3class': len(self.test_labels_3class)
            },
            'metrics': {}
        }

        for branch_name, branch_metrics in self.metrics.items():
            results['metrics'][branch_name] = {}
            for k, v in branch_metrics.items():
                if isinstance(v, np.ndarray):
                    results['metrics'][branch_name][k] = v.tolist()
                elif isinstance(v, (np.integer, np.floating)):
                    results['metrics'][branch_name][k] = float(v)
                else:
                    results['metrics'][branch_name][k] = v

        with open(output_dir / "detailed_results.json", 'w') as f:
            json.dump(results, f, indent=2, default=str)

        print(f"Detailed results saved to: {output_dir}/detailed_results.json")

    # ------------------------------------------------------------------
    # Full pipeline
    # ------------------------------------------------------------------

    def run_full_evaluation(self, output_dir: str = None):
        if output_dir:
            eval_dir = Path(output_dir)
        else:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            eval_dir = Path(f"multi_dual_evaluation_{timestamp}")

        eval_dir.mkdir(parents=True, exist_ok=True)

        print(f"\nStarting Multi-Dual System Evaluation")
        print(f"Output directory: {eval_dir}")

        if not self.load_model():
            return None

        self.prepare_test_data()
        self.evaluate_model()
        self.calculate_metrics()
        self.create_visualizations(eval_dir)
        self.generate_comprehensive_report(eval_dir)
        self.save_detailed_results(eval_dir)

        print(f"\nEvaluation completed successfully!")
        print(f"\nSUMMARY:")
        if 'branch1_3class' in self.metrics:
            print(f"   Branch 1 (3-Class): {self.metrics['branch1_3class']['accuracy']:.2%}")
        print(f"   Branch 2 (4-Class): {self.metrics['branch2_4class']['accuracy']:.2%}")
        print(f"   Fusion (Primary):   {self.metrics['fusion']['accuracy']:.2%}")
        print(f"   Binary Detection:   {self.metrics['binary_detection']['fusion_accuracy']:.2%}")

        return str(eval_dir)


# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description='Evaluate Multi-Dual System')
    parser.add_argument('--model', type=str, required=True)
    parser.add_argument('--data', type=str, required=True)
    parser.add_argument('--test_data', type=str, default=None)
    parser.add_argument('--config', type=str, default=None)
    parser.add_argument('--output', type=str, default=None)

    args = parser.parse_args()

    if not os.path.exists(args.model):
        print(f"Error: Model file {args.model} does not exist")
        return
    if not os.path.exists(args.data):
        print(f"Error: Data directory {args.data} does not exist")
        return

    evaluator = MultiDualSystemEvaluator(
        model_path=args.model,
        data_path=args.data,
        test_data_path=args.test_data,
        config_path=args.config
    )

    try:
        result_dir = evaluator.run_full_evaluation(args.output)
        if result_dir:
            print(f"\nResults available at: {result_dir}")
        else:
            print("Evaluation failed!")
    except Exception as e:
        print(f"Evaluation error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()