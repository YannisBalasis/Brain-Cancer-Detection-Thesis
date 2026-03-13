#!/usr/bin/env python3
"""
📊 Evaluation Script για Improved Dual Branch System
Comprehensive evaluation με medical metrics και comparison
"""

import os
import sys
import argparse
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from sklearn.metrics import (
    classification_report, confusion_matrix, 
    precision_recall_fscore_support, roc_auc_score
)
import tensorflow as tf
from tensorflow import keras
import json

class ImprovedDualEvaluator:
    """Evaluator για Improved Dual Branch System"""
    
    def __init__(self, model_path, test_data_path):
        self.model_path = model_path
        self.test_data_path = test_data_path
        self.model = None
        self.class_names = ['Glioma', 'Meningioma', 'No Tumor', 'Pituitary']
        
    def load_model_and_data(self):
        """Load model και test data"""
        print("📁 Loading Improved Dual System...")
        print(f"  Model: {self.model_path}")
        print(f"  Test data: {self.test_data_path}")
        
        # Load model
        self.model = keras.models.load_model(self.model_path)
        print(f"✅ Model loaded: {self.model.count_params():,} parameters")
        
        # Load test data
        test_data = np.load(self.test_data_path)
        X_test = test_data['X_test']
        y_test = test_data['y_test']
        
        print(f"📊 Test data: {len(X_test)} samples")
        return X_test, y_test
    
    def evaluate_model(self, X_test, y_test):
        """Comprehensive model evaluation"""
        print("\n🔮 Running Improved Dual System Evaluation...")
        
        # Normalize data
        X_test_norm = X_test / 255.0
        
        # Get predictions
        y_pred_proba = self.model.predict(X_test_norm, verbose=1)
        y_pred = np.argmax(y_pred_proba, axis=1)
        y_true = np.argmax(y_test, axis=1)
        
        # Overall metrics
        test_metrics = self.model.evaluate(X_test_norm, y_test, verbose=0)
        test_loss = test_metrics[0]
        test_accuracy = test_metrics[1]
        test_top2 = test_metrics[2]
        test_precision = test_metrics[3]
        test_recall = test_metrics[4]
        test_f1 = test_metrics[5]
        
        # Medical metrics
        medical_metrics = self.calculate_medical_metrics(y_true, y_pred, y_pred_proba)
        
        # Per-class metrics
        per_class_metrics = self.calculate_per_class_metrics(y_true, y_pred)
        
        # Confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        
        # Confidence analysis
        confidence_analysis = self.analyze_prediction_confidence(y_pred_proba, y_pred, y_true)
        
        results = {
            'test_accuracy': test_accuracy,
            'test_loss': test_loss,
            'test_top2_accuracy': test_top2,
            'test_precision': test_precision,
            'test_recall': test_recall,
            'test_f1_score': test_f1,
            'medical_metrics': medical_metrics,
            'per_class_metrics': per_class_metrics,
            'confusion_matrix': cm,
            'confidence_analysis': confidence_analysis,
            'predictions': y_pred,
            'true_labels': y_true,
            'prediction_probabilities': y_pred_proba
        }
        
        return results
    
    def calculate_medical_metrics(self, y_true, y_pred, y_pred_proba):
        """Calculate medical-specific metrics"""
        # Binary tumor detection (0=tumor, 1=no tumor becomes 0=tumor, 1=no tumor)
        # Convert to binary: 0,1,3 = tumor (1), 2 = no tumor (0)
        y_true_binary = (y_true != 2).astype(int)  # no_tumor=2, others are tumors
        y_pred_binary = (y_pred != 2).astype(int)
        
        # Tumor probabilities
        tumor_proba = 1 - y_pred_proba[:, 2]  # 1 - P(no_tumor)
        
        # Binary metrics
        from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
        
        binary_accuracy = accuracy_score(y_true_binary, y_pred_binary)
        tumor_sensitivity = recall_score(y_true_binary, y_pred_binary, pos_label=1)  # True positive rate για tumors
        tumor_specificity = recall_score(y_true_binary, y_pred_binary, pos_label=0)  # True negative rate για no_tumor
        binary_f1 = f1_score(y_true_binary, y_pred_binary, pos_label=1)
        
        try:
            binary_auc = roc_auc_score(y_true_binary, tumor_proba)
        except:
            binary_auc = 0.0
        
        # False negatives/positives (critical για medical)
        false_negatives = np.sum((y_true_binary == 1) & (y_pred_binary == 0))
        false_positives = np.sum((y_true_binary == 0) & (y_pred_binary == 1))
        
        return {
            'binary_tumor_detection_accuracy': binary_accuracy,
            'tumor_sensitivity': tumor_sensitivity,
            'tumor_specificity': tumor_specificity,
            'binary_f1_score': binary_f1,
            'binary_auc': binary_auc,
            'false_negatives': int(false_negatives),
            'false_positives': int(false_positives)
        }
    
    def calculate_per_class_metrics(self, y_true, y_pred):
        """Calculate per-class performance metrics"""
        precision, recall, f1, support = precision_recall_fscore_support(
            y_true, y_pred, labels=[0, 1, 2, 3], zero_division=0
        )
        
        per_class = {}
        for i, class_name in enumerate(self.class_names):
            per_class[class_name.lower()] = {
                'precision': precision[i],
                'recall': recall[i],
                'f1_score': f1[i],
                'support': int(support[i]),
                'accuracy': np.sum((y_true == i) & (y_pred == i)) / max(1, np.sum(y_true == i))
            }
        
        return per_class
    
    def analyze_prediction_confidence(self, y_pred_proba, y_pred, y_true):
        """Analyze prediction confidence levels"""
        max_proba = np.max(y_pred_proba, axis=1)
        
        # Confidence levels
        very_high = max_proba >= 0.9
        high = (max_proba >= 0.8) & (max_proba < 0.9)
        medium = (max_proba >= 0.6) & (max_proba < 0.8)
        low = max_proba < 0.6
        
        confidence_analysis = {}
        for level, mask in [('very_high', very_high), ('high', high), 
                           ('medium', medium), ('low', low)]:
            if np.sum(mask) > 0:
                accuracy = np.mean(y_pred[mask] == y_true[mask])
                confidence_analysis[level] = {
                    'count': int(np.sum(mask)),
                    'percentage': float(np.sum(mask) / len(y_pred) * 100),
                    'accuracy': float(accuracy)
                }
            else:
                confidence_analysis[level] = {
                    'count': 0, 'percentage': 0.0, 'accuracy': 0.0
                }
        
        confidence_analysis['mean_confidence'] = float(np.mean(max_proba))
        
        return confidence_analysis
    
    def create_visualizations(self, results, output_dir):
        """Create evaluation visualizations"""
        print("\n📊 Creating Evaluation Visualizations...")
        
        cm = results['confusion_matrix']
        
        # 1. Confusion Matrix
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=self.class_names, yticklabels=self.class_names,
                   cbar_kws={'label': 'Count'})
        plt.title('Improved Dual System - Confusion Matrix', 
                 fontsize=16, fontweight='bold', pad=20)
        plt.xlabel('Predicted Label', fontweight='bold')
        plt.ylabel('True Label', fontweight='bold')
        
        # Add accuracy text
        accuracy = results['test_accuracy']
        plt.figtext(0.5, 0.02, f'Test Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)', 
                   ha='center', fontsize=12, fontweight='bold')
        
        plt.tight_layout()
        cm_path = os.path.join(output_dir, 'improved_dual_confusion_matrix.png')
        plt.savefig(cm_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        # 2. Per-class Performance
        plt.figure(figsize=(12, 8))
        
        classes = list(results['per_class_metrics'].keys())
        metrics = ['precision', 'recall', 'f1_score']
        
        x = np.arange(len(classes))
        width = 0.25
        
        for i, metric in enumerate(metrics):
            values = [results['per_class_metrics'][cls][metric] for cls in classes]
            plt.bar(x + i*width, values, width, label=metric.capitalize())
        
        plt.xlabel('Classes', fontweight='bold')
        plt.ylabel('Score', fontweight='bold')
        plt.title('Improved Dual System - Per-Class Performance', 
                 fontsize=16, fontweight='bold')
        plt.xticks(x + width, [cls.capitalize() for cls in classes])
        plt.legend()
        plt.grid(axis='y', alpha=0.3)
        plt.ylim(0, 1.1)
        
        # Add value labels on bars
        for i, metric in enumerate(metrics):
            values = [results['per_class_metrics'][cls][metric] for cls in classes]
            for j, v in enumerate(values):
                plt.text(j + i*width, v + 0.01, f'{v:.3f}', 
                        ha='center', va='bottom', fontsize=9)
        
        plt.tight_layout()
        perf_path = os.path.join(output_dir, 'improved_dual_performance.png')
        plt.savefig(perf_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        # 3. Model Comparison
        plt.figure(figsize=(12, 8))
        
        # Historical results για comparison
        models = ['Simple Binary', '4-Class CNN', 'EfficientNet Dual', 'ResNet Dual', 'Improved Dual']
        accuracies = [99.57, 98.01, 91.27, 85.45, results['test_accuracy']*100]
        colors = ['gold', 'silver', 'lightcoral', 'lightblue', 'lightgreen']
        
        bars = plt.bar(models, accuracies, color=colors, alpha=0.8)
        
        plt.xlabel('Model Architecture', fontweight='bold')
        plt.ylabel('Test Accuracy (%)', fontweight='bold')
        plt.title('Model Performance Comparison', fontsize=16, fontweight='bold')
        plt.ylim(80, 100)
        
        # Add value labels
        for bar, acc in zip(bars, accuracies):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
                    f'{acc:.2f}%', ha='center', va='bottom', fontweight='bold')
        
        plt.xticks(rotation=45)
        plt.grid(axis='y', alpha=0.3)
        plt.tight_layout()
        
        comp_path = os.path.join(output_dir, 'improved_dual_comparison.png')
        plt.savefig(comp_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"💾 Visualizations saved:")
        print(f"  - Confusion Matrix: {cm_path}")
        print(f"  - Performance Chart: {perf_path}")
        print(f"  - Model Comparison: {comp_path}")
    
    def print_results_summary(self, results):
        """Print comprehensive results summary"""
        print("\n" + "="*70)
        print("🏆 IMPROVED DUAL SYSTEM - EVALUATION SUMMARY")
        print("="*70)
        
        # Overall performance
        print(f"\n📊 OVERALL PERFORMANCE:")
        print(f"  🎯 Test Accuracy: {results['test_accuracy']:.4f} ({results['test_accuracy']*100:.2f}%)")
        print(f"  🏆 Top-2 Accuracy: {results['test_top2_accuracy']:.4f} ({results['test_top2_accuracy']*100:.2f}%)")
        print(f"  📈 Precision: {results['test_precision']:.4f}")
        print(f"  🔍 Recall: {results['test_recall']:.4f}")
        print(f"  🎯 F1-Score: {results['test_f1_score']:.4f}")
        
        # Medical performance
        med = results['medical_metrics']
        print(f"\n🏥 MEDICAL PERFORMANCE:")
        print(f"  🔬 Binary Tumor Detection: {med['binary_tumor_detection_accuracy']:.4f} ({med['binary_tumor_detection_accuracy']*100:.2f}%)")
        print(f"  🎯 Tumor Sensitivity: {med['tumor_sensitivity']:.4f} ({med['tumor_sensitivity']*100:.2f}%)")
        print(f"  ⚖️ Tumor Specificity: {med['tumor_specificity']:.4f} ({med['tumor_specificity']*100:.2f}%)")
        print(f"  📊 Binary F1-Score: {med['binary_f1_score']:.4f}")
        print(f"  ❌ False Negatives: {med['false_negatives']} (Critical to minimize)")
        print(f"  ⚠️ False Positives: {med['false_positives']}")
        
        # Per-class performance
        print(f"\n🎯 PER-CLASS PERFORMANCE:")
        for class_name, metrics in results['per_class_metrics'].items():
            print(f"  {class_name.capitalize()}: {metrics['accuracy']:.4f} ({metrics['accuracy']*100:.2f}%) [{metrics['support']} samples]")
        
        # Confidence analysis
        conf = results['confidence_analysis']
        print(f"\n📊 CONFIDENCE ANALYSIS:")
        print(f"  📈 Mean Confidence: {conf['mean_confidence']:.4f}")
        for level in ['very_high', 'high', 'medium', 'low']:
            if conf[level]['count'] > 0:
                print(f"  {level.upper()}: {conf[level]['accuracy']:.3f} accuracy ({conf[level]['percentage']:.1f}% of predictions)")
        
        # Performance comparison
        acc = results['test_accuracy'] * 100
        print(f"\n📈 PERFORMANCE COMPARISON:")
        print(f"  Simple Binary CNN: 99.57%")
        print(f"  4-Class CNN: 98.01%")
        print(f"  Previous EfficientNet Dual: 91.27%")
        print(f"  Previous ResNet Dual: 85.45%")
        print(f"  ⭐ Improved Dual System: {acc:.2f}%")
        
        # Assessment
        if acc >= 95:
            grade = "EXCELLENT"
            status = "✅ CLINICAL GRADE"
        elif acc >= 92:
            grade = "GOOD" 
            status = "✅ RESEARCH GRADE"
        elif acc >= 88:
            grade = "MODERATE"
            status = "⚠️ NEEDS IMPROVEMENT"
        else:
            grade = "POOR"
            status = "❌ NOT READY"
        
        print(f"\n🏥 CLINICAL ASSESSMENT:")
        print(f"  📋 Grade: {grade}")
        print(f"  🚀 Status: {status}")
        
        if acc > 91.27:
            print(f"  🎉 SUCCESS: Improved over previous dual systems!")
        else:
            print(f"  📊 ANALYSIS: Performance vs previous dual systems")

def main():
    """Main evaluation function"""
    parser = argparse.ArgumentParser(description='Evaluate Improved Dual Branch System')
    parser.add_argument('--model', required=True, help='Path to trained model (.h5 file)')
    parser.add_argument('--test_data', required=True, help='Path to test data (.npz file)')
    parser.add_argument('--output', default='improved_dual_results', help='Output directory')
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output, exist_ok=True)
    
    print("🔬 IMPROVED DUAL SYSTEM EVALUATION")
    print("="*50)
    print(f"📅 Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"📁 Model: {args.model}")
    print(f"📁 Test Data: {args.test_data}")
    print(f"📁 Output: {args.output}")
    
    # Initialize evaluator
    evaluator = ImprovedDualEvaluator(args.model, args.test_data)
    
    # Load model and data
    X_test, y_test = evaluator.load_model_and_data()
    
    # Run evaluation
    results = evaluator.evaluate_model(X_test, y_test)
    
    # Print summary
    evaluator.print_results_summary(results)
    
    # Create visualizations
    evaluator.create_visualizations(results, args.output)
    
    # Save results
    results_path = os.path.join(args.output, 'improved_dual_evaluation_results.json')
    
    # Convert numpy types για JSON serialization
    json_results = {}
    for key, value in results.items():
        if isinstance(value, np.ndarray):
            json_results[key] = value.tolist()
        elif isinstance(value, np.integer):
            json_results[key] = int(value)
        elif isinstance(value, np.floating):
            json_results[key] = float(value)
        else:
            json_results[key] = value
    
    with open(results_path, 'w') as f:
        json.dump(json_results, f, indent=2)
    
    print(f"\n💾 Results saved: {results_path}")
    print(f"\n✅ Evaluation Complete!")

if __name__ == "__main__":
    main()