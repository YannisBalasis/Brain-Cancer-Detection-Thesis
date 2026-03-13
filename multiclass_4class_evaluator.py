import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
from sklearn.metrics import precision_recall_curve, average_precision_score
from sklearn.preprocessing import label_binarize
import pandas as pd
import json
from datetime import datetime
import os

class MultiClass4ClassEvaluator:
    """
    Comprehensive evaluation for 4-Class Brain Tumor Classification
    Advanced metrics and visualizations για clinical assessment
    """
    
    def __init__(self, model_path=None):
        self.model = None
        self.class_names = ['Glioma', 'Meningioma', 'No Tumor', 'Pituitary']
        self.num_classes = 4
        
        # Medical evaluation criteria
        self.medical_thresholds = {
            'clinical_grade': 0.95,     # 95%+ for clinical deployment
            'research_grade': 0.90,     # 90%+ for research use
            'acceptable': 0.85,         # 85%+ minimally acceptable
        }
        
        if model_path and os.path.exists(model_path):
            self.load_model(model_path)
    
    def load_model(self, model_path):
        """Load trained model"""
        print(f"📁 Loading model from: {model_path}")
        try:
            self.model = keras.models.load_model(model_path)
            print("✅ Model loaded successfully!")
            print(f"📊 Model parameters: {self.model.count_params():,}")
        except Exception as e:
            print(f"❌ Error loading model: {e}")
            return False
        return True
    
    def comprehensive_evaluation(self, X_test, y_test, save_results=True):
        """
        Complete evaluation με όλα τα medical metrics
        """
        print("\n🏥 Starting Comprehensive Medical Evaluation...")
        print("=" * 60)
        
        if self.model is None:
            print("❌ No model loaded!")
            return None
        
        # Normalize test data
        X_test_norm = X_test / 255.0
        
        # Get predictions
        print("🔮 Generating predictions...")
        y_pred_proba = self.model.predict(X_test_norm, verbose=1)
        y_pred = np.argmax(y_pred_proba, axis=1)
        y_true = np.argmax(y_test, axis=1)
        
        # Basic metrics
        basic_results = self._calculate_basic_metrics(y_true, y_pred, y_pred_proba)
        
        # Advanced medical metrics
        medical_results = self._calculate_medical_metrics(y_true, y_pred, y_pred_proba)
        
        # Per-class analysis
        class_results = self._calculate_per_class_metrics(y_true, y_pred, y_pred_proba)
        
        # Clinical assessment
        clinical_results = self._clinical_assessment(basic_results, class_results)
        
        # Compile all results
        evaluation_results = {
            'basic_metrics': basic_results,
            'medical_metrics': medical_results,
            'per_class_metrics': class_results,
            'clinical_assessment': clinical_results,
            'raw_data': {
                'y_true': y_true.tolist(),
                'y_pred': y_pred.tolist(),
                'y_pred_proba': y_pred_proba.tolist()
            },
            'evaluation_timestamp': datetime.now().isoformat()
        }
        
        if save_results:
            self._save_evaluation_results(evaluation_results)
            self._create_comprehensive_visualizations(evaluation_results)
        
        self._print_comprehensive_summary(evaluation_results)
        
        return evaluation_results
    
    def _calculate_basic_metrics(self, y_true, y_pred, y_pred_proba):
        """Calculate standard classification metrics"""
        print("📊 Calculating basic metrics...")
        
        # Overall accuracy
        accuracy = np.mean(y_true == y_pred)
        
        # Top-k accuracies
        top_2_acc = np.mean([y_true[i] in np.argsort(y_pred_proba[i])[-2:] for i in range(len(y_true))])
        top_3_acc = np.mean([y_true[i] in np.argsort(y_pred_proba[i])[-3:] for i in range(len(y_true))])
        
        # Confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        
        # Classification report
        class_report = classification_report(
            y_true, y_pred, 
            target_names=self.class_names,
            output_dict=True
        )
        
        return {
            'accuracy': float(accuracy),
            'top_2_accuracy': float(top_2_acc),
            'top_3_accuracy': float(top_3_acc),
            'confusion_matrix': cm.tolist(),
            'classification_report': class_report
        }
    
    def _calculate_medical_metrics(self, y_true, y_pred, y_pred_proba):
        """Calculate medical-specific metrics"""
        print("🏥 Calculating medical metrics...")
        
        # Sensitivity (Recall) για κάθε class - CRITICAL για medical
        sensitivity = {}
        specificity = {}
        ppv = {}  # Positive Predictive Value
        npv = {}  # Negative Predictive Value
        
        for i, class_name in enumerate(self.class_names):
            # Binary classification για each class
            y_true_binary = (y_true == i).astype(int)
            y_pred_binary = (y_pred == i).astype(int)
            
            tp = np.sum((y_true_binary == 1) & (y_pred_binary == 1))
            tn = np.sum((y_true_binary == 0) & (y_pred_binary == 0))
            fp = np.sum((y_true_binary == 0) & (y_pred_binary == 1))
            fn = np.sum((y_true_binary == 1) & (y_pred_binary == 0))
            
            # Calculate metrics
            sensitivity[class_name] = tp / (tp + fn) if (tp + fn) > 0 else 0
            specificity[class_name] = tn / (tn + fp) if (tn + fp) > 0 else 0
            ppv[class_name] = tp / (tp + fp) if (tp + fp) > 0 else 0
            npv[class_name] = tn / (tn + fn) if (tn + fn) > 0 else 0
        
        # Binary tumor detection performance (No Tumor vs Any Tumor)
        tumor_detection = self._calculate_tumor_detection_metrics(y_true, y_pred)
        
        # Confidence analysis
        confidence_analysis = self._analyze_prediction_confidence(y_pred_proba, y_true, y_pred)
        
        return {
            'sensitivity': sensitivity,
            'specificity': specificity,
            'positive_predictive_value': ppv,
            'negative_predictive_value': npv,
            'tumor_detection': tumor_detection,
            'confidence_analysis': confidence_analysis
        }
    
    def _calculate_tumor_detection_metrics(self, y_true, y_pred):
        """Evaluate binary tumor detection (critical for screening)"""
        print("🔍 Analyzing tumor detection performance...")
        
        # Convert to binary: No Tumor (2) vs Any Tumor (0,1,3)
        y_true_tumor = (y_true != 2).astype(int)  # 2 is No Tumor
        y_pred_tumor = (y_pred != 2).astype(int)
        
        accuracy = np.mean(y_true_tumor == y_pred_tumor)
        
        tp = np.sum((y_true_tumor == 1) & (y_pred_tumor == 1))
        tn = np.sum((y_true_tumor == 0) & (y_pred_tumor == 0))
        fp = np.sum((y_true_tumor == 0) & (y_pred_tumor == 1))
        fn = np.sum((y_true_tumor == 1) & (y_pred_tumor == 0))
        
        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        
        return {
            'binary_tumor_accuracy': float(accuracy),
            'tumor_sensitivity': float(sensitivity),    # Critical: detect tumors
            'tumor_specificity': float(specificity),    # Important: avoid false alarms
            'false_positives': int(fp),
            'false_negatives': int(fn)  # Most critical to minimize
        }
    
    def _analyze_prediction_confidence(self, y_pred_proba, y_true, y_pred):
        """Analyze prediction confidence for clinical reliability"""
        print("📊 Analyzing prediction confidence...")
        
        max_probs = np.max(y_pred_proba, axis=1)
        correct_predictions = (y_true == y_pred)
        
        # Confidence categories
        very_high_conf = max_probs >= 0.95
        high_conf = (max_probs >= 0.85) & (max_probs < 0.95)
        medium_conf = (max_probs >= 0.70) & (max_probs < 0.85)
        low_conf = max_probs < 0.70
        
        # Accuracy per confidence level
        confidence_accuracies = {}
        for conf_name, conf_mask in [
            ('very_high', very_high_conf),
            ('high', high_conf),
            ('medium', medium_conf),
            ('low', low_conf)
        ]:
            if np.sum(conf_mask) > 0:
                conf_acc = np.mean(correct_predictions[conf_mask])
                conf_count = np.sum(conf_mask)
            else:
                conf_acc = 0.0
                conf_count = 0
            
            confidence_accuracies[conf_name] = {
                'accuracy': float(conf_acc),
                'count': int(conf_count),
                'percentage': float(conf_count / len(y_true) * 100)
            }
        
        return {
            'mean_confidence': float(np.mean(max_probs)),
            'std_confidence': float(np.std(max_probs)),
            'confidence_accuracy_breakdown': confidence_accuracies
        }
    
    def _calculate_per_class_metrics(self, y_true, y_pred, y_pred_proba):
        """Detailed per-class analysis"""
        print("🎯 Calculating per-class metrics...")
        
        class_metrics = {}
        
        for i, class_name in enumerate(self.class_names):
            class_mask = (y_true == i)
            class_predictions = y_pred[class_mask]
            class_true = y_true[class_mask]
            
            if len(class_true) > 0:
                class_accuracy = np.mean(class_predictions == class_true)
                
                # Confusion for this class
                correct = np.sum(class_predictions == i)
                total = len(class_true)
                
                # Most common misclassifications
                misclassifications = {}
                for j, other_class in enumerate(self.class_names):
                    if i != j:
                        misc_count = np.sum(class_predictions == j)
                        if misc_count > 0:
                            misclassifications[other_class] = {
                                'count': int(misc_count),
                                'percentage': float(misc_count / total * 100)
                            }
                
                class_metrics[class_name] = {
                    'accuracy': float(class_accuracy),
                    'total_samples': int(total),
                    'correct_predictions': int(correct),
                    'misclassifications': misclassifications,
                    'mean_confidence': float(np.mean(y_pred_proba[class_mask, i])),
                }
            else:
                class_metrics[class_name] = {
                    'accuracy': 0.0,
                    'total_samples': 0,
                    'error': 'No samples in test set'
                }
        
        return class_metrics
    
    def _clinical_assessment(self, basic_results, class_results):
        """Clinical deployment readiness assessment"""
        print("🏥 Performing clinical assessment...")
        
        accuracy = basic_results['accuracy']
        
        # Determine clinical grade
        if accuracy >= self.medical_thresholds['clinical_grade']:
            grade = 'CLINICAL_GRADE'
            recommendation = 'Suitable for clinical deployment με appropriate oversight'
        elif accuracy >= self.medical_thresholds['research_grade']:
            grade = 'RESEARCH_GRADE'
            recommendation = 'Suitable for research and specialist consultation'
        elif accuracy >= self.medical_thresholds['acceptable']:
            grade = 'ACCEPTABLE'
            recommendation = 'Minimally acceptable, needs improvement for clinical use'
        else:
            grade = 'BELOW_ACCEPTABLE'
            recommendation = 'Not suitable for medical use, significant improvement needed'
        
        # Risk assessment για critical classes
        risk_assessment = {}
        critical_classes = ['Glioma', 'Meningioma', 'Pituitary']  # Tumor types
        
        for class_name in critical_classes:
            if class_name in class_results:
                class_acc = class_results[class_name]['accuracy']
                if class_acc >= 0.90:
                    risk_level = 'LOW'
                elif class_acc >= 0.80:
                    risk_level = 'MODERATE'
                else:
                    risk_level = 'HIGH'
                
                risk_assessment[class_name] = {
                    'risk_level': risk_level,
                    'accuracy': class_acc
                }
        
        return {
            'clinical_grade': grade,
            'overall_accuracy': float(accuracy),
            'recommendation': recommendation,
            'risk_assessment': risk_assessment,
            'deployment_ready': grade in ['CLINICAL_GRADE', 'RESEARCH_GRADE']
        }
    
    def _create_comprehensive_visualizations(self, results):
        """Create comprehensive visualization suite"""
        print("📊 Creating comprehensive visualizations...")
        
        # 1. Enhanced Confusion Matrix
        self._plot_enhanced_confusion_matrix(results['basic_metrics']['confusion_matrix'])
        
        # 2. Per-class Performance
        self._plot_per_class_performance(results['per_class_metrics'])
        
        # 3. Confidence Analysis
        self._plot_confidence_analysis(results['medical_metrics']['confidence_analysis'])
        
        # 4. Medical Metrics Dashboard
        self._plot_medical_dashboard(results['medical_metrics'])
        
        # 5. Clinical Summary
        self._create_clinical_summary_plot(results['clinical_assessment'])
    
    def _plot_enhanced_confusion_matrix(self, cm_list):
        """Enhanced confusion matrix με medical insights"""
        cm = np.array(cm_list)
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        
        # Absolute numbers
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax1,
                   xticklabels=self.class_names, yticklabels=self.class_names)
        ax1.set_title('Confusion Matrix - Absolute Counts', fontsize=14, fontweight='bold')
        ax1.set_xlabel('Predicted Label')
        ax1.set_ylabel('True Label')
        
        # Percentages
        cm_percent = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100
        sns.heatmap(cm_percent, annot=True, fmt='.1f', cmap='Blues', ax=ax2,
                   xticklabels=self.class_names, yticklabels=self.class_names,
                   cbar_kws={'label': 'Percentage (%)'})
        ax2.set_title('Confusion Matrix - Percentages', fontsize=14, fontweight='bold')
        ax2.set_xlabel('Predicted Label')
        ax2.set_ylabel('True Label')
        
        plt.tight_layout()
        plt.savefig('/home/claude/multiclass_4class_enhanced_confusion_matrix.png', 
                   dpi=300, bbox_inches='tight')
        print("💾 Enhanced confusion matrix saved")
    
    def _plot_per_class_performance(self, class_metrics):
        """Per-class performance analysis"""
        classes = list(class_metrics.keys())
        accuracies = [class_metrics[cls]['accuracy'] for cls in classes]
        sample_counts = [class_metrics[cls]['total_samples'] for cls in classes]
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Per-class accuracy
        bars1 = ax1.bar(classes, accuracies, color=['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4'])
        ax1.set_title('Per-Class Accuracy', fontsize=14, fontweight='bold')
        ax1.set_ylabel('Accuracy')
        ax1.set_ylim(0, 1.0)
        ax1.axhline(y=0.9, color='green', linestyle='--', alpha=0.7, label='Clinical Grade (90%)')
        ax1.axhline(y=0.85, color='orange', linestyle='--', alpha=0.7, label='Acceptable (85%)')
        
        # Add value labels on bars
        for bar, acc in zip(bars1, accuracies):
            ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f'{acc:.3f}', ha='center', va='bottom', fontweight='bold')
        
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Sample distribution
        bars2 = ax2.bar(classes, sample_counts, color=['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4'])
        ax2.set_title('Test Set Distribution', fontsize=14, fontweight='bold')
        ax2.set_ylabel('Number of Samples')
        
        # Add value labels
        for bar, count in zip(bars2, sample_counts):
            ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                    f'{count}', ha='center', va='bottom', fontweight='bold')
        
        plt.tight_layout()
        plt.savefig('/home/claude/multiclass_4class_per_class_performance.png', 
                   dpi=300, bbox_inches='tight')
        print("💾 Per-class performance plot saved")
    
    def _plot_confidence_analysis(self, confidence_data):
        """Confidence analysis visualization"""
        conf_levels = list(confidence_data['confidence_accuracy_breakdown'].keys())
        conf_accuracies = [confidence_data['confidence_accuracy_breakdown'][level]['accuracy'] 
                          for level in conf_levels]
        conf_percentages = [confidence_data['confidence_accuracy_breakdown'][level]['percentage'] 
                           for level in conf_levels]
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Confidence vs Accuracy
        bars1 = ax1.bar(conf_levels, conf_accuracies, 
                       color=['#2ECC71', '#F39C12', '#E74C3C', '#9B59B6'])
        ax1.set_title('Accuracy by Confidence Level', fontsize=14, fontweight='bold')
        ax1.set_ylabel('Accuracy')
        ax1.set_ylim(0, 1.0)
        
        for bar, acc in zip(bars1, conf_accuracies):
            ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f'{acc:.3f}', ha='center', va='bottom', fontweight='bold')
        
        # Distribution of confidence levels
        bars2 = ax2.bar(conf_levels, conf_percentages, 
                       color=['#2ECC71', '#F39C12', '#E74C3C', '#9B59B6'])
        ax2.set_title('Distribution of Confidence Levels', fontsize=14, fontweight='bold')
        ax2.set_ylabel('Percentage of Predictions (%)')
        
        for bar, perc in zip(bars2, conf_percentages):
            ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                    f'{perc:.1f}%', ha='center', va='bottom', fontweight='bold')
        
        plt.tight_layout()
        plt.savefig('/home/claude/multiclass_4class_confidence_analysis.png', 
                   dpi=300, bbox_inches='tight')
        print("💾 Confidence analysis plot saved")
    
    def _plot_medical_dashboard(self, medical_metrics):
        """Medical metrics dashboard"""
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        
        # Sensitivity (Recall)
        classes = list(medical_metrics['sensitivity'].keys())
        sensitivity_values = list(medical_metrics['sensitivity'].values())
        
        bars1 = ax1.bar(classes, sensitivity_values, color='#E74C3C', alpha=0.7)
        ax1.set_title('Sensitivity (Recall) - Critical for Medical', fontsize=12, fontweight='bold')
        ax1.set_ylabel('Sensitivity')
        ax1.set_ylim(0, 1.0)
        ax1.axhline(y=0.9, color='green', linestyle='--', label='Target: 90%+')
        
        for bar, val in zip(bars1, sensitivity_values):
            ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f'{val:.3f}', ha='center', va='bottom', fontsize=10, fontweight='bold')
        
        # Specificity
        specificity_values = list(medical_metrics['specificity'].values())
        bars2 = ax2.bar(classes, specificity_values, color='#3498DB', alpha=0.7)
        ax2.set_title('Specificity - Avoid False Positives', fontsize=12, fontweight='bold')
        ax2.set_ylabel('Specificity')
        ax2.set_ylim(0, 1.0)
        ax2.axhline(y=0.9, color='green', linestyle='--', label='Target: 90%+')
        
        for bar, val in zip(bars2, specificity_values):
            ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f'{val:.3f}', ha='center', va='bottom', fontsize=10, fontweight='bold')
        
        # Tumor Detection Performance
        tumor_metrics = medical_metrics['tumor_detection']
        tumor_labels = ['Tumor\nSensitivity', 'Tumor\nSpecificity', 'Binary\nAccuracy']
        tumor_values = [tumor_metrics['tumor_sensitivity'], 
                       tumor_metrics['tumor_specificity'],
                       tumor_metrics['binary_tumor_accuracy']]
        
        bars3 = ax3.bar(tumor_labels, tumor_values, color=['#E74C3C', '#3498DB', '#2ECC71'])
        ax3.set_title('Binary Tumor Detection Performance', fontsize=12, fontweight='bold')
        ax3.set_ylabel('Performance')
        ax3.set_ylim(0, 1.0)
        ax3.axhline(y=0.95, color='green', linestyle='--', label='Target: 95%+')
        
        for bar, val in zip(bars3, tumor_values):
            ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f'{val:.3f}', ha='center', va='bottom', fontsize=10, fontweight='bold')
        
        # Error Analysis
        fp_count = tumor_metrics['false_positives']
        fn_count = tumor_metrics['false_negatives']
        
        ax4.bar(['False\nPositives', 'False\nNegatives'], [fp_count, fn_count], 
               color=['#F39C12', '#E74C3C'])
        ax4.set_title('Critical Errors Analysis', fontsize=12, fontweight='bold')
        ax4.set_ylabel('Count')
        
        plt.tight_layout()
        plt.savefig('/home/claude/multiclass_4class_medical_dashboard.png', 
                   dpi=300, bbox_inches='tight')
        print("💾 Medical dashboard saved")
    
    def _create_clinical_summary_plot(self, clinical_assessment):
        """Clinical summary visualization"""
        fig, ax = plt.subplots(1, 1, figsize=(10, 8))
        
        # Remove axes για text-based summary
        ax.axis('off')
        
        grade = clinical_assessment['clinical_grade']
        accuracy = clinical_assessment['overall_accuracy']
        recommendation = clinical_assessment['recommendation']
        deployment_ready = clinical_assessment['deployment_ready']
        
        # Color coding για grades
        grade_colors = {
            'CLINICAL_GRADE': '#2ECC71',
            'RESEARCH_GRADE': '#F39C12', 
            'ACCEPTABLE': '#E67E22',
            'BELOW_ACCEPTABLE': '#E74C3C'
        }
        
        # Main title
        ax.text(0.5, 0.9, '🏥 CLINICAL ASSESSMENT SUMMARY', 
               ha='center', va='center', fontsize=24, fontweight='bold',
               transform=ax.transAxes)
        
        # Grade
        ax.text(0.5, 0.75, f'Clinical Grade: {grade}', 
               ha='center', va='center', fontsize=18, fontweight='bold',
               color=grade_colors.get(grade, 'black'), transform=ax.transAxes)
        
        # Accuracy
        ax.text(0.5, 0.65, f'Overall Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)', 
               ha='center', va='center', fontsize=16, transform=ax.transAxes)
        
        # Deployment status
        deployment_text = "✅ DEPLOYMENT READY" if deployment_ready else "❌ NOT DEPLOYMENT READY"
        deployment_color = '#2ECC71' if deployment_ready else '#E74C3C'
        ax.text(0.5, 0.55, deployment_text, 
               ha='center', va='center', fontsize=14, fontweight='bold',
               color=deployment_color, transform=ax.transAxes)
        
        # Recommendation (wrapped text)
        ax.text(0.5, 0.4, 'Recommendation:', 
               ha='center', va='center', fontsize=14, fontweight='bold',
               transform=ax.transAxes)
        
        # Wrap recommendation text
        import textwrap
        wrapped_rec = textwrap.fill(recommendation, width=50)
        ax.text(0.5, 0.25, wrapped_rec, 
               ha='center', va='center', fontsize=12,
               transform=ax.transAxes)
        
        plt.savefig('/home/claude/multiclass_4class_clinical_summary.png', 
                   dpi=300, bbox_inches='tight')
        print("💾 Clinical summary saved")
    
    def _save_evaluation_results(self, results):
        """Save comprehensive evaluation results"""
        print("💾 Saving comprehensive evaluation results...")
        
        # Save detailed results
        with open('/home/claude/multiclass_4class_detailed_evaluation.json', 'w') as f:
            json.dump(results, f, indent=2)
        
        # Create summary report
        summary = {
            'evaluation_date': results['evaluation_timestamp'],
            'model_type': '4-Class Brain Tumor Multiclass CNN',
            'overall_performance': {
                'accuracy': results['basic_metrics']['accuracy'],
                'top_2_accuracy': results['basic_metrics']['top_2_accuracy'],
                'clinical_grade': results['clinical_assessment']['clinical_grade'],
                'deployment_ready': results['clinical_assessment']['deployment_ready']
            },
            'medical_performance': {
                'tumor_detection_accuracy': results['medical_metrics']['tumor_detection']['binary_tumor_accuracy'],
                'tumor_sensitivity': results['medical_metrics']['tumor_detection']['tumor_sensitivity'],
                'false_negatives': results['medical_metrics']['tumor_detection']['false_negatives']
            },
            'per_class_summary': {}
        }
        
        # Add per-class summary
        for class_name, metrics in results['per_class_metrics'].items():
            if 'accuracy' in metrics:
                summary['per_class_summary'][class_name] = {
                    'accuracy': metrics['accuracy'],
                    'sample_count': metrics['total_samples']
                }
        
        with open('/home/claude/multiclass_4class_evaluation_summary.json', 'w') as f:
            json.dump(summary, f, indent=2)
        
        print("✅ Evaluation results saved:")
        print("  - Detailed: multiclass_4class_detailed_evaluation.json")
        print("  - Summary: multiclass_4class_evaluation_summary.json")
    
    def _print_comprehensive_summary(self, results):
        """Print comprehensive evaluation summary"""
        print("\n" + "="*80)
        print("🏥 COMPREHENSIVE 4-CLASS MULTICLASS EVALUATION SUMMARY")
        print("="*80)
        
        # Basic Performance
        basic = results['basic_metrics']
        print(f"\n📊 OVERALL PERFORMANCE:")
        print(f"  🎯 Test Accuracy: {basic['accuracy']:.4f} ({basic['accuracy']*100:.2f}%)")
        print(f"  🏆 Top-2 Accuracy: {basic['top_2_accuracy']:.4f} ({basic['top_2_accuracy']*100:.2f}%)")
        print(f"  🥉 Top-3 Accuracy: {basic['top_3_accuracy']:.4f} ({basic['top_3_accuracy']*100:.2f}%)")
        
        # Medical Performance
        medical = results['medical_metrics']
        tumor_det = medical['tumor_detection']
        print(f"\n🏥 MEDICAL PERFORMANCE:")
        print(f"  🔍 Binary Tumor Detection: {tumor_det['binary_tumor_accuracy']:.4f} ({tumor_det['binary_tumor_accuracy']*100:.2f}%)")
        print(f"  🎯 Tumor Sensitivity: {tumor_det['tumor_sensitivity']:.4f} ({tumor_det['tumor_sensitivity']*100:.2f}%)")
        print(f"  ⚖️ Tumor Specificity: {tumor_det['tumor_specificity']:.4f} ({tumor_det['tumor_specificity']*100:.2f}%)")
        print(f"  ❌ False Negatives: {tumor_det['false_negatives']} (Critical to minimize)")
        print(f"  ⚠️ False Positives: {tumor_det['false_positives']}")
        
        # Per-Class Performance
        print(f"\n🎯 PER-CLASS PERFORMANCE:")
        for class_name, metrics in results['per_class_metrics'].items():
            if 'accuracy' in metrics:
                print(f"  {class_name}: {metrics['accuracy']:.4f} ({metrics['accuracy']*100:.2f}%) "
                      f"[{metrics['total_samples']} samples]")
        
        # Clinical Assessment
        clinical = results['clinical_assessment']
        print(f"\n🏥 CLINICAL ASSESSMENT:")
        print(f"  📋 Grade: {clinical['clinical_grade']}")
        print(f"  🚀 Deployment Ready: {'✅ YES' if clinical['deployment_ready'] else '❌ NO'}")
        print(f"  💡 Recommendation: {clinical['recommendation']}")
        
        # Confidence Analysis
        conf = medical['confidence_analysis']
        print(f"\n📊 CONFIDENCE ANALYSIS:")
        print(f"  📈 Mean Confidence: {conf['mean_confidence']:.4f}")
        for level, data in conf['confidence_accuracy_breakdown'].items():
            print(f"  {level.upper()}: {data['accuracy']:.3f} accuracy "
                  f"({data['percentage']:.1f}% of predictions)")
        
        print("\n" + "="*80)
        print("📁 All evaluation files saved in /home/claude/")
        print("="*80)


def main():
    """Main evaluation function"""
    print("🔬 4-Class Multiclass Model Evaluation")
    print("=" * 60)
    
    # Initialize evaluator
    evaluator = MultiClass4ClassEvaluator()
    
    # Note: In practice, you would load your trained model και test data
    # model_path = '/home/claude/best_multiclass_4class_model.h5'
    # evaluator.load_model(model_path)
    # results = evaluator.comprehensive_evaluation(X_test, y_test)
    
    print("\n✅ Evaluation framework ready!")
    print("📋 To use: evaluator.comprehensive_evaluation(X_test, y_test)")

if __name__ == "__main__":
    main()