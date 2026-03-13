#!/usr/bin/env python3
"""
Evaluation script for Dual Branch System
Comprehensive performance analysis and comparison with other models
"""

import os
import sys
import argparse
import json
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix
import logging
from datetime import datetime

# Import our modules
from dual_system import DualBranchSystem
from utils_dual import (
    DataLoader, ModelEvaluator, Visualizer,
    save_results, load_results, setup_logging
)
from config_dual import get_config, COMPARISON_MODELS, DATA_CONFIG

class DualSystemEvaluator:
    """
    Comprehensive evaluator for Dual Branch System
    """
    
    def __init__(self, model_path, data_path, results_dir=None):
        """
        Initialize evaluator
        
        Args:
            model_path: Path to trained dual system model
            data_path: Path to dataset
            results_dir: Directory to save results
        """
        self.model_path = model_path
        self.data_path = data_path
        self.results_dir = results_dir or "dual_system_evaluation"
        
        # Create results directory
        os.makedirs(self.results_dir, exist_ok=True)
        
        # Setup logging
        log_file = os.path.join(self.results_dir, 'evaluation.log')
        setup_logging(log_file)
        self.logger = logging.getLogger(__name__)
        
        # Initialize components
        self.config = get_config()
        self.model = None
        self.data_loader = None
        self.evaluator = None
        self.visualizer = None
        
        # Evaluation results
        self.evaluation_results = {}
        self.comparison_results = {}
        
        self.logger.info("Dual System Evaluator initialized")
        self.logger.info(f"Model path: {model_path}")
        self.logger.info(f"Data path: {data_path}")
        self.logger.info(f"Results directory: {self.results_dir}")
    
    def setup_components(self):
        """
        Setup evaluation components
        """
        self.logger.info("Setting up evaluation components...")
        
        # Initialize data loader
        self.data_loader = DataLoader(self.data_path, self.config['data'])
        
        # Initialize evaluator and visualizer
        self.evaluator = ModelEvaluator(DATA_CONFIG['CLASS_NAMES_DISPLAY'])
        self.visualizer = Visualizer(DATA_CONFIG['CLASS_NAMES_DISPLAY'])
        
        self.logger.info("Components setup completed")
    
    def load_model(self):
        """
        Load trained dual system model
        """
        self.logger.info(f"Loading model from: {self.model_path}")
        
        try:
            self.model = tf.keras.models.load_model(self.model_path)
            self.logger.info("Model loaded successfully")
            
            # Log model information
            total_params = self.model.count_params()
            trainable_params = sum([layer.count_params() for layer in self.model.layers if layer.trainable])
            
            self.logger.info(f"Total parameters: {total_params:,}")
            self.logger.info(f"Trainable parameters: {trainable_params:,}")
            
        except Exception as e:
            self.logger.error(f"Failed to load model: {e}")
            raise
    
    def load_test_data(self):
        """
        Load test data for evaluation
        
        Returns:
            tuple: (test_generator, test_steps)
        """
        self.logger.info("Loading test data...")
        
        # Validate dataset structure
        if not self.data_loader.validate_dataset_structure():
            raise ValueError("Invalid dataset structure")
        
        # Create data generators
        train_gen, val_gen, test_gen, steps_info = self.data_loader.create_data_generators(
            validation_split=self.config['data']['VALIDATION_SPLIT'],
            test_split=self.config['data']['TEST_SPLIT']
        )
        
        self.logger.info(f"Test samples: {steps_info['test_samples']}")
        self.logger.info(f"Test steps: {steps_info['test_steps']}")
        
        return test_gen, steps_info['test_steps']
    
    def evaluate_model_performance(self):
        """
        Evaluate dual system model performance
        """
        self.logger.info("Starting model performance evaluation...")
        
        # Load test data
        test_generator, test_steps = self.load_test_data()
        
        # Perform evaluation
        results = self.evaluator.evaluate_model(self.model, test_generator, test_steps)
        
        # Add model-specific information
        results['model_info'] = {
            'model_type': 'Dual Branch System',
            'architecture': 'Custom CNN + ResNet-50',
            'fusion_strategy': self.config['model']['FUSION']['STRATEGY'],
            'total_parameters': self.model.count_params(),
            'evaluation_timestamp': datetime.now().isoformat()
        }
        
        # Store results
        self.evaluation_results = results
        
        # Log key metrics
        self.logger.info("Evaluation completed")
        self.logger.info(f"Overall Accuracy: {results['accuracy']:.4f} ({results['accuracy']*100:.2f}%)")
        self.logger.info(f"Macro Precision: {results['precision_macro']:.4f}")
        self.logger.info(f"Macro Recall: {results['recall_macro']:.4f}")
        self.logger.info(f"Macro F1-Score: {results['f1_macro']:.4f}")
        
        # Log per-class performance
        class_names = DATA_CONFIG['CLASS_NAMES_DISPLAY']
        self.logger.info("Per-class performance:")
        for i, class_name in enumerate(class_names):
            precision = results['precision_per_class'][i]
            recall = results['recall_per_class'][i]
            f1 = results['f1_per_class'][i]
            support = results['support_per_class'][i]
            
            self.logger.info(f"  {class_name}:")
            self.logger.info(f"    Precision: {precision:.4f}")
            self.logger.info(f"    Recall: {recall:.4f}")
            self.logger.info(f"    F1-Score: {f1:.4f}")
            self.logger.info(f"    Support: {support}")
    
    def calculate_clinical_metrics(self):
        """
        Calculate clinical-relevant metrics
        """
        self.logger.info("Calculating clinical metrics...")
        
        results = self.evaluation_results
        cm = np.array(results['confusion_matrix'])
        class_names = DATA_CONFIG['CLASS_NAMES_DISPLAY']
        
        # Clinical assessment
        clinical_metrics = {
            'overall_accuracy': results['accuracy'],
            'clinical_grade': results['accuracy'] >= 0.95,  # Clinical grade threshold
            'per_class_analysis': {}
        }
        
        # Per-class clinical metrics
        for i, class_name in enumerate(class_names):
            tp = cm[i, i]
            fp = cm[:, i].sum() - tp
            fn = cm[i, :].sum() - tp
            tn = cm.sum() - tp - fp - fn
            
            # Clinical metrics
            sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0  # Recall
            specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
            ppv = tp / (tp + fp) if (tp + fp) > 0 else 0  # Precision
            npv = tn / (tn + fn) if (tn + fn) > 0 else 0
            
            clinical_metrics['per_class_analysis'][class_name] = {
                'sensitivity_recall': sensitivity,
                'specificity': specificity,
                'positive_predictive_value': ppv,
                'negative_predictive_value': npv,
                'true_positives': int(tp),
                'false_positives': int(fp),
                'false_negatives': int(fn),
                'true_negatives': int(tn)
            }
            
            self.logger.info(f"{class_name} - Sensitivity: {sensitivity:.4f}, Specificity: {specificity:.4f}")
        
        # Add to evaluation results
        self.evaluation_results['clinical_metrics'] = clinical_metrics
        
        # Overall clinical assessment
        if clinical_metrics['clinical_grade']:
            self.logger.info("CLINICAL ASSESSMENT: EXCELLENT (≥95% accuracy)")
        elif results['accuracy'] >= 0.90:
            self.logger.info("CLINICAL ASSESSMENT: VERY GOOD (≥90% accuracy)")
        elif results['accuracy'] >= 0.85:
            self.logger.info("CLINICAL ASSESSMENT: GOOD (≥85% accuracy)")
        else:
            self.logger.info("CLINICAL ASSESSMENT: NEEDS IMPROVEMENT (<85% accuracy)")
    
    def compare_with_other_models(self):
        """
        Compare dual system performance with other implemented models
        """
        self.logger.info("Comparing with other models...")
        
        dual_accuracy = self.evaluation_results['accuracy']
        
        # Comparison data
        comparison_data = {
            'Dual Branch System': {
                'accuracy': dual_accuracy,
                'architecture': 'Custom CNN + ResNet-50',
                'parameters': self.model.count_params(),
                'status': 'CURRENT'
            }
        }
        
        # Add other models from configuration
        for model_key, model_info in COMPARISON_MODELS.items():
            comparison_data[model_info['NAME']] = {
                'accuracy': model_info['ACCURACY'],
                'architecture': model_key,
                'parameters': 'N/A',
                'status': 'REFERENCE'
            }
        
        # Sort by accuracy
        sorted_models = sorted(comparison_data.items(), key=lambda x: x[1]['accuracy'], reverse=True)
        
        # Create ranking
        ranking = []
        for i, (model_name, model_data) in enumerate(sorted_models, 1):
            ranking.append({
                'rank': i,
                'model': model_name,
                'accuracy': model_data['accuracy'],
                'accuracy_percent': model_data['accuracy'] * 100,
                'architecture': model_data['architecture'],
                'parameters': model_data['parameters'],
                'status': model_data['status']
            })
        
        self.comparison_results = {
            'dual_system_rank': next(r['rank'] for r in ranking if r['model'] == 'Dual Branch System'),
            'ranking': ranking,
            'performance_analysis': self._analyze_performance_ranking(ranking)
        }
        
        # Log comparison results
        self.logger.info("Model Performance Ranking:")
        for rank_info in ranking:
            status_symbol = "←" if rank_info['status'] == 'CURRENT' else " "
            self.logger.info(f"  {rank_info['rank']}. {rank_info['model']}: "
                           f"{rank_info['accuracy_percent']:.2f}% {status_symbol}")
        
        # Performance assessment
        dual_rank = self.comparison_results['dual_system_rank']
        if dual_rank == 1:
            self.logger.info("🏆 NEW CHAMPION: Dual System achieves SOTA performance!")
        elif dual_rank == 2:
            self.logger.info("🥈 EXCELLENT: Dual System achieves second-best performance!")
        elif dual_rank == 3:
            self.logger.info("🥉 VERY GOOD: Dual System achieves third-best performance!")
        else:
            self.logger.info(f"📊 GOOD: Dual System ranks #{dual_rank}")
    
    def _analyze_performance_ranking(self, ranking):
        """
        Analyze performance ranking and gaps
        
        Args:
            ranking: List of ranked models
            
        Returns:
            dict: Performance analysis
        """
        dual_entry = next(r for r in ranking if r['model'] == 'Dual Branch System')
        dual_rank = dual_entry['rank']
        dual_acc = dual_entry['accuracy_percent']
        
        analysis = {
            'dual_system_rank': dual_rank,
            'dual_system_accuracy': dual_acc,
            'total_models': len(ranking),
            'performance_gaps': {},
            'performance_category': 'UNKNOWN'
        }
        
        # Calculate performance gaps
        if dual_rank > 1:
            # Gap to best model
            best_model = ranking[0]
            gap_to_best = best_model['accuracy_percent'] - dual_acc
            analysis['performance_gaps']['to_best'] = {
                'model': best_model['model'],
                'accuracy': best_model['accuracy_percent'],
                'gap': gap_to_best
            }
        
        if dual_rank < len(ranking):
            # Gap from next lower model
            next_lower = ranking[dual_rank]  # dual_rank is 1-indexed, so this is correct
            gap_from_lower = dual_acc - next_lower['accuracy_percent']
            analysis['performance_gaps']['from_lower'] = {
                'model': next_lower['model'],
                'accuracy': next_lower['accuracy_percent'],
                'gap': gap_from_lower
            }
        
        # Performance category
        if dual_rank == 1:
            analysis['performance_category'] = 'CHAMPION'
        elif dual_rank <= 2:
            analysis['performance_category'] = 'EXCELLENT'
        elif dual_rank <= 3:
            analysis['performance_category'] = 'VERY_GOOD'
        else:
            analysis['performance_category'] = 'GOOD'
        
        return analysis
    
    def create_visualizations(self):
        """
        Create comprehensive evaluation visualizations
        """
        self.logger.info("Creating evaluation visualizations...")
        
        results = self.evaluation_results
        
        # 1. Confusion Matrix
        cm_path = os.path.join(self.results_dir, 'confusion_matrix.png')
        self.visualizer.plot_confusion_matrix(
            np.array(results['confusion_matrix']), 
            save_path=cm_path
        )
        
        # 2. Normalized Confusion Matrix
        cm_norm_path = os.path.join(self.results_dir, 'confusion_matrix_normalized.png')
        self.visualizer.plot_confusion_matrix(
            np.array(results['confusion_matrix']), 
            save_path=cm_norm_path,
            normalize=True
        )
        
        # 3. ROC Curves
        if len(results['predictions']) > 0:
            roc_data = self.evaluator.calculate_roc_curves(
                results['true_classes'], 
                np.array(results['predictions']),
                len(DATA_CONFIG['CLASS_NAMES'])
            )
            roc_path = os.path.join(self.results_dir, 'roc_curves.png')
            self.visualizer.plot_roc_curves(roc_data, save_path=roc_path)
        
        # 4. Performance Comparison
        if self.comparison_results:
            comparison_data = {r['model']: {'accuracy': r['accuracy']} 
                             for r in self.comparison_results['ranking']}
            comparison_path = os.path.join(self.results_dir, 'performance_comparison.png')
            self.visualizer.plot_performance_comparison(comparison_data, save_path=comparison_path)
        
        self.logger.info("Visualizations created successfully")
    
    def save_evaluation_results(self):
        """
        Save comprehensive evaluation results
        """
        self.logger.info("Saving evaluation results...")
        
        # Complete results
        complete_results = {
            'dual_system_evaluation': self.evaluation_results,
            'model_comparison': self.comparison_results,
            'evaluation_metadata': {
                'model_path': self.model_path,
                'data_path': self.data_path,
                'evaluation_timestamp': datetime.now().isoformat(),
                'evaluator_version': '1.0'
            }
        }
        
        # Save main results
        results_file = os.path.join(self.results_dir, 'evaluation_results.json')
        save_results(complete_results, results_file)
        
        # Save summary report
        self._create_summary_report()
        
        # Save detailed classification report
        classification_file = os.path.join(self.results_dir, 'classification_report.txt')
        with open(classification_file, 'w') as f:
            f.write("DUAL BRANCH SYSTEM - DETAILED CLASSIFICATION REPORT\n")
            f.write("=" * 60 + "\n\n")
            f.write(self.evaluation_results['classification_report'])
        
        self.logger.info(f"Results saved to: {self.results_dir}")
    
    def _create_summary_report(self):
        """
        Create human-readable summary report
        """
        results = self.evaluation_results
        comparison = self.comparison_results
        
        report_content = f"""# DUAL BRANCH SYSTEM EVALUATION REPORT

## Evaluation Summary
- **Date**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
- **Model**: Dual Branch System (Custom CNN + ResNet-50)
- **Fusion Strategy**: {results['model_info']['fusion_strategy']}
- **Total Parameters**: {results['model_info']['total_parameters']:,}

## Performance Results

### Overall Performance
- **Test Accuracy**: {results['accuracy']:.4f} ({results['accuracy']*100:.2f}%)
- **Macro Precision**: {results['precision_macro']:.4f}
- **Macro Recall**: {results['recall_macro']:.4f}
- **Macro F1-Score**: {results['f1_macro']:.4f}

### Clinical Assessment
- **Clinical Grade**: {'✅ YES' if results['clinical_metrics']['clinical_grade'] else '❌ NO'} (≥95% threshold)
- **Clinical Category**: {comparison['performance_analysis']['performance_category']}

### Per-Class Performance
"""
        
        class_names = DATA_CONFIG['CLASS_NAMES_DISPLAY']
        for i, class_name in enumerate(class_names):
            precision = results['precision_per_class'][i]
            recall = results['recall_per_class'][i]
            f1 = results['f1_per_class'][i]
            support = results['support_per_class'][i]
            
            report_content += f"""
**{class_name}:**
- Precision: {precision:.4f}
- Recall: {recall:.4f}
- F1-Score: {f1:.4f}
- Support: {support}
"""
        
        report_content += f"""
## Model Comparison

### Performance Ranking
"""
        
        for rank_info in comparison['ranking']:
            status = " ← DUAL SYSTEM" if rank_info['status'] == 'CURRENT' else ""
            report_content += f"{rank_info['rank']}. **{rank_info['model']}**: {rank_info['accuracy_percent']:.2f}%{status}\n"
        
        # Performance analysis
        perf_analysis = comparison['performance_analysis']
        report_content += f"""
### Performance Analysis
- **Dual System Rank**: #{perf_analysis['dual_system_rank']} out of {perf_analysis['total_models']} models
- **Performance Category**: {perf_analysis['performance_category']}
"""
        
        if 'to_best' in perf_analysis['performance_gaps']:
            gap_info = perf_analysis['performance_gaps']['to_best']
            report_content += f"- **Gap to Best Model**: {gap_info['gap']:.2f}% behind {gap_info['model']}\n"
        
        if perf_analysis['dual_system_rank'] == 1:
            report_content += "\n🏆 **NEW CHAMPION**: Dual System achieves state-of-the-art performance!\n"
        
        report_content += f"""
## Technical Details

### Architecture
- **Branch 1**: Custom Medical CNN (4-block progressive design)
- **Branch 2**: ResNet-50 (pre-trained on ImageNet)
- **Fusion**: {results['model_info']['fusion_strategy']} strategy
- **Classification**: Dense layers with medical-optimized regularization

### Training Strategy
- **Phase 1**: Frozen ResNet-50 training
- **Phase 2**: ResNet-50 fine-tuning
- **Phase 3**: Full model optimization (optional)

## Conclusions

**Key Findings:**
- Dual Branch System achieves {results['accuracy']*100:.2f}% test accuracy
- {'Exceeds' if results['clinical_metrics']['clinical_grade'] else 'Approaches'} clinical deployment threshold (95%)
- Ranks #{perf_analysis['dual_system_rank']} among implemented approaches

**Recommendation:**
- {'Deploy as new SOTA approach' if perf_analysis['dual_system_rank'] == 1 else 'Consider for specialized applications' if perf_analysis['dual_system_rank'] <= 3 else 'Continue with proven multiclass approaches'}
"""
        
        # Save report
        report_file = os.path.join(self.results_dir, 'EVALUATION_SUMMARY_REPORT.md')
        with open(report_file, 'w') as f:
            f.write(report_content)
        
        self.logger.info(f"Summary report saved to: {report_file}")
    
    def run_complete_evaluation(self):
        """
        Run complete evaluation pipeline
        """
        self.logger.info("Starting complete dual system evaluation")
        
        # Setup
        self.setup_components()
        self.load_model()
        
        # Core evaluation
        self.evaluate_model_performance()
        self.calculate_clinical_metrics()
        self.compare_with_other_models()
        
        # Visualization and reporting
        self.create_visualizations()
        self.save_evaluation_results()
        
        self.logger.info("Complete evaluation finished")
        
        # Final summary
        accuracy = self.evaluation_results['accuracy']
        rank = self.comparison_results['dual_system_rank']
        
        print("\n" + "=" * 80)
        print("DUAL BRANCH SYSTEM EVALUATION COMPLETED")
        print("=" * 80)
        print(f"Final Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
        print(f"Performance Rank: #{rank}")
        print(f"Clinical Grade: {'YES' if accuracy >= 0.95 else 'NO'}")
        print(f"Results Directory: {self.results_dir}")
        print("=" * 80)

def main():
    """
    Main evaluation function
    """
    parser = argparse.ArgumentParser(description='Evaluate Dual Branch System')
    parser.add_argument('--model', type=str, required=True,
                       help='Path to trained dual system model')
    parser.add_argument('--data', type=str, required=True,
                       help='Path to dataset directory')
    parser.add_argument('--output', type=str,
                       help='Output directory for results')
    
    args = parser.parse_args()
    
    # Print header
    print("=" * 80)
    print("DUAL BRANCH SYSTEM EVALUATION")
    print("=" * 80)
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Model: {args.model}")
    print(f"Dataset: {args.data}")
    print("=" * 80)
    
    try:
        # Initialize evaluator
        evaluator = DualSystemEvaluator(
            model_path=args.model,
            data_path=args.data,
            results_dir=args.output
        )
        
        # Run complete evaluation
        evaluator.run_complete_evaluation()
        
    except Exception as e:
        print(f"Evaluation failed: {e}")
        logging.error(f"Evaluation failed: {e}", exc_info=True)
        sys.exit(1)

if __name__ == "__main__":
    main()