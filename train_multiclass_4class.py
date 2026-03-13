#!/usr/bin/env python3
"""
🧠 4-Class Brain Tumor Multiclass Classification Training Pipeline
Complete training και evaluation system για Glioma/Meningioma/Pituitary/No Tumor

Author: Διπλωματική Εργασία - Brain Tumor Classification
Date: December 2025
"""

import os
import sys
import argparse
import json
from datetime import datetime
import tensorflow as tf
import numpy as np

# Import our custom modules
from multiclass_4class_model import MultiClass4ClassModel
from multiclass_4class_evaluator import MultiClass4ClassEvaluator

def setup_gpu_config():
    """Configure GPU settings για optimal training"""
    print("⚙️ Configuring GPU settings...")
    
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            # Enable memory growth για avoid OOM errors
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            print(f"✅ Found {len(gpus)} GPU(s), memory growth enabled")
        except RuntimeError as e:
            print(f"⚠️ GPU configuration error: {e}")
    else:
        print("⚠️ No GPUs found, using CPU (will be slow)")
    
    return len(gpus) > 0

def validate_dataset_structure(data_path):
    """Validate dataset structure και report statistics"""
    print(f"🔍 Validating dataset structure at: {data_path}")
    
    if not os.path.exists(data_path):
        raise ValueError(f"Dataset path does not exist: {data_path}")
    
    expected_classes = ['glioma', 'meningioma', 'no_tumor', 'pituitary']
    found_classes = {}
    total_images = 0
    
    for class_name in expected_classes:
        class_path = os.path.join(data_path, class_name)
        if os.path.exists(class_path):
            image_files = [f for f in os.listdir(class_path) 
                          if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff'))]
            found_classes[class_name] = len(image_files)
            total_images += len(image_files)
            print(f"  📁 {class_name}: {len(image_files):,} images")
        else:
            print(f"  ❌ {class_name}: folder not found!")
            found_classes[class_name] = 0
    
    print(f"📊 Total dataset: {total_images:,} images")
    
    # Check για balance
    if total_images > 0:
        print("📊 Class distribution:")
        for class_name, count in found_classes.items():
            percentage = (count / total_images) * 100
            print(f"  {class_name}: {percentage:.1f}%")
        
        # Check if reasonably balanced
        percentages = [count/total_images for count in found_classes.values() if count > 0]
        imbalance_ratio = max(percentages) / min(percentages) if len(percentages) > 1 else 1.0
        
        if imbalance_ratio > 3.0:
            print(f"⚠️ Dataset appears imbalanced (ratio: {imbalance_ratio:.1f}:1)")
        else:
            print(f"✅ Dataset reasonably balanced (ratio: {imbalance_ratio:.1f}:1)")
    
    return found_classes, total_images

def create_experiment_directory():
    """Create timestamped experiment directory"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    # Use current working directory instead of hardcoded path
    current_dir = os.getcwd()
    exp_dir = os.path.join(current_dir, f"multiclass_4class_experiment_{timestamp}")
    
    os.makedirs(exp_dir, exist_ok=True)
    print(f"📁 Created experiment directory: {exp_dir}")
    
    return exp_dir

def save_experiment_config(exp_dir, config):
    """Save experiment configuration"""
    config_path = os.path.join(exp_dir, "experiment_config.json")
    
    config_data = {
        'experiment_timestamp': datetime.now().isoformat(),
        'model_type': '4-Class Multiclass CNN',
        'dataset_path': config['data_path'],
        'architecture': 'Custom CNN (Binary-inspired)',
        'training_params': {
            'batch_size': 32,
            'max_epochs': 50,
            'patience': 15,
            'learning_rate': 0.001
        },
        'augmentation': {
            'rotation_range': 15,
            'width_shift_range': 0.1,
            'height_shift_range': 0.1,
            'zoom_range': 0.1,
            'brightness_range': [0.9, 1.1],
            'horizontal_flip': True
        },
        'medical_requirements': {
            'target_accuracy': 0.90,
            'clinical_grade_threshold': 0.95,
            'acceptable_threshold': 0.85
        }
    }
    
    with open(config_path, 'w') as f:
        json.dump(config_data, f, indent=2)
    
    print(f"💾 Experiment config saved: {config_path}")
    return config_path

def train_model(data_path, exp_dir):
    """Complete model training pipeline"""
    print("\n" + "="*80)
    print("🚀 STARTING 4-CLASS MULTICLASS TRAINING")
    print("="*80)
    
    # Initialize model
    print("\n1️⃣ Initializing Model...")
    model_trainer = MultiClass4ClassModel()
    
    # Create model architecture
    print("\n2️⃣ Building Architecture...")
    model_trainer.create_model()
    model_trainer.compile_model()
    
    # Load and preprocess data
    print("\n3️⃣ Loading Dataset...")
    try:
        X, y, class_counts = model_trainer.load_and_preprocess_data(data_path)
        print(f"✅ Successfully loaded {len(X):,} images")
    except Exception as e:
        print(f"❌ Error loading data: {e}")
        return None, None
    
    # Create data splits
    print("\n4️⃣ Creating Data Splits...")
    X_train, X_val, X_test, y_train, y_val, y_test = model_trainer.create_data_splits(X, y)
    
    # Save test data για later evaluation
    np.save(os.path.join(exp_dir, 'X_test.npy'), X_test)
    np.save(os.path.join(exp_dir, 'y_test.npy'), y_test)
    print(f"💾 Test data saved για evaluation")
    
    # Train model
    print("\n5️⃣ Training Model...")
    start_time = datetime.now()
    
    try:
        history = model_trainer.train_model(X_train, y_train, X_val, y_val, exp_dir)
        
        training_time = datetime.now() - start_time
        print(f"✅ Training completed in {training_time}")
        
    except Exception as e:
        print(f"❌ Training failed: {e}")
        return None, None
    
    # Create training visualizations
    print("\n6️⃣ Creating Training Visualizations...")
    try:
        fig = model_trainer.plot_training_history(exp_dir)
        if fig is not None:
            print(f"📊 Training visualization saved in: {exp_dir}")
    except Exception as e:
        print(f"⚠️ Visualization creation failed: {e}")
    
    print("\n✅ Training pipeline completed!")
    return model_trainer, (X_test, y_test)

def evaluate_model(model_trainer, test_data, exp_dir):
    """Comprehensive model evaluation"""
    print("\n" + "="*80)  
    print("📊 STARTING COMPREHENSIVE EVALUATION")
    print("="*80)
    
    if model_trainer is None or test_data is None:
        print("❌ Cannot evaluate: training failed ή no test data")
        return None
    
    X_test, y_test = test_data
    
    # Initialize evaluator
    evaluator = MultiClass4ClassEvaluator()
    
    # Load best model
    best_model_path = os.path.join(exp_dir, 'best_multiclass_4class_model.h5')
    if os.path.exists(best_model_path):
        evaluator.load_model(best_model_path)
    else:
        print("⚠️ Best model not found, using current model")
        evaluator.model = model_trainer.model
    
    # Run comprehensive evaluation
    try:
        results = evaluator.comprehensive_evaluation(X_test, y_test, save_results=False)
        
        # Move evaluation results to experiment directory
        eval_files = [
            '/home/claude/multiclass_4class_enhanced_confusion_matrix.png',
            '/home/claude/multiclass_4class_per_class_performance.png', 
            '/home/claude/multiclass_4class_confidence_analysis.png',
            '/home/claude/multiclass_4class_medical_dashboard.png',
            '/home/claude/multiclass_4class_clinical_summary.png'
        ]
        
        for file_path in eval_files:
            if os.path.exists(file_path):
                filename = os.path.basename(file_path)
                new_path = os.path.join(exp_dir, filename)
                os.rename(file_path, new_path)
        
        # Save evaluation results in experiment directory
        eval_results_path = os.path.join(exp_dir, 'evaluation_results.json')
        eval_summary_path = os.path.join(exp_dir, 'evaluation_summary.json')
        
        with open(eval_results_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        # Create summary
        summary = {
            'experiment_date': datetime.now().isoformat(),
            'model_performance': {
                'test_accuracy': results['basic_metrics']['accuracy'],
                'top_2_accuracy': results['basic_metrics']['top_2_accuracy'], 
                'clinical_grade': results['clinical_assessment']['clinical_grade'],
                'deployment_ready': results['clinical_assessment']['deployment_ready']
            },
            'medical_metrics': {
                'tumor_detection_accuracy': results['medical_metrics']['tumor_detection']['binary_tumor_accuracy'],
                'tumor_sensitivity': results['medical_metrics']['tumor_detection']['tumor_sensitivity'],
                'false_negatives': results['medical_metrics']['tumor_detection']['false_negatives']
            }
        }
        
        with open(eval_summary_path, 'w') as f:
            json.dump(summary, f, indent=2)
        
        print(f"💾 Evaluation results saved in: {exp_dir}")
        return results
        
    except Exception as e:
        print(f"❌ Evaluation failed: {e}")
        return None

def create_final_report(exp_dir, results):
    """Create final experiment report"""
    print("\n📋 Creating Final Experiment Report...")
    
    report_path = os.path.join(exp_dir, 'EXPERIMENT_REPORT.md')
    
    if results is None:
        print("⚠️ No results available για report")
        return
    
    report_content = f"""# 4-Class Brain Tumor Classification - Experiment Report

## Experiment Information
- **Date**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
- **Model Type**: 4-Class Multiclass CNN (Glioma/Meningioma/Pituitary/No Tumor)
- **Architecture**: Custom CNN based on proven binary model

## Performance Results

### Overall Performance
- **Test Accuracy**: {results['basic_metrics']['accuracy']:.4f} ({results['basic_metrics']['accuracy']*100:.2f}%)
- **Top-2 Accuracy**: {results['basic_metrics']['top_2_accuracy']:.4f} ({results['basic_metrics']['top_2_accuracy']*100:.2f}%)
- **Top-3 Accuracy**: {results['basic_metrics']['top_3_accuracy']:.4f} ({results['basic_metrics']['top_3_accuracy']*100:.2f}%)

### Clinical Assessment
- **Clinical Grade**: {results['clinical_assessment']['clinical_grade']}
- **Deployment Ready**: {"✅ YES" if results['clinical_assessment']['deployment_ready'] else "❌ NO"}
- **Recommendation**: {results['clinical_assessment']['recommendation']}

### Medical Performance
- **Binary Tumor Detection**: {results['medical_metrics']['tumor_detection']['binary_tumor_accuracy']:.4f} ({results['medical_metrics']['tumor_detection']['binary_tumor_accuracy']*100:.2f}%)
- **Tumor Sensitivity**: {results['medical_metrics']['tumor_detection']['tumor_sensitivity']:.4f} ({results['medical_metrics']['tumor_detection']['tumor_sensitivity']*100:.2f}%)
- **Tumor Specificity**: {results['medical_metrics']['tumor_detection']['tumor_specificity']:.4f} ({results['medical_metrics']['tumor_detection']['tumor_specificity']*100:.2f}%)
- **False Negatives**: {results['medical_metrics']['tumor_detection']['false_negatives']} (Critical)
- **False Positives**: {results['medical_metrics']['tumor_detection']['false_positives']}

### Per-Class Performance
"""
    
    # Add per-class results
    for class_name, metrics in results['per_class_metrics'].items():
        if 'accuracy' in metrics:
            report_content += f"- **{class_name}**: {metrics['accuracy']:.4f} ({metrics['accuracy']*100:.2f}%) [{metrics['total_samples']} samples]\n"
    
    report_content += f"""
### Confidence Analysis
- **Mean Confidence**: {results['medical_metrics']['confidence_analysis']['mean_confidence']:.4f}
- **Confidence Distribution**:
"""
    
    # Add confidence breakdown
    for level, data in results['medical_metrics']['confidence_analysis']['confidence_accuracy_breakdown'].items():
        report_content += f"  - **{level.upper()}**: {data['accuracy']:.3f} accuracy ({data['percentage']:.1f}% of predictions)\n"
    
    report_content += """
## Generated Files
- `best_multiclass_4class_model.h5` - Best trained model
- `training_history.png` - Training curves
- `multiclass_4class_enhanced_confusion_matrix.png` - Confusion matrices
- `multiclass_4class_per_class_performance.png` - Per-class analysis
- `multiclass_4class_confidence_analysis.png` - Confidence analysis
- `multiclass_4class_medical_dashboard.png` - Medical metrics
- `multiclass_4class_clinical_summary.png` - Clinical assessment
- `evaluation_results.json` - Detailed results
- `evaluation_summary.json` - Summary results

## Conclusions
"""
    
    # Add conclusions based on performance
    accuracy = results['basic_metrics']['accuracy']
    if accuracy >= 0.95:
        report_content += "🏆 **EXCELLENT**: Model achieves clinical-grade performance suitable for medical deployment.\n"
    elif accuracy >= 0.90:
        report_content += "✅ **VERY GOOD**: Model achieves research-grade performance suitable for specialist consultation.\n"
    elif accuracy >= 0.85:
        report_content += "⚠️ **ACCEPTABLE**: Model performance is minimally acceptable but needs improvement.\n"
    else:
        report_content += "❌ **NEEDS IMPROVEMENT**: Model performance below acceptable levels for medical use.\n"
    
    # Write report
    with open(report_path, 'w') as f:
        f.write(report_content)
    
    print(f"📋 Final report saved: {report_path}")

def main():
    """Main execution function"""
    parser = argparse.ArgumentParser(description='4-Class Brain Tumor Classification Training')
    parser.add_argument('--data', type=str, required=True, 
                       help='Path to dataset directory containing class folders')
    parser.add_argument('--gpu', action='store_true', 
                       help='Force GPU configuration')
    parser.add_argument('--no-eval', action='store_true',
                       help='Skip evaluation step')
    
    args = parser.parse_args()
    
    # Print header
    print("🧠 4-CLASS BRAIN TUMOR MULTICLASS CLASSIFICATION")
    print("=" * 80)
    print(f"📅 Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 80)
    
    # Setup GPU
    has_gpu = setup_gpu_config()
    if args.gpu and not has_gpu:
        print("⚠️ GPU requested but not available")
    
    # Validate dataset
    try:
        class_counts, total_images = validate_dataset_structure(args.data)
        if total_images == 0:
            print("❌ No images found in dataset")
            return
    except Exception as e:
        print(f"❌ Dataset validation failed: {e}")
        return
    
    # Create experiment directory
    exp_dir = create_experiment_directory()
    
    # Save experiment config
    config = {'data_path': args.data}
    save_experiment_config(exp_dir, config)
    
    # Train model
    model_trainer, test_data = train_model(args.data, exp_dir)
    
    # Evaluate model (unless skipped)
    results = None
    if not args.no_eval:
        results = evaluate_model(model_trainer, test_data, exp_dir)
    
    # Create final report
    if results:
        create_final_report(exp_dir, results)
    
    # Final summary
    print("\n" + "="*80)
    print("🏁 EXPERIMENT COMPLETED")
    print("="*80)
    print(f"📁 All results saved in: {exp_dir}")
    if results:
        print(f"🎯 Final Accuracy: {results['basic_metrics']['accuracy']:.4f} ({results['basic_metrics']['accuracy']*100:.2f}%)")
        print(f"🏥 Clinical Grade: {results['clinical_assessment']['clinical_grade']}")
        print(f"🚀 Deployment Ready: {'✅ YES' if results['clinical_assessment']['deployment_ready'] else '❌ NO'}")
    print("="*80)

if __name__ == "__main__":
    main()