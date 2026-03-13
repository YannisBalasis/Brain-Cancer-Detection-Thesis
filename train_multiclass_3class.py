#!/usr/bin/env python3
"""
🧠 3-Class Brain Tumor Multiclass Classification Training Pipeline
Tumor-only classification system για Glioma/Meningioma/Pituitary (No healthy brain)

Based on proven 4-class methodology adapted for tumor discrimination
"""

import os
import sys
import argparse
import json
from datetime import datetime
import tensorflow as tf
import numpy as np

# Import our custom module
from multiclass_3class_model import MultiClass3ClassModel

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

def validate_tumor_dataset_structure(data_path):
    """Validate dataset structure για 3-class tumor classification"""
    print(f"🔍 Validating 3-Class Tumor Dataset at: {data_path}")
    
    if not os.path.exists(data_path):
        raise ValueError(f"Dataset path does not exist: {data_path}")
    
    expected_classes = ['glioma', 'meningioma', 'pituitary']
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
    
    print(f"📊 Total tumor images: {total_images:,}")
    
    # Check για balance
    if total_images > 0:
        print("📊 Tumor class distribution:")
        for class_name, count in found_classes.items():
            percentage = (count / total_images) * 100
            print(f"  {class_name}: {percentage:.1f}%")
        
        # Check if reasonably balanced
        percentages = [count/total_images for count in found_classes.values() if count > 0]
        imbalance_ratio = max(percentages) / min(percentages) if len(percentages) > 1 else 1.0
        
        if imbalance_ratio > 2.5:
            print(f"⚠️ Tumor dataset appears imbalanced (ratio: {imbalance_ratio:.1f}:1)")
        else:
            print(f"✅ Tumor dataset reasonably balanced (ratio: {imbalance_ratio:.1f}:1)")
    
    return found_classes, total_images

def create_experiment_directory():
    """Create timestamped experiment directory"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    current_dir = os.getcwd()
    exp_dir = os.path.join(current_dir, f"multiclass_3class_experiment_{timestamp}")
    
    os.makedirs(exp_dir, exist_ok=True)
    print(f"📁 Created experiment directory: {exp_dir}")
    
    return exp_dir

def save_experiment_config(exp_dir, config):
    """Save experiment configuration"""
    config_path = os.path.join(exp_dir, "experiment_config.json")
    
    config_data = {
        'experiment_timestamp': datetime.now().isoformat(),
        'model_type': '3-Class Tumor Multiclass CNN',
        'dataset_path': config['data_path'],
        'classes': ['Glioma', 'Meningioma', 'Pituitary'],
        'architecture': 'Custom CNN (Based on proven 4-class)',
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
        'medical_focus': {
            'target': 'Tumor type discrimination only',
            'no_healthy_brain': 'Excluded για focused learning',
            'expected_performance': '92-96% accuracy'
        }
    }
    
    with open(config_path, 'w') as f:
        json.dump(config_data, f, indent=2)
    
    print(f"💾 Experiment config saved: {config_path}")
    return config_path

def train_tumor_model(data_path, exp_dir):
    """Complete 3-class tumor model training pipeline"""
    print("\n" + "="*80)
    print("🚀 STARTING 3-CLASS TUMOR MULTICLASS TRAINING")
    print("="*80)
    
    # Initialize model
    print("\n1️⃣ Initializing 3-Class Tumor Model...")
    model_trainer = MultiClass3ClassModel()
    
    # Create model architecture
    print("\n2️⃣ Building Architecture...")
    model_trainer.create_model()
    model_trainer.compile_model()
    
    # Load and preprocess data
    print("\n3️⃣ Loading Tumor Dataset...")
    try:
        X, y, class_counts = model_trainer.load_and_preprocess_data(data_path)
        print(f"✅ Successfully loaded {len(X):,} tumor images")
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
    print("\n5️⃣ Training Tumor Classification Model...")
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
    
    print("\n✅ 3-Class tumor training pipeline completed!")
    return model_trainer, (X_test, y_test)

def evaluate_tumor_model(model_trainer, test_data, exp_dir):
    """Comprehensive evaluation of tumor classification model"""
    print("\n" + "="*80)  
    print("📊 STARTING 3-CLASS TUMOR EVALUATION")
    print("="*80)
    
    if model_trainer is None or test_data is None:
        print("❌ Cannot evaluate: training failed ή no test data")
        return None
    
    X_test, y_test = test_data
    
    # Run evaluation
    try:
        results = model_trainer.evaluate_model(X_test, y_test, exp_dir)
        
        # Create visualizations
        model_trainer.plot_confusion_matrix(results['confusion_matrix'], exp_dir)
        
        # Save evaluation results in experiment directory
        eval_results_path = os.path.join(exp_dir, 'evaluation_results.json')
        eval_summary_path = os.path.join(exp_dir, 'evaluation_summary.json')
        
        with open(eval_results_path, 'w') as f:
            # Convert numpy arrays to lists για JSON serialization
            results_serializable = results.copy()
            results_serializable['predictions'] = results_serializable['predictions'].tolist()
            results_serializable['true_labels'] = results_serializable['true_labels'].tolist()
            results_serializable['prediction_probabilities'] = results_serializable['prediction_probabilities'].tolist()
            results_serializable['confusion_matrix'] = results_serializable['confusion_matrix'].tolist()
            json.dump(results_serializable, f, indent=2)
        
        # Create summary
        summary = {
            'experiment_date': datetime.now().isoformat(),
            'model_type': '3-Class Tumor Classification',
            'classes': ['Glioma', 'Meningioma', 'Pituitary'],
            'model_performance': {
                'test_accuracy': float(results['test_accuracy']),
                'top_2_accuracy': float(results['top_2_accuracy']), 
                'precision': float(results['precision']),
                'recall': float(results['recall'])
            },
            'tumor_focus': {
                'advantage': 'No healthy brain confusion',
                'challenge': 'Tumor type discrimination',
                'comparison_baseline': '4-class model tumor accuracy'
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
    
    report_content = f"""# 3-Class Tumor Classification - Experiment Report

## Experiment Information
- **Date**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
- **Model Type**: 3-Class Tumor Multiclass CNN (Glioma/Meningioma/Pituitary)
- **Architecture**: Custom CNN based on proven 4-class model
- **Focus**: Tumor type discrimination only (no healthy brain confusion)

## Performance Results

### Overall Performance
- **Test Accuracy**: {results['test_accuracy']:.4f} ({results['test_accuracy']*100:.2f}%)
- **Top-2 Accuracy**: {results['top_2_accuracy']:.4f} ({results['top_2_accuracy']*100:.2f}%)
- **Precision**: {results['precision']:.4f}
- **Recall**: {results['recall']:.4f}

### Tumor Classification Focus
- **Advantage**: No confusion με healthy brain tissue
- **Focus**: Pure tumor type discrimination
- **Dataset**: Only pathological cases (tumor images)

### Expected vs Actual
- **Hypothesis**: 3-class should perform better on tumor discrimination
- **Prediction**: 92-96% accuracy range
- **Actual Result**: {results['test_accuracy']*100:.2f}% accuracy
- **Validation**: {"✅ CONFIRMED" if results['test_accuracy'] >= 0.92 else "❌ BELOW EXPECTATION"}

## Comparison με 4-Class Model
- **4-Class Overall**: 98.72% (including No Tumor: 100%)
- **4-Class Tumor-only**: ~97.5% average (Glioma: 97.53%, Meningioma: 97.58%, Pituitary: 99.43%)
- **3-Class Result**: {results['test_accuracy']*100:.2f}%
- **Advantage**: {"✅ IMPROVED" if results['test_accuracy'] > 0.975 else "⚖️ COMPARABLE" if results['test_accuracy'] > 0.95 else "❌ DECREASED"}

## Technical Analysis
- **Architecture**: Same proven CNN structure
- **Parameters**: ~1.27M (slightly less than 4-class)
- **Training Focus**: Tumor discrimination only
- **Data Efficiency**: Higher tumor data density per class

## Medical Insights
- **Clinical Relevance**: Focused tumor type identification
- **Diagnostic Context**: After initial tumor detection
- **Specialist Use**: Oncology and neurosurgery planning
- **Treatment Impact**: Tumor-specific therapy selection

## Generated Files
- `best_multiclass_3class_model.h5` - Best trained model
- `multiclass_3class_training_history.png` - Training curves
- `multiclass_3class_confusion_matrix.png` - Confusion analysis
- `evaluation_results.json` - Detailed results
- `evaluation_summary.json` - Summary results

## Conclusions
"""
    
    # Add conclusions based on performance
    accuracy = results['test_accuracy']
    if accuracy >= 0.96:
        report_content += "🏆 **EXCELLENT**: 3-Class model achieves outstanding tumor discrimination performance.\n"
    elif accuracy >= 0.92:
        report_content += "✅ **VERY GOOD**: 3-Class model meets expectations για focused tumor classification.\n"
    elif accuracy >= 0.88:
        report_content += "⚠️ **ACCEPTABLE**: 3-Class performance acceptable but room για improvement.\n"
    else:
        report_content += "❌ **NEEDS IMPROVEMENT**: 3-Class performance below expectations.\n"
    
    report_content += f"\n**Key Finding**: Removing healthy brain class {'improved' if accuracy > 0.975 else 'maintained' if accuracy > 0.95 else 'decreased'} tumor discrimination performance.\n"
    
    # Write report
    with open(report_path, 'w') as f:
        f.write(report_content)
    
    print(f"📋 Final report saved: {report_path}")

def main():
    """Main execution function"""
    parser = argparse.ArgumentParser(description='3-Class Tumor Classification Training')
    parser.add_argument('--data', type=str, required=True, 
                       help='Path to dataset directory containing tumor class folders only')
    parser.add_argument('--gpu', action='store_true', 
                       help='Force GPU configuration')
    parser.add_argument('--no-eval', action='store_true',
                       help='Skip evaluation step')
    
    args = parser.parse_args()
    
    # Print header
    print("🧠 3-CLASS TUMOR MULTICLASS CLASSIFICATION")
    print("=" * 80)
    print(f"📅 Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("🎯 Focus: Tumor type discrimination only (Glioma/Meningioma/Pituitary)")
    print("=" * 80)
    
    # Setup GPU
    has_gpu = setup_gpu_config()
    if args.gpu and not has_gpu:
        print("⚠️ GPU requested but not available")
    
    # Validate dataset
    try:
        class_counts, total_images = validate_tumor_dataset_structure(args.data)
        if total_images == 0:
            print("❌ No tumor images found in dataset")
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
    model_trainer, test_data = train_tumor_model(args.data, exp_dir)
    
    # Evaluate model (unless skipped)
    results = None
    if not args.no_eval:
        results = evaluate_tumor_model(model_trainer, test_data, exp_dir)
    
    # Create final report
    if results:
        create_final_report(exp_dir, results)
    
    # Final summary
    print("\n" + "="*80)
    print("🏁 3-CLASS TUMOR EXPERIMENT COMPLETED")
    print("="*80)
    print(f"📁 All results saved in: {exp_dir}")
    if results:
        print(f"🎯 Final Accuracy: {results['test_accuracy']:.4f} ({results['test_accuracy']*100:.2f}%)")
        print(f"🎯 Top-2 Accuracy: {results['top_2_accuracy']:.4f} ({results['top_2_accuracy']*100:.2f}%)")
        comparison_4class = "IMPROVED" if results['test_accuracy'] > 0.975 else "COMPARABLE" if results['test_accuracy'] > 0.95 else "DECREASED"
        print(f"📊 vs 4-Class Tumor Performance: {comparison_4class}")
    print("="*80)

if __name__ == "__main__":
    main()