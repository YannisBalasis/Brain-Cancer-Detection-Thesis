#!/usr/bin/env python3
"""
🥊 1-vs-1 Binary Ensemble Brain Tumor Classification Training Pipeline
Three binary models με probability-based ensemble voting

Models: Glioma vs Meningioma, Glioma vs Pituitary, Meningioma vs Pituitary
Decision: Probability combination για final classification
"""

import os
import sys
import argparse
import json
from datetime import datetime
import tensorflow as tf
import numpy as np

# Import our custom module
from binary_ensemble_1vs1 import OneVsOneEnsemble

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

def validate_ensemble_dataset_structure(data_path):
    """Validate dataset structure για 1-vs-1 ensemble"""
    print(f"🔍 Validating 1-vs-1 Ensemble Dataset at: {data_path}")
    
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
    
    print(f"📊 Total images για ensemble: {total_images:,}")
    
    # Show binary pair distributions
    pairs = [('glioma', 'meningioma'), ('glioma', 'pituitary'), ('meningioma', 'pituitary')]
    print(f"\n🥊 Binary Model Distributions:")
    for class1, class2 in pairs:
        count1, count2 = found_classes[class1], found_classes[class2]
        total_pair = count1 + count2
        if total_pair > 0:
            ratio = max(count1, count2) / min(count1, count2) if min(count1, count2) > 0 else float('inf')
            print(f"  {class1} vs {class2}: {total_pair:,} samples (ratio: {ratio:.1f}:1)")
    
    return found_classes, total_images

def create_experiment_directory():
    """Create timestamped experiment directory"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    current_dir = os.getcwd()
    exp_dir = os.path.join(current_dir, f"binary_ensemble_1vs1_experiment_{timestamp}")
    
    os.makedirs(exp_dir, exist_ok=True)
    print(f"📁 Created experiment directory: {exp_dir}")
    
    return exp_dir

def save_experiment_config(exp_dir, config):
    """Save experiment configuration"""
    config_path = os.path.join(exp_dir, "experiment_config.json")
    
    config_data = {
        'experiment_timestamp': datetime.now().isoformat(),
        'model_type': '1-vs-1 Binary Ensemble',
        'dataset_path': config['data_path'],
        'approach': 'Three binary models με probability voting',
        'binary_models': [
            'Glioma vs Meningioma',
            'Glioma vs Pituitary', 
            'Meningioma vs Pituitary'
        ],
        'architecture': 'Custom CNN (Same as proven models)',
        'ensemble_method': 'Probability-based voting',
        'training_params': {
            'batch_size': 32,
            'max_epochs': 50,
            'patience': 15,
            'learning_rate': 0.001
        },
        'expected_advantage': 'Leverage excellent binary classification performance'
    }
    
    with open(config_path, 'w') as f:
        json.dump(config_data, f, indent=2)
    
    print(f"💾 Experiment config saved: {config_path}")
    return config_path

def train_ensemble_system(data_path, exp_dir):
    """Complete 1-vs-1 ensemble training pipeline"""
    print("\n" + "="*80)
    print("🥊 STARTING 1-VS-1 BINARY ENSEMBLE TRAINING")
    print("="*80)
    
    # Initialize ensemble
    print("\n1️⃣ Initializing 1-vs-1 Binary Ensemble...")
    ensemble = OneVsOneEnsemble()
    
    # Create all binary models
    print("\n2️⃣ Creating Binary Models...")
    ensemble.create_all_models()
    
    # Load and preprocess data
    print("\n3️⃣ Loading Dataset...")
    try:
        X, y, class_counts = ensemble.load_and_preprocess_data(data_path)
        print(f"✅ Successfully loaded {len(X):,} images για ensemble")
    except Exception as e:
        print(f"❌ Error loading data: {e}")
        return None, None
    
    # Create data splits
    print("\n4️⃣ Creating Data Splits...")
    X_train, X_val, X_test, y_train, y_val, y_test = ensemble.create_data_splits(X, y)
    
    # Save test data για later evaluation
    np.save(os.path.join(exp_dir, 'X_test.npy'), X_test)
    np.save(os.path.join(exp_dir, 'y_test.npy'), y_test)
    print(f"💾 Test data saved για ensemble evaluation")
    
    # Train all binary models
    print("\n5️⃣ Training All Binary Models...")
    start_time = datetime.now()
    
    try:
        training_results = ensemble.train_all_models(X_train, y_train, X_val, y_val, exp_dir)
        
        total_training_time = datetime.now() - start_time
        print(f"✅ All binary models trained in {total_training_time}")
        
        # Save training summary
        training_summary = {}
        for model_key, results in training_results.items():
            training_summary[model_key] = {
                'training_time': str(results['training_time']),
                'train_samples': results['train_samples'],
                'val_samples': results['val_samples'],
                'total_time': str(total_training_time)
            }
        
        summary_path = os.path.join(exp_dir, 'training_summary.json')
        with open(summary_path, 'w') as f:
            json.dump(training_summary, f, indent=2)
        
    except Exception as e:
        print(f"❌ Training failed: {e}")
        return None, None
    
    print("\n✅ 1-vs-1 ensemble training pipeline completed!")
    return ensemble, (X_test, y_test)

def evaluate_ensemble_system(ensemble, test_data, exp_dir):
    """Comprehensive evaluation of ensemble system"""
    print("\n" + "="*80)  
    print("📊 STARTING 1-VS-1 ENSEMBLE EVALUATION")
    print("="*80)
    
    if ensemble is None or test_data is None:
        print("❌ Cannot evaluate: training failed ή no test data")
        return None
    
    X_test, y_test = test_data
    
    # Run ensemble evaluation
    try:
        print("\n🔮 Making Ensemble Predictions...")
        results = ensemble.predict_ensemble(X_test, y_test, exp_dir)
        
        # Create visualizations
        ensemble.plot_ensemble_results(results, exp_dir)
        
        # Analyze individual binary model contributions
        print(f"\n🔍 Binary Model Analysis:")
        for model_key, binary_pred in results['binary_predictions'].items():
            class1, class2 = binary_pred['class1'], binary_pred['class2']
            probs = binary_pred['probabilities']
            avg_confidence = np.mean(np.maximum(probs.flatten(), 1 - probs.flatten()))
            print(f"  {model_key}: Average confidence = {avg_confidence:.4f}")
        
        # Save detailed evaluation results
        eval_results_path = os.path.join(exp_dir, 'ensemble_evaluation_results.json')
        
        # Prepare serializable results
        serializable_results = {
            'ensemble_accuracy': float(results['ensemble_accuracy']),
            'evaluation_timestamp': datetime.now().isoformat(),
            'model_type': '1-vs-1 Binary Ensemble',
            'binary_models': ['Glioma vs Meningioma', 'Glioma vs Pituitary', 'Meningioma vs Pituitary'],
            'ensemble_method': 'Probability-based voting',
            'confusion_matrix': results['confusion_matrix'].tolist(),
            'classification_report': results['classification_report'],
            'predictions': results['ensemble_predictions'].tolist(),
            'true_labels': results['true_labels'].tolist()
        }
        
        with open(eval_results_path, 'w') as f:
            json.dump(serializable_results, f, indent=2)
        
        # Create evaluation summary
        summary = {
            'experiment_date': datetime.now().isoformat(),
            'model_type': '1-vs-1 Binary Ensemble',
            'approach': 'Three binary models με probability voting',
            'ensemble_performance': {
                'test_accuracy': float(results['ensemble_accuracy']),
                'ensemble_method': 'Probability-based voting'
            },
            'binary_models_info': {
                'total_models': 3,
                'model_pairs': ['Glioma vs Meningioma', 'Glioma vs Pituitary', 'Meningioma vs Pituitary'],
                'combination_method': 'Average probabilities'
            }
        }
        
        summary_path = os.path.join(exp_dir, 'ensemble_evaluation_summary.json')
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)
        
        print(f"💾 Ensemble evaluation results saved in: {exp_dir}")
        return results
        
    except Exception as e:
        print(f"❌ Ensemble evaluation failed: {e}")
        return None

def create_final_ensemble_report(exp_dir, results):
    """Create final experiment report"""
    print("\n📋 Creating Final Ensemble Report...")
    
    report_path = os.path.join(exp_dir, 'ENSEMBLE_EXPERIMENT_REPORT.md')
    
    if results is None:
        print("⚠️ No results available για report")
        return
    
    accuracy = results['ensemble_accuracy']
    
    report_content = f"""# 1-vs-1 Binary Ensemble - Experiment Report

## Experiment Information
- **Date**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
- **Model Type**: 1-vs-1 Binary Ensemble
- **Architecture**: Three Binary CNNs με Probability Voting
- **Approach**: Pairwise binary classification με ensemble combination

## Binary Models
1. **Glioma vs Meningioma** - Binary CNN
2. **Glioma vs Pituitary** - Binary CNN  
3. **Meningioma vs Pituitary** - Binary CNN

## Ensemble Method
- **Combination**: Probability-based voting
- **Decision Logic**: Average probabilities from all relevant binary models
- **Final Prediction**: Highest combined probability

## Performance Results

### Ensemble Performance
- **Test Accuracy**: {accuracy:.4f} ({accuracy*100:.2f}%)
- **Ensemble Method**: Probability-based voting
- **Model Count**: 3 binary models

### Individual Binary Models
Each binary model trained independently on pairwise data:
- Same CNN architecture as proven models
- Binary classification optimization
- Individual model checkpointing

## Comparison με Previous Approaches
- **4-Class Multiclass**: 98.72% accuracy
- **3-Class Multiclass**: 97.81% accuracy  
- **1-vs-1 Ensemble**: {accuracy*100:.2f}% accuracy
- **Ranking**: {"🏆 NEW BEST" if accuracy > 0.9872 else "🥈 SECOND BEST" if accuracy > 0.9781 else "🥉 THIRD PLACE" if accuracy > 0.90 else "❌ BELOW EXPECTATIONS"}

## Technical Analysis
- **Architecture**: Same proven CNN structure για each binary model
- **Total Parameters**: ~3.8M (3 × 1.27M each)
- **Training Approach**: Independent binary classification
- **Ensemble Complexity**: High (3 separate models)

## Binary Classification Advantage
- **Leverages**: Excellent binary classification capability
- **Focus**: Each model specialized για 2-class discrimination  
- **Robustness**: Multiple independent decisions
- **Confidence**: Probability-based combination

## Medical Implications
- **Clinical Use**: Requires all 3 models για prediction
- **Deployment**: More complex but potentially more robust
- **Interpretation**: Each binary model provides specific insights
- **Confidence**: Multiple model agreement indicates reliability

## Generated Files
- `best_glioma_vs_meningioma_model.h5` - Binary Model 1
- `best_glioma_vs_pituitary_model.h5` - Binary Model 2  
- `best_meningioma_vs_pituitary_model.h5` - Binary Model 3
- `ensemble_confusion_matrix.png` - Final results visualization
- `ensemble_evaluation_results.json` - Detailed results
- `training_summary.json` - Training statistics

## Conclusions
"""
    
    # Add conclusions based on performance
    if accuracy >= 0.99:
        report_content += "🏆 **OUTSTANDING**: Binary ensemble achieves exceptional performance, setting new benchmark.\n"
    elif accuracy >= 0.985:
        report_content += "🥇 **EXCELLENT**: Binary ensemble matches/exceeds 4-class performance.\n"
    elif accuracy >= 0.975:
        report_content += "✅ **VERY GOOD**: Binary ensemble shows competitive performance.\n"
    elif accuracy >= 0.95:
        report_content += "⚠️ **GOOD**: Binary ensemble performs well but doesn't exceed multiclass.\n"
    else:
        report_content += "❌ **NEEDS IMPROVEMENT**: Binary ensemble underperforms compared to multiclass approaches.\n"
    
    # Comparison insight
    if accuracy > 0.9872:
        report_content += f"\n**Key Finding**: 1-vs-1 ensemble approach successfully leverages binary classification strength to achieve new SOTA performance ({accuracy*100:.2f}%).\n"
    elif accuracy > 0.9781:
        report_content += f"\n**Key Finding**: 1-vs-1 ensemble outperforms 3-class but not 4-class approach, showing mixed effectiveness.\n"
    else:
        report_content += f"\n**Key Finding**: 1-vs-1 ensemble does not outperform simpler multiclass approaches, suggesting multiclass optimization is superior.\n"
    
    # Write report
    with open(report_path, 'w') as f:
        f.write(report_content)
    
    print(f"📋 Final ensemble report saved: {report_path}")

def main():
    """Main execution function"""
    parser = argparse.ArgumentParser(description='1-vs-1 Binary Ensemble Training')
    parser.add_argument('--data', type=str, required=True, 
                       help='Path to dataset directory containing tumor class folders')
    parser.add_argument('--gpu', action='store_true', 
                       help='Force GPU configuration')
    parser.add_argument('--no-eval', action='store_true',
                       help='Skip evaluation step')
    
    args = parser.parse_args()
    
    # Print header
    print("🥊 1-VS-1 BINARY ENSEMBLE BRAIN TUMOR CLASSIFICATION")
    print("=" * 80)
    print(f"📅 Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("🎯 Approach: Three binary models με probability-based ensemble")
    print("🎯 Models: Glioma vs Meningioma, Glioma vs Pituitary, Meningioma vs Pituitary")
    print("=" * 80)
    
    # Setup GPU
    has_gpu = setup_gpu_config()
    if args.gpu and not has_gpu:
        print("⚠️ GPU requested but not available")
    
    # Validate dataset
    try:
        class_counts, total_images = validate_ensemble_dataset_structure(args.data)
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
    
    # Train ensemble
    ensemble, test_data = train_ensemble_system(args.data, exp_dir)
    
    # Evaluate ensemble (unless skipped)
    results = None
    if not args.no_eval:
        results = evaluate_ensemble_system(ensemble, test_data, exp_dir)
    
    # Create final report
    if results:
        create_final_ensemble_report(exp_dir, results)
    
    # Final summary
    print("\n" + "="*80)
    print("🏁 1-VS-1 BINARY ENSEMBLE EXPERIMENT COMPLETED")
    print("="*80)
    print(f"📁 All results saved in: {exp_dir}")
    if results:
        accuracy = results['ensemble_accuracy']
        print(f"🎯 Final Ensemble Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
        
        # Compare με previous approaches
        print(f"\n📊 Performance Comparison:")
        print(f"  4-Class Multiclass: 98.72%")
        print(f"  3-Class Multiclass: 97.81%")  
        print(f"  1-vs-1 Ensemble:   {accuracy*100:.2f}%")
        
        if accuracy > 0.9872:
            print(f"🏆 NEW BEST PERFORMANCE!")
        elif accuracy > 0.9781:
            print(f"🥈 Second best - beats 3-class")
        else:
            print(f"🥉 Third place - room for improvement")
    print("="*80)

if __name__ == "__main__":
    main()