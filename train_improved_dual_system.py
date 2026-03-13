#!/usr/bin/env python3
"""
🚀 Training Script για Improved Dual Branch System
Complete pipeline από data loading μέχρι evaluation
"""

import os
import sys
import argparse
import numpy as np
import cv2
from datetime import datetime
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
import tensorflow as tf

# Import our improved dual system
from improved_dual_branch_system import ImprovedDualBranchModel

def load_dataset(data_path, seed=789):
    """
    Load και preprocess το 4-class dataset
    Using new seed για improved dual system
    """
    print(f"\n📁 Loading Dataset από: {data_path}")
    print(f"🎲 Using seed: {seed} (different από previous models)")
    
    images = []
    labels = []
    class_counts = {}
    
    # Class mapping
    class_to_idx = {
        'glioma': 0,
        'meningioma': 1, 
        'no_tumor': 2,
        'pituitary': 3
    }
    
    class_names = ['Glioma', 'Meningioma', 'No Tumor', 'Pituitary']
    
    for class_name, class_idx in class_to_idx.items():
        class_path = os.path.join(data_path, class_name)
        if not os.path.exists(class_path):
            print(f"⚠️ Warning: {class_name} folder not found")
            continue
        
        image_files = [f for f in os.listdir(class_path) 
                      if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        
        print(f"📊 Loading {class_name}: {len(image_files)} images")
        
        for img_file in image_files:
            img_path = os.path.join(class_path, img_file)
            try:
                img = cv2.imread(img_path)
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                img = cv2.resize(img, (224, 224))
                
                images.append(img)
                labels.append(class_idx)
                
            except Exception as e:
                print(f"⚠️ Error loading {img_path}: {e}")
                continue
        
        class_counts[class_name] = len([l for l in labels if l == class_idx])
    
    X = np.array(images, dtype=np.float32)
    y = np.array(labels)
    
    print(f"\n📊 Dataset Statistics:")
    print(f"  Total images: {len(X):,}")
    for class_name, count in class_counts.items():
        percentage = (count / len(X)) * 100
        print(f"  {class_name.capitalize()}: {count:,} images ({percentage:.1f}%)")
    
    # Convert to categorical
    y_categorical = to_categorical(y, num_classes=4)
    
    # Create splits με new seed
    print(f"\n✂️ Creating Data Splits με seed={seed}:")
    
    # First split: test set
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y_categorical, test_size=0.1, 
        stratify=y, random_state=seed
    )
    
    # Second split: train/val
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=0.2, 
        stratify=y_temp.argmax(axis=1), random_state=seed
    )
    
    print(f"  Training: {len(X_train):,} images")
    print(f"  Validation: {len(X_val):,} images")
    print(f"  Test: {len(X_test):,} images")
    
    return X_train, X_val, X_test, y_train, y_val, y_test, class_names

def create_experiment_directory():
    """Create experiment directory με timestamp"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    experiment_dir = f"improved_dual_experiment_{timestamp}"
    os.makedirs(experiment_dir, exist_ok=True)
    
    print(f"\n📁 Created experiment directory: {experiment_dir}")
    return experiment_dir

def train_improved_dual_system(data_path):
    """Main training function"""
    print("🚀 IMPROVED DUAL BRANCH SYSTEM TRAINING")
    print("=" * 60)
    print(f"📅 Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("🎯 Goal: Surpass previous dual systems (>91.27%)")
    
    # Create experiment directory
    experiment_dir = create_experiment_directory()
    
    # Load dataset
    X_train, X_val, X_test, y_train, y_val, y_test, class_names = load_dataset(data_path)
    
    # Save test data για evaluation
    test_data_path = os.path.join(experiment_dir, 'test_data.npz')
    np.savez(test_data_path, X_test=X_test, y_test=y_test)
    print(f"💾 Test data saved: {test_data_path}")
    
    # Initialize model
    print("\n🏗️ Initializing Improved Dual Branch Model...")
    model = ImprovedDualBranchModel(input_shape=(224, 224, 3), num_classes=4)
    
    # Create architecture
    model.create_improved_dual_system()
    
    # Compile model
    model.compile_model()
    
    # Train model
    history = model.train_model(X_train, y_train, X_val, y_val, experiment_dir)
    
    # Save model and config
    final_model_path = model.save_model_and_config(experiment_dir)
    
    # Quick evaluation
    print("\n📊 Quick Validation Evaluation:")
    val_loss, val_acc, val_top2, val_precision, val_recall, val_f1 = model.model.evaluate(
        X_val / 255.0, y_val, verbose=0
    )
    
    print(f"  Validation Accuracy: {val_acc:.4f} ({val_acc*100:.2f}%)")
    print(f"  Validation Top-2: {val_top2:.4f} ({val_top2*100:.2f}%)")
    print(f"  Validation Precision: {val_precision:.4f}")
    print(f"  Validation Recall: {val_recall:.4f}")
    print(f"  Validation F1-Score: {val_f1:.4f}")
    
    # Performance comparison
    print(f"\n📈 Performance Comparison:")
    print(f"  Previous EfficientNet Dual: 91.27% (test)")
    print(f"  Previous ResNet Dual: 85.45% (test)")
    print(f"  Improved Dual System: {val_acc*100:.2f}% (validation)")
    
    if val_acc > 0.95:
        print("🎉 EXCELLENT! Validation >95% - looking promising!")
    elif val_acc > 0.92:
        print("✅ GOOD! Validation >92% - improvement achieved!")
    else:
        print("⚠️ MODERATE - may need further optimization")
    
    # Create training plots
    create_training_plots(history, experiment_dir)
    
    print(f"\n✅ Training Complete!")
    print(f"📁 All files saved in: {experiment_dir}")
    print(f"🎯 Best model: {final_model_path}")
    print("\n🚀 Next: Run evaluation με test set για final accuracy!")
    
    return experiment_dir, final_model_path

def create_training_plots(history, experiment_dir):
    """Create training visualization plots"""
    import matplotlib.pyplot as plt
    
    print("\n📈 Creating Training Visualizations...")
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
    
    # Accuracy
    ax1.plot(history.history['accuracy'], label='Training', linewidth=2)
    ax1.plot(history.history['val_accuracy'], label='Validation', linewidth=2)
    ax1.set_title('Improved Dual System - Accuracy', fontweight='bold')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Accuracy')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Loss
    ax2.plot(history.history['loss'], label='Training', linewidth=2)
    ax2.plot(history.history['val_loss'], label='Validation', linewidth=2)
    ax2.set_title('Improved Dual System - Loss', fontweight='bold')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Loss')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # F1 Score
    ax3.plot(history.history['f1_score'], label='Training F1', linewidth=2)
    ax3.plot(history.history['val_f1_score'], label='Validation F1', linewidth=2)
    ax3.set_title('Improved Dual System - F1 Score', fontweight='bold')
    ax3.set_xlabel('Epoch')
    ax3.set_ylabel('F1 Score')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # Learning Rate
    if 'lr' in history.history:
        ax4.plot(history.history['lr'], linewidth=2, color='red')
        ax4.set_title('Learning Rate Schedule', fontweight='bold')
        ax4.set_xlabel('Epoch')
        ax4.set_ylabel('Learning Rate')
        ax4.set_yscale('log')
        ax4.grid(True, alpha=0.3)
    else:
        ax4.text(0.5, 0.5, 'Learning Rate\nSchedule', 
                transform=ax4.transAxes, ha='center', va='center')
    
    plt.tight_layout()
    plot_path = os.path.join(experiment_dir, 'improved_dual_training_plots.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"💾 Training plots saved: {plot_path}")

def main():
    """Main function με command line arguments"""
    parser = argparse.ArgumentParser(description='Train Improved Dual Branch System')
    parser.add_argument('--data', required=True, 
                       help='Path to 4-class dataset directory')
    
    args = parser.parse_args()
    
    # Check data path
    if not os.path.exists(args.data):
        print(f"❌ Error: Data path does not exist: {args.data}")
        sys.exit(1)
    
    # Start training
    experiment_dir, model_path = train_improved_dual_system(args.data)
    
    print(f"\n🎯 TRAINING SUMMARY:")
    print(f"  Experiment: {experiment_dir}")
    print(f"  Best Model: {model_path}")
    print(f"  Next Step: Run evaluation για final test accuracy")

if __name__ == "__main__":
    main()