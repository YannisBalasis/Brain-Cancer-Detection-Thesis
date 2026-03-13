#!/usr/bin/env python3
"""
Utility functions for Dual Branch System
Includes data loading, preprocessing, evaluation, visualization, and helper functions
"""

import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import cv2
from sklearn.metrics import (
    classification_report, confusion_matrix, 
    precision_recall_fscore_support, roc_curve, auc,
    precision_recall_curve
)
from sklearn.preprocessing import label_binarize
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.utils import to_categorical
import logging
from datetime import datetime
from config_dual import DATA_CONFIG, AUGMENTATION_CONFIG

# Set up logging
logger = logging.getLogger(__name__)

class DataLoader:
    """
    Data loading and preprocessing utilities for dual system
    """
    
    def __init__(self, data_path, config=None):
        """
        Initialize DataLoader
        
        Args:
            data_path: Path to dataset directory
            config: Data configuration dictionary
        """
        self.data_path = data_path
        self.config = config if config else DATA_CONFIG
        self.class_names = self.config['CLASS_NAMES']
        self.class_names_display = self.config['CLASS_NAMES_DISPLAY']
        self.image_size = self.config['IMAGE_SIZE'][:2]  # (height, width)
        
        logger.info(f"DataLoader initialized with path: {data_path}")
        logger.info(f"Classes: {self.class_names}")
    
    def validate_dataset_structure(self):
        """
        Validate dataset directory structure
        
        Returns:
            bool: True if structure is valid
        """
        if not os.path.exists(self.data_path):
            logger.error(f"Dataset path does not exist: {self.data_path}")
            return False
        
        missing_classes = []
        class_counts = {}
        
        for class_name in self.class_names:
            class_path = os.path.join(self.data_path, class_name)
            if not os.path.exists(class_path):
                missing_classes.append(class_name)
            else:
                # Count images in class directory
                image_files = [f for f in os.listdir(class_path) 
                             if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))]
                class_counts[class_name] = len(image_files)
        
        if missing_classes:
            logger.error(f"Missing class directories: {missing_classes}")
            return False
        
        logger.info("Dataset structure validation successful")
        logger.info("Class distribution:")
        total_images = sum(class_counts.values())
        for class_name, count in class_counts.items():
            percentage = (count / total_images) * 100
            logger.info(f"  {class_name}: {count} images ({percentage:.1f}%)")
        
        return True
    
    def create_data_generators(self, validation_split=0.2, test_split=0.1):
        """
        Create data generators for training, validation, and testing
        
        Args:
            validation_split: Fraction of training data for validation
            test_split: Fraction of data for testing
            
        Returns:
            tuple: (train_gen, val_gen, test_gen, steps_info)
        """
        # Validation
        if not self.validate_dataset_structure():
            raise ValueError("Invalid dataset structure")
        
        batch_size = self.config['BATCH_SIZE']
        
        # Data augmentation for training
        if AUGMENTATION_CONFIG['ENABLED']:
            train_datagen = ImageDataGenerator(
                rescale=1./255,
                rotation_range=AUGMENTATION_CONFIG['ROTATION_RANGE'],
                width_shift_range=AUGMENTATION_CONFIG['WIDTH_SHIFT_RANGE'],
                height_shift_range=AUGMENTATION_CONFIG['HEIGHT_SHIFT_RANGE'],
                zoom_range=AUGMENTATION_CONFIG['ZOOM_RANGE'],
                horizontal_flip=AUGMENTATION_CONFIG['HORIZONTAL_FLIP'],
                vertical_flip=AUGMENTATION_CONFIG['VERTICAL_FLIP'],
                fill_mode=AUGMENTATION_CONFIG['FILL_MODE'],
                brightness_range=AUGMENTATION_CONFIG['BRIGHTNESS_RANGE'],
                validation_split=validation_split + test_split
            )
        else:
            train_datagen = ImageDataGenerator(
                rescale=1./255,
                validation_split=validation_split + test_split
            )
        
        # Validation and test data (no augmentation)
        val_test_datagen = ImageDataGenerator(
            rescale=1./255,
            validation_split=test_split / (validation_split + test_split)
        )
        
        # Training generator
        train_generator = train_datagen.flow_from_directory(
            self.data_path,
            target_size=self.image_size,
            batch_size=batch_size,
            class_mode='categorical',
            subset='training',
            shuffle=True,
            seed=42
        )
        
        # Validation + Test generator (combined)
        val_test_generator = train_datagen.flow_from_directory(
            self.data_path,
            target_size=self.image_size,
            batch_size=batch_size,
            class_mode='categorical',
            subset='validation',
            shuffle=False,
            seed=42
        )
        
        # Split validation and test from the validation subset
        val_test_samples = val_test_generator.samples
        val_samples = int(val_test_samples * (validation_split / (validation_split + test_split)))
        
        # Create separate validation and test generators
        validation_generator = val_test_datagen.flow_from_directory(
            self.data_path,
            target_size=self.image_size,
            batch_size=batch_size,
            class_mode='categorical',
            subset='training',
            shuffle=False,
            seed=42
        )
        
        test_generator = val_test_datagen.flow_from_directory(
            self.data_path,
            target_size=self.image_size,
            batch_size=batch_size,
            class_mode='categorical',
            subset='validation',
            shuffle=False,
            seed=42
        )
        
        # Calculate steps
        steps_info = {
            'train_steps': train_generator.samples // batch_size,
            'val_steps': validation_generator.samples // batch_size,
            'test_steps': test_generator.samples // batch_size,
            'train_samples': train_generator.samples,
            'val_samples': validation_generator.samples,
            'test_samples': test_generator.samples
        }
        
        logger.info(f"Data generators created successfully")
        logger.info(f"Training samples: {steps_info['train_samples']}")
        logger.info(f"Validation samples: {steps_info['val_samples']}")
        logger.info(f"Test samples: {steps_info['test_samples']}")
        
        return train_generator, validation_generator, test_generator, steps_info

class ModelEvaluator:
    """
    Model evaluation utilities
    """
    
    def __init__(self, class_names=None):
        """
        Initialize ModelEvaluator
        
        Args:
            class_names: List of class names for display
        """
        self.class_names = class_names if class_names else DATA_CONFIG['CLASS_NAMES_DISPLAY']
    
    def evaluate_model(self, model, test_generator, steps):
        """
        Comprehensive model evaluation
        
        Args:
            model: Trained model
            test_generator: Test data generator
            steps: Number of steps for evaluation
            
        Returns:
            dict: Evaluation results
        """
        logger.info("Starting model evaluation...")
        
        # Get predictions
        test_generator.reset()
        predictions = model.predict(test_generator, steps=steps, verbose=1)
        predicted_classes = np.argmax(predictions, axis=1)
        
        # Get true labels
        true_classes = test_generator.classes[:len(predicted_classes)]
        
        # Calculate metrics
        accuracy = np.mean(predicted_classes == true_classes)
        precision, recall, f1, support = precision_recall_fscore_support(
            true_classes, predicted_classes, average=None
        )
        
        # Overall metrics
        precision_macro = np.mean(precision)
        recall_macro = np.mean(recall)
        f1_macro = np.mean(f1)
        
        # Confusion matrix
        cm = confusion_matrix(true_classes, predicted_classes)
        
        # Classification report
        class_report = classification_report(
            true_classes, predicted_classes,
            target_names=self.class_names,
            digits=4
        )
        
        results = {
            'accuracy': accuracy,
            'precision_macro': precision_macro,
            'recall_macro': recall_macro,
            'f1_macro': f1_macro,
            'precision_per_class': precision.tolist(),
            'recall_per_class': recall.tolist(),
            'f1_per_class': f1.tolist(),
            'support_per_class': support.tolist(),
            'confusion_matrix': cm.tolist(),
            'classification_report': class_report,
            'predictions': predictions.tolist(),
            'predicted_classes': predicted_classes.tolist(),
            'true_classes': true_classes.tolist()
        }
        
        logger.info(f"Evaluation completed. Accuracy: {accuracy:.4f}")
        
        return results
    
    def calculate_roc_curves(self, true_classes, predictions, n_classes):
        """
        Calculate ROC curves for multiclass classification
        
        Args:
            true_classes: True class labels
            predictions: Model prediction probabilities
            n_classes: Number of classes
            
        Returns:
            dict: ROC curve data
        """
        # Binarize labels
        y_true_bin = label_binarize(true_classes, classes=list(range(n_classes)))
        
        roc_data = {}
        
        # Calculate ROC for each class
        for i in range(n_classes):
            fpr, tpr, _ = roc_curve(y_true_bin[:, i], predictions[:, i])
            roc_auc = auc(fpr, tpr)
            
            roc_data[f'class_{i}'] = {
                'fpr': fpr.tolist(),
                'tpr': tpr.tolist(),
                'auc': roc_auc,
                'class_name': self.class_names[i]
            }
        
        return roc_data

class Visualizer:
    """
    Visualization utilities for dual system
    """
    
    def __init__(self, class_names=None):
        """
        Initialize Visualizer
        
        Args:
            class_names: List of class names for display
        """
        self.class_names = class_names if class_names else DATA_CONFIG['CLASS_NAMES_DISPLAY']
        plt.style.use('default')
        
    def plot_training_history(self, history, save_path=None):
        """
        Plot training history
        
        Args:
            history: Training history dictionary or Keras History object
            save_path: Path to save the plot
        """
        if hasattr(history, 'history'):
            hist_dict = history.history
        else:
            hist_dict = history
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Dual System Training History', fontsize=16, fontweight='bold')
        
        # Accuracy plot
        axes[0, 0].plot(hist_dict['accuracy'], label='Training Accuracy', linewidth=2)
        if 'val_accuracy' in hist_dict:
            axes[0, 0].plot(hist_dict['val_accuracy'], label='Validation Accuracy', linewidth=2)
        axes[0, 0].set_title('Model Accuracy', fontweight='bold')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Accuracy')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # Loss plot
        axes[0, 1].plot(hist_dict['loss'], label='Training Loss', linewidth=2)
        if 'val_loss' in hist_dict:
            axes[0, 1].plot(hist_dict['val_loss'], label='Validation Loss', linewidth=2)
        axes[0, 1].set_title('Model Loss', fontweight='bold')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Loss')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # Precision plot
        if 'precision' in hist_dict:
            axes[1, 0].plot(hist_dict['precision'], label='Training Precision', linewidth=2)
            if 'val_precision' in hist_dict:
                axes[1, 0].plot(hist_dict['val_precision'], label='Validation Precision', linewidth=2)
            axes[1, 0].set_title('Model Precision', fontweight='bold')
            axes[1, 0].set_xlabel('Epoch')
            axes[1, 0].set_ylabel('Precision')
            axes[1, 0].legend()
            axes[1, 0].grid(True, alpha=0.3)
        
        # Recall plot
        if 'recall' in hist_dict:
            axes[1, 1].plot(hist_dict['recall'], label='Training Recall', linewidth=2)
            if 'val_recall' in hist_dict:
                axes[1, 1].plot(hist_dict['val_recall'], label='Validation Recall', linewidth=2)
            axes[1, 1].set_title('Model Recall', fontweight='bold')
            axes[1, 1].set_xlabel('Epoch')
            axes[1, 1].set_ylabel('Recall')
            axes[1, 1].legend()
            axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Training history plot saved to: {save_path}")
        
        plt.show()
    
    def plot_confusion_matrix(self, cm, save_path=None, normalize=False):
        """
        Plot confusion matrix
        
        Args:
            cm: Confusion matrix
            save_path: Path to save the plot
            normalize: Whether to normalize the confusion matrix
        """
        if normalize:
            cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
            fmt = '.2f'
            title = 'Normalized Confusion Matrix'
        else:
            fmt = 'd'
            title = 'Confusion Matrix'
        
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt=fmt, cmap='Blues',
                   xticklabels=self.class_names,
                   yticklabels=self.class_names,
                   square=True, linewidths=0.5,
                   cbar_kws={'shrink': 0.8})
        
        plt.title(title, fontsize=16, fontweight='bold', pad=20)
        plt.xlabel('Predicted Label', fontsize=12, fontweight='bold')
        plt.ylabel('True Label', fontsize=12, fontweight='bold')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Confusion matrix saved to: {save_path}")
        
        plt.show()
    
    def plot_roc_curves(self, roc_data, save_path=None):
        """
        Plot ROC curves for all classes
        
        Args:
            roc_data: ROC curve data dictionary
            save_path: Path to save the plot
        """
        plt.figure(figsize=(10, 8))
        
        colors = ['blue', 'red', 'green', 'orange', 'purple']
        
        for i, (class_key, data) in enumerate(roc_data.items()):
            plt.plot(data['fpr'], data['tpr'], 
                    color=colors[i % len(colors)], linewidth=2,
                    label=f"{data['class_name']} (AUC = {data['auc']:.3f})")
        
        plt.plot([0, 1], [0, 1], 'k--', linewidth=1)
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate', fontsize=12, fontweight='bold')
        plt.ylabel('True Positive Rate', fontsize=12, fontweight='bold')
        plt.title('ROC Curves - Dual System', fontsize=16, fontweight='bold')
        plt.legend(loc="lower right")
        plt.grid(True, alpha=0.3)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"ROC curves saved to: {save_path}")
        
        plt.show()
    
    def plot_performance_comparison(self, comparison_data, save_path=None):
        """
        Plot performance comparison with other models
        
        Args:
            comparison_data: Dictionary with model names and accuracies
            save_path: Path to save the plot
        """
        models = list(comparison_data.keys())
        accuracies = [comparison_data[model]['accuracy'] * 100 for model in models]
        
        # Color code: dual system in different color
        colors = ['skyblue'] * len(models)
        for i, model in enumerate(models):
            if 'dual' in model.lower():
                colors[i] = 'orange'
        
        plt.figure(figsize=(12, 8))
        bars = plt.bar(models, accuracies, color=colors, alpha=0.8, edgecolor='black')
        
        # Add value labels on bars
        for bar, acc in zip(bars, accuracies):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                    f'{acc:.2f}%', ha='center', va='bottom', 
                    fontweight='bold', fontsize=10)
        
        plt.title('Model Performance Comparison', fontsize=16, fontweight='bold')
        plt.xlabel('Models', fontsize=12, fontweight='bold')
        plt.ylabel('Accuracy (%)', fontsize=12, fontweight='bold')
        plt.ylim(0, 105)
        plt.xticks(rotation=45, ha='right')
        plt.grid(True, alpha=0.3, axis='y')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Performance comparison saved to: {save_path}")
        
        plt.show()

def save_results(results, file_path):
    """
    Save evaluation results to JSON file
    
    Args:
        results: Results dictionary
        file_path: Path to save file
    """
    # Convert numpy arrays to lists for JSON serialization
    json_results = {}
    for key, value in results.items():
        if isinstance(value, np.ndarray):
            json_results[key] = value.tolist()
        else:
            json_results[key] = value
    
    with open(file_path, 'w') as f:
        json.dump(json_results, f, indent=2)
    
    logger.info(f"Results saved to: {file_path}")

def load_results(file_path):
    """
    Load evaluation results from JSON file
    
    Args:
        file_path: Path to results file
        
    Returns:
        dict: Results dictionary
    """
    with open(file_path, 'r') as f:
        results = json.load(f)
    
    logger.info(f"Results loaded from: {file_path}")
    return results

def create_directory_structure(base_path):
    """
    Create directory structure for experiment
    
    Args:
        base_path: Base experiment directory path
    """
    directories = [
        'models', 'logs', 'results', 'plots'
    ]
    
    for directory in directories:
        dir_path = os.path.join(base_path, directory)
        os.makedirs(dir_path, exist_ok=True)
    
    logger.info(f"Directory structure created at: {base_path}")

def setup_logging(log_file=None, level=logging.INFO):
    """
    Setup logging configuration
    
    Args:
        log_file: Path to log file
        level: Logging level
    """
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(log_file) if log_file else logging.NullHandler()
        ]
    )

def set_random_seeds(seed=42):
    """
    Set random seeds for reproducibility
    
    Args:
        seed: Random seed value
    """
    np.random.seed(seed)
    tf.random.set_seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    
    # Additional TensorFlow deterministic operations
    tf.config.experimental.enable_op_determinism()
    
    logger.info(f"Random seeds set to {seed}")

if __name__ == "__main__":
    # Test utilities
    print("Testing Dual System Utilities...")
    
    # Test configuration loading
    from config_dual import get_config
    config = get_config()
    print(f"Configuration loaded: {len(config)} sections")
    
    # Test data loader initialization
    try:
        # This would normally use a real dataset path
        data_loader = DataLoader("/fake/path")
        print("DataLoader initialized successfully")
    except Exception as e:
        print(f"DataLoader test skipped: {e}")
    
    print("Utilities test completed!")