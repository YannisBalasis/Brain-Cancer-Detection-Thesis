
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing.image import ImageDataGenerator

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    classification_report, confusion_matrix, 
    roc_auc_score, roc_curve, auc,
    precision_recall_curve, average_precision_score
)
import os
from pathlib import Path

class BinaryModelEvaluator:
    """
    Complete evaluation για το trained binary classification model
    """
    
    def __init__(self, model_path, data_dir, target_size=(224, 224), batch_size=32):
        """
        Αρχικοποίηση του evaluator
        
        Args:
            model_path: Path στο trained model (.h5 file)
            data_dir: Path στο dataset directory
            target_size: Target size για τις εικόνες
            batch_size: Batch size για evaluation
        """
        self.model_path = model_path
        self.data_dir = data_dir
        self.target_size = target_size
        self.batch_size = batch_size
        
        # Binary mapping (ίδιο με training)
        self.tumor_classes = ['glioma', 'meningioma', 'pituitary']
        self.no_tumor_classes = ['notumor']
        
        # Placeholders
        self.model = None
        self.test_generator = None
        self.predictions = None
        self.true_labels = None
        
        print(" Binary Model Evaluator Initialized")
        print(f" Model path: {model_path}")
        print(f" Data path: {data_dir}")
        print(f" Target size: {target_size}")
    
    def load_model(self):
        """
        Φορτώνει το trained model
        """
        print(f"\n Loading trained model...")
        
        if not os.path.exists(self.model_path):
            print(f" Model file not found: {self.model_path}")
            return None
        
        try:
            self.model = keras.models.load_model(self.model_path)
            print(f" Model loaded successfully!")
            print(f" Model parameters: {self.model.count_params():,}")
            
            return self.model
            
        except Exception as e:
            print(f" Error loading model: {e}")
            return None
    
    def create_test_dataframe(self):
        """
        Δημιουργεί DataFrame με test data (ίδιο split με training)
        """
        print(f"\n Creating test dataset...")
        
        image_paths = []
        binary_labels = []
        original_classes = []
        
        # Collect all data
        for split in ['Training', 'Testing']:
            split_dir = os.path.join(self.data_dir, split)
            
            if not os.path.exists(split_dir):
                continue
            
            for class_name in os.listdir(split_dir):
                class_path = os.path.join(split_dir, class_name)
                
                if not os.path.isdir(class_path):
                    continue
                
                # Binary label mapping
                if class_name.lower() in [c.lower() for c in self.tumor_classes]:
                    binary_label = 'tumor'
                elif class_name.lower() in [c.lower() for c in self.no_tumor_classes]:
                    binary_label = 'no_tumor'
                else:
                    binary_label = 'tumor'  # Default
                
                # Collect image paths
                for image_file in os.listdir(class_path):
                    if image_file.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
                        image_path = os.path.join(class_path, image_file)
                        image_paths.append(image_path)
                        binary_labels.append(binary_label)
                        original_classes.append(class_name)
        
        # Create DataFrame
        df = pd.DataFrame({
            'image_path': image_paths,
            'binary_label': binary_labels,
            'original_class': original_classes
        })
        
        # Same split as training (70/20/10)
        train_df, temp_df = train_test_split(
            df, test_size=0.3, stratify=df['binary_label'], random_state=123
        )
        val_df, test_df = train_test_split(
            temp_df, test_size=0.33, stratify=temp_df['binary_label'], random_state=123
        )
        
        print(f" Test dataset created: {len(test_df)} images")
        print(f" Test distribution:")
        print(test_df['binary_label'].value_counts())
        
        return test_df
    
    def create_test_generator(self, test_df):
        """
        Δημιουργεί test generator (χωρίς augmentation)
        """
        print(f"\n Creating test generator...")
        
        test_datagen = ImageDataGenerator(rescale=1./255)
        
        self.test_generator = test_datagen.flow_from_dataframe(
            test_df,
            x_col='image_path',
            y_col='binary_label',
            target_size=self.target_size,
            batch_size=self.batch_size,
            class_mode='binary',
            shuffle=False,  # Σημαντικό για evaluation
            seed=42
        )
        
        print(f" Test generator created!")
        print(f" Test samples: {self.test_generator.samples}")
        print(f" Class indices: {self.test_generator.class_indices}")
        
        return self.test_generator
    
    def make_predictions(self):
        """
        Κάνει predictions στο test set
        """
        print(f"\n Making predictions on test set...")
        
        if self.model is None:
            print(" Model not loaded. Call load_model() first.")
            return None, None
        
        if self.test_generator is None:
            print(" Test generator not created. Call create_test_generator() first.")
            return None, None
        
        # Predictions (probabilities)
        self.predictions = self.model.predict(self.test_generator, verbose=1)
        
        # True labels
        self.true_labels = np.array(self.test_generator.classes)
        
        # Binary predictions (threshold = 0.5)
        binary_predictions = (self.predictions > 0.5).astype(int)
        
        print(f" Predictions completed!")
        print(f" Prediction shape: {self.predictions.shape}")
        print(f" True labels shape: {self.true_labels.shape}")
        print(f" Prediction range: [{self.predictions.min():.3f}, {self.predictions.max():.3f}]")
        
        return self.predictions, self.true_labels
    
    def calculate_metrics(self):
        """
        Υπολογίζει όλα τα evaluation metrics
        """
        print(f"\n Calculating evaluation metrics...")
        
        if self.predictions is None or self.true_labels is None:
            print(" No predictions found. Run make_predictions() first.")
            return None
        
        # Binary predictions
        binary_preds = (self.predictions > 0.5).astype(int).flatten()
        
        # Calculate metrics
        accuracy = np.mean(binary_preds == self.true_labels)
        
        # Classification report
        report = classification_report(
            self.true_labels, 
            binary_preds, 
            target_names=['No Tumor', 'Tumor'],
            output_dict=True
        )
        
        # Confusion matrix
        cm = confusion_matrix(self.true_labels, binary_preds)
        
        # ROC AUC
        roc_auc = roc_auc_score(self.true_labels, self.predictions)
        
        # Precision-Recall AUC
        pr_auc = average_precision_score(self.true_labels, self.predictions)
        
        metrics = {
            'accuracy': accuracy,
            'classification_report': report,
            'confusion_matrix': cm,
            'roc_auc': roc_auc,
            'pr_auc': pr_auc,
            'binary_predictions': binary_preds,
            'prediction_probabilities': self.predictions.flatten()
        }
        
        # Print summary
        print(f" Test Results Summary:")
        print(f"    Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
        print(f"    Precision (No Tumor): {report['No Tumor']['precision']:.4f}")
        print(f"    Precision (Tumor): {report['Tumor']['precision']:.4f}")
        print(f"    Recall (No Tumor): {report['No Tumor']['recall']:.4f}")
        print(f"    Recall (Tumor): {report['Tumor']['recall']:.4f}")
        print(f"    F1-Score (No Tumor): {report['No Tumor']['f1-score']:.4f}")
        print(f"    F1-Score (Tumor): {report['Tumor']['f1-score']:.4f}")
        print(f"    ROC-AUC: {roc_auc:.4f}")
        print(f"    PR-AUC: {pr_auc:.4f}")
        
        return metrics
    
    def plot_confusion_matrix(self, cm, save_path='confusion_matrix_test.png'):
        """
        Plot confusion matrix
        """
        plt.figure(figsize=(8, 6))
        
        # Normalized και raw confusion matrix
        cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        
        # Create annotation με counts και percentages
        annot = np.empty_like(cm).astype(str)
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                count = cm[i, j]
                pct = cm_norm[i, j]
                annot[i, j] = f'{count}\n({pct:.1%})'
        
        sns.heatmap(cm, annot=annot, fmt='', cmap='Blues', 
                   xticklabels=['No Tumor', 'Tumor'], 
                   yticklabels=['No Tumor', 'Tumor'],
                   cbar_kws={'label': 'Count'})
        
        plt.title('Test Set Confusion Matrix\n(Count and Percentage)', fontsize=14, fontweight='bold')
        plt.xlabel('Predicted Label', fontsize=12)
        plt.ylabel('True Label', fontsize=12)
        
        # Add accuracy text
        accuracy = np.trace(cm) / np.sum(cm)
        plt.text(0.02, 0.98, f'Accuracy: {accuracy:.3f}', 
                transform=plt.gca().transAxes, fontsize=12,
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f" Confusion matrix saved: {save_path}")
        plt.show()
    
    def plot_roc_curve(self, save_path='roc_curve_test.png'):
        """
        Plot ROC curve
        """
        # Calculate ROC curve
        fpr, tpr, thresholds = roc_curve(self.true_labels, self.predictions)
        roc_auc = auc(fpr, tpr)
        
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, color='darkorange', lw=2, 
                label=f'ROC curve (AUC = {roc_auc:.4f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', 
                label='Random classifier')
        
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate', fontsize=12)
        plt.ylabel('True Positive Rate', fontsize=12)
        plt.title('ROC Curve - Test Set', fontsize=14, fontweight='bold')
        plt.legend(loc="lower right")
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f" ROC curve saved: {save_path}")
        plt.show()
    
    def plot_precision_recall_curve(self, save_path='precision_recall_curve_test.png'):
        """
        Plot Precision-Recall curve
        """
        precision, recall, thresholds = precision_recall_curve(self.true_labels, self.predictions)
        pr_auc = average_precision_score(self.true_labels, self.predictions)
        
        plt.figure(figsize=(8, 6))
        plt.plot(recall, precision, color='darkorange', lw=2,
                label=f'PR curve (AUC = {pr_auc:.4f})')
        
        # Baseline (random classifier)
        no_skill = len(self.true_labels[self.true_labels == 1]) / len(self.true_labels)
        plt.plot([0, 1], [no_skill, no_skill], linestyle='--', 
                label=f'Random classifier (AP = {no_skill:.4f})')
        
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('Recall', fontsize=12)
        plt.ylabel('Precision', fontsize=12)
        plt.title('Precision-Recall Curve - Test Set', fontsize=14, fontweight='bold')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f" Precision-Recall curve saved: {save_path}")
        plt.show()
    
    def analyze_misclassified(self, test_df, max_examples=8):
        """
        Αναλύει misclassified examples
        """
        print(f"\n Analyzing misclassified examples...")
        
        binary_preds = (self.predictions > 0.5).astype(int).flatten()
        
        # Find misclassified indices
        misclassified_mask = binary_preds != self.true_labels
        misclassified_indices = np.where(misclassified_mask)[0]
        
        if len(misclassified_indices) == 0:
            print(" No misclassified examples found! Perfect performance!")
            return
        
        print(f" Found {len(misclassified_indices)} misclassified examples")
        print(f" Error rate: {len(misclassified_indices)/len(self.true_labels)*100:.2f}%")
        
        # Get details for misclassified examples
        misclassified_data = []
        for idx in misclassified_indices:
            file_idx = idx  # Assuming test_generator is not shuffled
            true_label = self.true_labels[idx]
            pred_prob = self.predictions[idx][0]
            pred_label = binary_preds[idx]
            
            misclassified_data.append({
                'index': idx,
                'true_label': 'Tumor' if true_label == 1 else 'No Tumor',
                'predicted_label': 'Tumor' if pred_label == 1 else 'No Tumor',
                'confidence': pred_prob if pred_label == 1 else (1 - pred_prob),
                'probability': pred_prob
            })
        
        # Sort by confidence (most confident wrong predictions first)
        misclassified_data.sort(key=lambda x: x['confidence'], reverse=True)
        
        print(f"\n Top misclassified examples (sorted by confidence):")
        for i, data in enumerate(misclassified_data[:max_examples]):
            print(f"   {i+1}. True: {data['true_label']}, "
                  f"Predicted: {data['predicted_label']}, "
                  f"Confidence: {data['confidence']:.1%}")
        
        return misclassified_data


def main():
    """
    Κύρια συνάρτηση evaluation pipeline
    """
    print(" Binary Brain Tumor Model Evaluation - Step 4")
    print("=" * 60)
    
    # Paths - ΕΔΩ αλλάζεις τα paths σου
    model_path = "best_binary_model.h5"  # Από το training
    data_dir = "/Users/yannisbalasis/Documents/thesis/dataset_binary_class"
    
    # Initialize evaluator
    evaluator = BinaryModelEvaluator(
        model_path=model_path,
        data_dir=data_dir,
        target_size=(224, 224),
        batch_size=32
    )
    
    # Step 1: Load trained model
    model = evaluator.load_model()
    if model is None:
        print(" Cannot proceed without model. Check model path.")
        return
    
    # Step 2: Create test dataset
    test_df = evaluator.create_test_dataframe()
    
    # Step 3: Create test generator
    test_gen = evaluator.create_test_generator(test_df)
    
    # Step 4: Make predictions
    predictions, true_labels = evaluator.make_predictions()
    
    # Step 5: Calculate metrics
    metrics = evaluator.calculate_metrics()
    
    # Step 6: Create visualizations
    print(f"\n Creating evaluation visualizations...")
    evaluator.plot_confusion_matrix(metrics['confusion_matrix'])
    evaluator.plot_roc_curve()
    evaluator.plot_precision_recall_curve()
    
    # Step 7: Analyze misclassified examples
    misclassified = evaluator.analyze_misclassified(test_df)
    
    print(f"\n Step 4 Complete!")
    print(f" Generated files:")
    print(f"    confusion_matrix_test.png")
    print(f"    roc_curve_test.png")
    print(f"    precision_recall_curve_test.png")
    print(f" Binary classification evaluation completed!")


if __name__ == "__main__":
    main()