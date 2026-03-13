"""
Multiclass Brain Tumor CNN V2
============================
Clean, purpose-built 4-class CNN για brain tumor classification:
- Glioma (Class 0)
- Meningioma (Class 1) 
- Pituitary (Class 2)
- No Tumor (Class 3)

Strategy:
- Custom CNN architecture (no transfer learning drama)
- Proven techniques από το successful binary model
- Balanced design για all 4 classes
- Medical-grade reliability focus

Author: Διπλωματική Εργασία - Multiclass V2 (Clean Rebuild)
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import (
    Conv2D, MaxPooling2D, Dense, Dropout, BatchNormalization,
    GlobalAveragePooling2D, Input, Flatten
)
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import classification_report, confusion_matrix

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from pathlib import Path

class MultiClassBrainTumorCNNV2:
    """
    Purpose-built 4-class CNN για brain tumor classification
    """
    
    def __init__(self, input_shape=(224, 224, 3), num_classes=4):
        """
        Αρχικοποίηση του V2 multiclass model
        """
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.model = None
        
        # Class configuration
        self.class_names = ['glioma', 'meningioma', 'pituitary', 'notumor']
        self.class_labels = ['Glioma', 'Meningioma', 'Pituitary', 'No Tumor']
        
        print("🧠 Multiclass Brain Tumor CNN V2")
        print("=" * 40)
        print(f"🎯 Classes: {', '.join(self.class_labels)}")
        print(f"📏 Input shape: {input_shape}")
        print(f"🔧 Strategy: Clean custom CNN")
        print(f"✅ No transfer learning complications")
    
    def create_robust_cnn_architecture(self):
        """
        Creates robust custom CNN architecture
        Inspired by the successful binary model but adapted για 4-class
        """
        print(f"\n🤖 Creating robust 4-class CNN architecture...")
        
        # Input layer
        inputs = Input(shape=self.input_shape, name='input_mri')
        
        # Block 1: Initial feature extraction
        x = Conv2D(32, (3, 3), activation='relu', padding='same', name='conv1_1')(inputs)
        x = BatchNormalization(name='bn1_1')(x)
        x = Conv2D(32, (3, 3), activation='relu', padding='same', name='conv1_2')(x)
        x = BatchNormalization(name='bn1_2')(x)
        x = MaxPooling2D((2, 2), name='pool1')(x)
        x = Dropout(0.25, name='dropout1')(x)
        
        # Block 2: Deeper features
        x = Conv2D(64, (3, 3), activation='relu', padding='same', name='conv2_1')(x)
        x = BatchNormalization(name='bn2_1')(x)
        x = Conv2D(64, (3, 3), activation='relu', padding='same', name='conv2_2')(x)
        x = BatchNormalization(name='bn2_2')(x)
        x = MaxPooling2D((2, 2), name='pool2')(x)
        x = Dropout(0.25, name='dropout2')(x)
        
        # Block 3: Complex patterns
        x = Conv2D(128, (3, 3), activation='relu', padding='same', name='conv3_1')(x)
        x = BatchNormalization(name='bn3_1')(x)
        x = Conv2D(128, (3, 3), activation='relu', padding='same', name='conv3_2')(x)
        x = BatchNormalization(name='bn3_2')(x)
        x = MaxPooling2D((2, 2), name='pool3')(x)
        x = Dropout(0.3, name='dropout3')(x)
        
        # Block 4: High-level features
        x = Conv2D(256, (3, 3), activation='relu', padding='same', name='conv4_1')(x)
        x = BatchNormalization(name='bn4_1')(x)
        x = Conv2D(256, (3, 3), activation='relu', padding='same', name='conv4_2')(x)
        x = BatchNormalization(name='bn4_2')(x)
        x = MaxPooling2D((2, 2), name='pool4')(x)
        x = Dropout(0.3, name='dropout4')(x)
        
        # Global feature extraction
        x = GlobalAveragePooling2D(name='global_avg_pool')(x)
        
        # Classification head - designed για 4-class balance
        x = Dense(512, activation='relu', name='fc1')(x)
        x = BatchNormalization(name='bn_fc1')(x)
        x = Dropout(0.5, name='dropout_fc1')(x)
        
        x = Dense(256, activation='relu', name='fc2')(x)
        x = BatchNormalization(name='bn_fc2')(x)
        x = Dropout(0.4, name='dropout_fc2')(x)
        
        x = Dense(128, activation='relu', name='fc3')(x)
        x = BatchNormalization(name='bn_fc3')(x)
        x = Dropout(0.3, name='dropout_fc3')(x)
        
        # Final classification layer
        outputs = Dense(self.num_classes, activation='softmax', name='multiclass_prediction')(x)
        
        # Create model
        self.model = keras.Model(inputs=inputs, outputs=outputs, name='BrainTumorCNN_V2')
        
        # Model summary
        total_params = self.model.count_params()
        print(f"✅ Robust CNN V2 created!")
        print(f"📊 Total parameters: {total_params:,}")
        print(f"🎯 Architecture: 4 conv blocks + 3 dense layers")
        print(f"🔧 Features: BatchNorm + Dropout + GlobalAvgPool")
        
        return self.model
    
    def compile_model(self, learning_rate=0.001):
        """
        Compiles model με appropriate settings για 4-class
        """
        print(f"\n⚙️ Compiling model για 4-class classification...")
        
        self.model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
            loss='categorical_crossentropy',
            metrics=[
                'accuracy',
                'top_k_categorical_accuracy',
                keras.metrics.Precision(name='precision'),
                keras.metrics.Recall(name='recall')
            ]
        )
        
        print(f"✅ Model compiled!")
        print(f"🎯 Optimizer: Adam (lr={learning_rate})")
        print(f"📊 Loss: Categorical Crossentropy")
        print(f"📈 Metrics: Accuracy, Top-K, Precision, Recall")
        
        return self.model
    
    def print_model_summary(self):
        """
        Prints detailed model summary
        """
        print(f"\n📋 MODEL ARCHITECTURE SUMMARY")
        print("=" * 45)
        
        self.model.summary()
        
        # Calculate memory usage
        total_params = self.model.count_params()
        memory_mb = (total_params * 4) / (1024 * 1024)  # 4 bytes per param
        
        print(f"\n📊 Model Statistics:")
        print(f"   🎯 Total Parameters: {total_params:,}")
        print(f"   💾 Estimated Memory: {memory_mb:.1f} MB")
        print(f"   🏗️ Architecture Type: Custom CNN")
        print(f"   🎯 Classes: {self.num_classes}")


class MultiClassTrainerV2:
    """
    Training pipeline για το V2 multiclass model
    """
    
    def __init__(self, data_dir, target_size=(224, 224), batch_size=32):
        """
        Αρχικοποίηση trainer
        """
        self.data_dir = data_dir
        self.target_size = target_size
        self.batch_size = batch_size
        
        # Class configuration
        self.class_names = ['glioma', 'meningioma', 'pituitary', 'notumor']
        self.class_labels = ['Glioma', 'Meningioma', 'Pituitary', 'No Tumor']
        self.num_classes = 4
        
        # Training parameters - conservative but effective
        self.epochs = 50
        self.initial_lr = 0.001
        self.patience_early_stop = 15
        self.patience_lr_reduce = 8
        
        print("🚀 Multiclass Training Pipeline V2")
        print("=" * 45)
        print(f"📁 Data directory: {data_dir}")
        print(f"🎯 Strategy: Single-phase robust training")
        print(f"⏰ Max epochs: {self.epochs}")
        print(f"🎯 Learning rate: {self.initial_lr}")
    
    def create_multiclass_dataset(self):
        """
        Creates complete 4-class dataset
        """
        print(f"\n📊 Creating complete 4-class dataset...")
        
        image_paths = []
        labels = []
        
        # Collect ALL classes (including healthy brains)
        for split in ['Training', 'Testing']:
            split_dir = os.path.join(self.data_dir, split)
            
            if not os.path.exists(split_dir):
                continue
                
            for class_name in os.listdir(split_dir):
                class_path = os.path.join(split_dir, class_name)
                
                if not os.path.isdir(class_path):
                    continue
                
                normalized_class = class_name.lower()
                if normalized_class not in self.class_names:
                    print(f"⚠️ Skipping unknown class: {normalized_class}")
                    continue
                
                # Collect all images
                for image_file in os.listdir(class_path):
                    if image_file.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
                        image_path = os.path.join(class_path, image_file)
                        image_paths.append(image_path)
                        labels.append(normalized_class)
        
        # Create DataFrame
        df = pd.DataFrame({
            'image_path': image_paths,
            'class_label': labels
        })
        
        # Add numeric labels
        label_to_num = {name: i for i, name in enumerate(self.class_names)}
        df['numeric_label'] = df['class_label'].map(label_to_num)
        
        print(f"✅ Complete 4-class dataset created: {len(df)} images")
        print(f"📊 Class distribution:")
        class_counts = df['class_label'].value_counts()
        for class_name in self.class_names:
            count = class_counts.get(class_name, 0)
            percentage = (count / len(df)) * 100
            print(f"   {class_name:12s}: {count:4d} images ({percentage:5.1f}%)")
        
        return df
    
    def create_balanced_splits(self, df):
        """
        Creates balanced train/val/test splits
        """
        print(f"\n📊 Creating balanced data splits...")
        
        # Stratified split to maintain class balance
        train_df, temp_df = train_test_split(
            df, test_size=0.3, 
            stratify=df['numeric_label'], 
            random_state=42
        )
        
        val_df, test_df = train_test_split(
            temp_df, test_size=0.33, 
            stratify=temp_df['numeric_label'], 
            random_state=42
        )
        
        print(f"📈 Balanced splits created:")
        print(f"   🚀 Training: {len(train_df)} images")
        print(f"   ✅ Validation: {len(val_df)} images") 
        print(f"   🧪 Testing: {len(test_df)} images")
        
        # Verify balance
        print(f"\n📊 Training set balance:")
        train_counts = train_df['class_label'].value_counts()
        for class_name in self.class_names:
            count = train_counts.get(class_name, 0)
            pct = (count / len(train_df)) * 100
            print(f"   {class_name:12s}: {count:4d} images ({pct:5.1f}%)")
        
        return train_df, val_df, test_df
    
    def calculate_class_weights(self, train_df):
        """
        Calculate balanced class weights
        """
        print(f"\n⚖️ Calculating balanced class weights...")
        
        labels_numeric = train_df['numeric_label'].values
        class_weights_array = compute_class_weight(
            'balanced',
            classes=np.unique(labels_numeric),
            y=labels_numeric
        )
        
        class_weights = {i: weight for i, weight in enumerate(class_weights_array)}
        
        print(f"⚖️ Class weights:")
        for i, class_name in enumerate(self.class_names):
            print(f"   {class_name:12s} ({i}): {class_weights[i]:.3f}")
        
        return class_weights
    
    def create_data_generators(self, train_df, val_df, test_df):
        """
        Creates optimized data generators
        """
        print(f"\n🔄 Creating optimized data generators...")
        
        # Training augmentation - moderate για stability
        train_datagen = ImageDataGenerator(
            rescale=1./255,
            rotation_range=15,
            horizontal_flip=True,
            zoom_range=0.1,
            brightness_range=[0.9, 1.1],
            fill_mode='nearest'
        )
        
        # Validation/test - no augmentation
        val_test_datagen = ImageDataGenerator(rescale=1./255)
        
        train_generator = train_datagen.flow_from_dataframe(
            train_df,
            x_col='image_path',
            y_col='class_label',
            target_size=self.target_size,
            batch_size=self.batch_size,
            class_mode='categorical',
            shuffle=True,
            seed=42
        )
        
        val_generator = val_test_datagen.flow_from_dataframe(
            val_df,
            x_col='image_path',
            y_col='class_label',
            target_size=self.target_size,
            batch_size=self.batch_size,
            class_mode='categorical',
            shuffle=False,
            seed=42
        )
        
        test_generator = val_test_datagen.flow_from_dataframe(
            test_df,
            x_col='image_path',
            y_col='class_label',
            target_size=self.target_size,
            batch_size=self.batch_size,
            class_mode='categorical',
            shuffle=False,
            seed=42
        )
        
        print(f"✅ Data generators created!")
        print(f"📊 Class indices: {train_generator.class_indices}")
        
        # Verify class order
        expected_order = {'glioma': 0, 'meningioma': 1, 'notumor': 2, 'pituitary': 3}
        actual_order = train_generator.class_indices
        
        if actual_order != expected_order:
            print(f"⚠️ WARNING: Class order mismatch!")
            print(f"   Expected: {expected_order}")
            print(f"   Actual: {actual_order}")
        else:
            print(f"✅ Class order verified correctly!")
        
        return train_generator, val_generator, test_generator
    
    def get_callbacks(self):
        """
        Get training callbacks
        """
        callbacks = [
            EarlyStopping(
                monitor='val_accuracy',
                patience=self.patience_early_stop,
                restore_best_weights=True,
                verbose=1,
                mode='max'
            ),
            ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=self.patience_lr_reduce,
                min_lr=1e-7,
                verbose=1
            ),
            ModelCheckpoint(
                'best_multiclass_v2_model.h5',
                monitor='val_accuracy',
                save_best_only=True,
                verbose=1,
                mode='max'
            )
        ]
        
        return callbacks
    
    def train_model(self, model, train_gen, val_gen, class_weights):
        """
        Train the V2 model
        """
        print(f"\n🚀 Training Multiclass V2 Model")
        print("=" * 40)
        print(f"📊 Epochs: {self.epochs}")
        print(f"🎯 Strategy: Single-phase robust training")
        print(f"⚖️ Using class weights για balance")
        
        history = model.fit(
            train_gen,
            epochs=self.epochs,
            validation_data=val_gen,
            class_weight=class_weights,
            callbacks=self.get_callbacks(),
            verbose=1
        )
        
        best_val_acc = max(history.history['val_accuracy'])
        best_train_acc = max(history.history['accuracy'])
        
        print(f"\n✅ Training completed!")
        print(f"🏆 Best validation accuracy: {best_val_acc:.4f}")
        print(f"📊 Best training accuracy: {best_train_acc:.4f}")
        
        return history
    
    def plot_training_history(self, history):
        """
        Plot training history
        """
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
        
        epochs = range(1, len(history.history['accuracy']) + 1)
        
        # Accuracy
        ax1.plot(epochs, history.history['accuracy'], 'b-', label='Training Accuracy')
        ax1.plot(epochs, history.history['val_accuracy'], 'r-', label='Validation Accuracy')
        ax1.set_title('Model Accuracy - Multiclass V2')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Accuracy')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Loss
        ax2.plot(epochs, history.history['loss'], 'b-', label='Training Loss')
        ax2.plot(epochs, history.history['val_loss'], 'r-', label='Validation Loss')
        ax2.set_title('Model Loss - Multiclass V2')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Loss')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # Precision
        ax3.plot(epochs, history.history['precision'], 'g-', label='Training Precision')
        ax3.plot(epochs, history.history['val_precision'], 'orange', label='Validation Precision')
        ax3.set_title('Model Precision - Multiclass V2')
        ax3.set_xlabel('Epoch')
        ax3.set_ylabel('Precision')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # Recall
        ax4.plot(epochs, history.history['recall'], 'purple', label='Training Recall')
        ax4.plot(epochs, history.history['val_recall'], 'brown', label='Validation Recall')
        ax4.set_title('Model Recall - Multiclass V2')
        ax4.set_xlabel('Epoch')
        ax4.set_ylabel('Recall')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        plt.suptitle('Multiclass V2 Training History', fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        plt.savefig('multiclass_v2_training_history.png', dpi=300, bbox_inches='tight')
        print(f"📊 Training history saved: multiclass_v2_training_history.png")
        plt.show()


def main():
    """
    Main training pipeline για Multiclass V2
    """
    print("🧠 Multiclass Brain Tumor CNN V2 - Clean Rebuild")
    print("=" * 60)
    
    # Paths
    data_dir = "/Users/yannisbalasis/Documents/thesis/dataset_binary_class"
    
    try:
        # Step 1: Initialize trainer
        trainer = MultiClassTrainerV2(
            data_dir=data_dir,
            target_size=(224, 224),
            batch_size=32
        )
        
        # Step 2: Create dataset
        df = trainer.create_multiclass_dataset()
        
        # Step 3: Create balanced splits
        train_df, val_df, test_df = trainer.create_balanced_splits(df)
        
        # Step 4: Calculate class weights
        class_weights = trainer.calculate_class_weights(train_df)
        
        # Step 5: Create data generators
        train_gen, val_gen, test_gen = trainer.create_data_generators(train_df, val_df, test_df)
        
        # Step 6: Create model
        model_builder = MultiClassBrainTumorCNNV2(
            input_shape=(224, 224, 3),
            num_classes=4
        )
        
        model = model_builder.create_robust_cnn_architecture()
        model = model_builder.compile_model(learning_rate=0.001)
        model_builder.print_model_summary()
        
        # Step 7: Train model
        history = trainer.train_model(model, train_gen, val_gen, class_weights)
        
        # Step 8: Plot results
        trainer.plot_training_history(history)
        
        print(f"\n🎉 Multiclass V2 training completed!")
        print(f"💾 Model saved: best_multiclass_v2_model.h5")
        print(f"🎯 Ready για evaluation!")
        
    except Exception as e:
        print(f"❌ Training failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()