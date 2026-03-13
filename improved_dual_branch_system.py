#!/usr/bin/env python3
"""
🚀 Improved Dual Branch System για Brain Tumor Classification
Single-phase training με medical-optimized design

Key Improvements:
- Single-phase training (no complex multi-phase)
- Medical-optimized branches (no ImageNet transfer)
- Simple concatenation fusion (no attention overhead)
- Balanced parameter count (~2-3M)
- Proper medical regularization
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models, callbacks
import numpy as np
import matplotlib.pyplot as plt
import os
import cv2
from datetime import datetime
import json

class ImprovedDualBranchModel:
    """
    Improved Dual Branch System με single-phase training
    Medical-optimized design για better performance
    """
    
    def __init__(self, input_shape=(224, 224, 3), num_classes=4):
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.model = None
        self.history = None
        self.class_names = ['Glioma', 'Meningioma', 'No Tumor', 'Pituitary']
        
        # Medical-optimized config
        self.config = {
            'batch_size': 32,
            'epochs': 40,  # Reduced for single-phase
            'patience': 12,
            'learning_rate': 0.001,
            'l2_regularization': 1e-4,
            'dropout_rate': 0.3
        }
    
    def create_medical_cnn_branch(self, name_prefix="branch1"):
        """
        Create medical-optimized CNN branch
        Designed specifically για brain MRI patterns
        """
        inputs = layers.Input(shape=self.input_shape, name=f'{name_prefix}_input')
        
        # Block 1: Fine-grained features για medical textures
        x = layers.Conv2D(32, (3, 3), activation='relu', padding='same', 
                         kernel_regularizer=keras.regularizers.l2(self.config['l2_regularization']),
                         name=f'{name_prefix}_conv1_1')(inputs)
        x = layers.BatchNormalization(name=f'{name_prefix}_bn1_1')(x)
        x = layers.Conv2D(32, (3, 3), activation='relu', padding='same',
                         kernel_regularizer=keras.regularizers.l2(self.config['l2_regularization']),
                         name=f'{name_prefix}_conv1_2')(x)
        x = layers.BatchNormalization(name=f'{name_prefix}_bn1_2')(x)
        x = layers.MaxPooling2D(pool_size=(2, 2), name=f'{name_prefix}_pool1')(x)
        x = layers.Dropout(self.config['dropout_rate'], name=f'{name_prefix}_dropout1')(x)
        
        # Block 2: Mid-level medical features
        x = layers.Conv2D(64, (3, 3), activation='relu', padding='same',
                         kernel_regularizer=keras.regularizers.l2(self.config['l2_regularization']),
                         name=f'{name_prefix}_conv2_1')(x)
        x = layers.BatchNormalization(name=f'{name_prefix}_bn2_1')(x)
        x = layers.Conv2D(64, (3, 3), activation='relu', padding='same',
                         kernel_regularizer=keras.regularizers.l2(self.config['l2_regularization']),
                         name=f'{name_prefix}_conv2_2')(x)
        x = layers.BatchNormalization(name=f'{name_prefix}_bn2_2')(x)
        x = layers.MaxPooling2D(pool_size=(2, 2), name=f'{name_prefix}_pool2')(x)
        x = layers.Dropout(self.config['dropout_rate'], name=f'{name_prefix}_dropout2')(x)
        
        # Block 3: High-level patterns
        x = layers.Conv2D(128, (3, 3), activation='relu', padding='same',
                         kernel_regularizer=keras.regularizers.l2(self.config['l2_regularization']),
                         name=f'{name_prefix}_conv3_1')(x)
        x = layers.BatchNormalization(name=f'{name_prefix}_bn3_1')(x)
        x = layers.Conv2D(128, (3, 3), activation='relu', padding='same',
                         kernel_regularizer=keras.regularizers.l2(self.config['l2_regularization']),
                         name=f'{name_prefix}_conv3_2')(x)
        x = layers.BatchNormalization(name=f'{name_prefix}_bn3_2')(x)
        x = layers.MaxPooling2D(pool_size=(2, 2), name=f'{name_prefix}_pool3')(x)
        x = layers.Dropout(self.config['dropout_rate'] + 0.1, name=f'{name_prefix}_dropout3')(x)
        
        # Global feature extraction
        x = layers.GlobalAveragePooling2D(name=f'{name_prefix}_gap')(x)
        x = layers.Dense(256, activation='relu',
                        kernel_regularizer=keras.regularizers.l2(self.config['l2_regularization']),
                        name=f'{name_prefix}_dense1')(x)
        x = layers.BatchNormalization(name=f'{name_prefix}_bn_dense1')(x)
        x = layers.Dropout(self.config['dropout_rate'] + 0.2, name=f'{name_prefix}_dropout_dense')(x)
        
        # Final branch output
        outputs = layers.Dense(128, activation='relu',
                              kernel_regularizer=keras.regularizers.l2(self.config['l2_regularization']),
                              name=f'{name_prefix}_features')(x)
        
        return keras.Model(inputs, outputs, name=f'{name_prefix}_branch')
    
    def create_complementary_branch(self, name_prefix="branch2"):
        """
        Create complementary branch με different receptive field patterns
        Focuses on different scale features
        """
        inputs = layers.Input(shape=self.input_shape, name=f'{name_prefix}_input')
        
        # Block 1: Larger receptive fields για global patterns
        x = layers.Conv2D(32, (5, 5), activation='relu', padding='same',
                         kernel_regularizer=keras.regularizers.l2(self.config['l2_regularization']),
                         name=f'{name_prefix}_conv1_1')(inputs)
        x = layers.BatchNormalization(name=f'{name_prefix}_bn1_1')(x)
        x = layers.Conv2D(32, (3, 3), activation='relu', padding='same',
                         kernel_regularizer=keras.regularizers.l2(self.config['l2_regularization']),
                         name=f'{name_prefix}_conv1_2')(x)
        x = layers.BatchNormalization(name=f'{name_prefix}_bn1_2')(x)
        x = layers.MaxPooling2D(pool_size=(2, 2), name=f'{name_prefix}_pool1')(x)
        x = layers.Dropout(self.config['dropout_rate'], name=f'{name_prefix}_dropout1')(x)
        
        # Block 2: Mixed scale convolutions
        # Path 1: 3x3 convolutions
        path1 = layers.Conv2D(32, (3, 3), activation='relu', padding='same',
                             kernel_regularizer=keras.regularizers.l2(self.config['l2_regularization']),
                             name=f'{name_prefix}_path1_conv1')(x)
        
        # Path 2: 5x5 convolutions (via two 3x3)
        path2 = layers.Conv2D(32, (3, 3), activation='relu', padding='same',
                             kernel_regularizer=keras.regularizers.l2(self.config['l2_regularization']),
                             name=f'{name_prefix}_path2_conv1')(x)
        path2 = layers.Conv2D(32, (3, 3), activation='relu', padding='same',
                             kernel_regularizer=keras.regularizers.l2(self.config['l2_regularization']),
                             name=f'{name_prefix}_path2_conv2')(path2)
        
        # Combine paths
        x = layers.Concatenate(name=f'{name_prefix}_concat_paths')([path1, path2])
        x = layers.BatchNormalization(name=f'{name_prefix}_bn2_1')(x)
        x = layers.MaxPooling2D(pool_size=(2, 2), name=f'{name_prefix}_pool2')(x)
        x = layers.Dropout(self.config['dropout_rate'], name=f'{name_prefix}_dropout2')(x)
        
        # Block 3: Feature consolidation
        x = layers.Conv2D(128, (3, 3), activation='relu', padding='same',
                         kernel_regularizer=keras.regularizers.l2(self.config['l2_regularization']),
                         name=f'{name_prefix}_conv3_1')(x)
        x = layers.BatchNormalization(name=f'{name_prefix}_bn3_1')(x)
        x = layers.MaxPooling2D(pool_size=(2, 2), name=f'{name_prefix}_pool3')(x)
        x = layers.Dropout(self.config['dropout_rate'] + 0.1, name=f'{name_prefix}_dropout3')(x)
        
        # Global feature extraction
        x = layers.GlobalAveragePooling2D(name=f'{name_prefix}_gap')(x)
        x = layers.Dense(256, activation='relu',
                        kernel_regularizer=keras.regularizers.l2(self.config['l2_regularization']),
                        name=f'{name_prefix}_dense1')(x)
        x = layers.BatchNormalization(name=f'{name_prefix}_bn_dense1')(x)
        x = layers.Dropout(self.config['dropout_rate'] + 0.2, name=f'{name_prefix}_dropout_dense')(x)
        
        # Final branch output  
        outputs = layers.Dense(128, activation='relu',
                              kernel_regularizer=keras.regularizers.l2(self.config['l2_regularization']),
                              name=f'{name_prefix}_features')(x)
        
        return keras.Model(inputs, outputs, name=f'{name_prefix}_branch')
    
    def create_improved_dual_system(self):
        """
        Create improved dual branch system με simple fusion
        Single-phase training για better optimization
        """
        print("🏗️ Creating Improved Dual Branch System...")
        print("📊 Classes: Glioma, Meningioma, Pituitary, No Tumor")
        print("🎯 Improvements: Single-phase, Medical-optimized, Simple fusion")
        
        # Create input layer
        main_input = layers.Input(shape=self.input_shape, name='main_input')
        
        # Create medical-optimized branches
        print("🔧 Creating Branch 1: Medical-optimized CNN...")
        branch1 = self.create_medical_cnn_branch("medical_branch")
        branch1_features = branch1(main_input)
        
        print("🔧 Creating Branch 2: Complementary patterns...")
        branch2 = self.create_complementary_branch("complement_branch")
        branch2_features = branch2(main_input)
        
        # Simple concatenation fusion (no complex attention)
        print("🔗 Creating Simple Fusion Layer...")
        fused_features = layers.Concatenate(name='simple_fusion')([branch1_features, branch2_features])
        
        # Fusion processing
        x = layers.Dense(256, activation='relu',
                        kernel_regularizer=keras.regularizers.l2(self.config['l2_regularization']),
                        name='fusion_dense1')(fused_features)
        x = layers.BatchNormalization(name='fusion_bn1')(x)
        x = layers.Dropout(0.5, name='fusion_dropout1')(x)
        
        x = layers.Dense(128, activation='relu',
                        kernel_regularizer=keras.regularizers.l2(self.config['l2_regularization']),
                        name='fusion_dense2')(x)
        x = layers.BatchNormalization(name='fusion_bn2')(x)
        x = layers.Dropout(0.4, name='fusion_dropout2')(x)
        
        # Final classification
        outputs = layers.Dense(self.num_classes, activation='softmax',
                              name='classification_output')(x)
        
        # Create complete model
        self.model = keras.Model(inputs=main_input, outputs=outputs, 
                                name='ImprovedDualBranchSystem')
        
        # Print model summary
        print("\n📋 Improved Dual System Architecture:")
        self.model.summary()
        
        # Calculate parameters
        total_params = self.model.count_params()
        print(f"\n📊 Total Parameters: {total_params:,}")
        print(f"💾 Model Size: ~{total_params * 4 / 1024 / 1024:.1f} MB")
        
        # Compare με previous dual systems
        print(f"\n📈 Parameter Comparison:")
        print(f"  Previous EfficientNet Dual: 12,819,923 parameters")
        print(f"  Previous ResNet Dual: ~8,000,000+ parameters")
        print(f"  Improved Dual System: {total_params:,} parameters")
        print(f"  Efficiency improvement: {(12819923 - total_params) / 12819923 * 100:.1f}% fewer parameters")
        
        return self.model
    
    def compile_model(self):
        """Compile model με medical-optimized settings"""
        print("\n⚙️ Compiling Improved Dual System...")
        
        # Use lower learning rate για stable single-phase training
        optimizer = keras.optimizers.Adam(
            learning_rate=self.config['learning_rate'],
            beta_1=0.9,
            beta_2=0.999,
            epsilon=1e-7
        )
        
        self.model.compile(
            optimizer=optimizer,
            loss='categorical_crossentropy',
            metrics=[
                'accuracy',
                keras.metrics.TopKCategoricalAccuracy(k=2, name='top_2_accuracy'),
                keras.metrics.Precision(name='precision'),
                keras.metrics.Recall(name='recall'),
                keras.metrics.F1Score(name='f1_score')
            ]
        )
        
        print("✅ Model compiled successfully!")
        print("🎯 Optimizer: Adam με medical-optimized settings")
        print("📊 Metrics: Accuracy, Top-2, Precision, Recall, F1-Score")
    
    def prepare_callbacks(self, experiment_dir):
        """Setup callbacks για improved training"""
        print("\n📞 Setting up Medical Training Callbacks...")
        
        callbacks_list = [
            # Early stopping με patience
            callbacks.EarlyStopping(
                monitor='val_accuracy',
                patience=self.config['patience'],
                restore_best_weights=True,
                verbose=1,
                min_delta=0.001
            ),
            
            # Learning rate reduction
            callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=6,
                min_lr=1e-7,
                verbose=1,
                min_delta=0.001
            ),
            
            # Model checkpoint
            callbacks.ModelCheckpoint(
                filepath=os.path.join(experiment_dir, 'best_improved_dual_model.h5'),
                monitor='val_accuracy',
                save_best_only=True,
                save_weights_only=False,
                verbose=1
            ),
            
            # CSV logger
            callbacks.CSVLogger(
                os.path.join(experiment_dir, 'improved_dual_training_log.csv'),
                append=False
            ),
            
            # Learning rate scheduler για fine-tuning
            callbacks.LearningRateScheduler(
                lambda epoch: self.config['learning_rate'] * (0.95 ** epoch),
                verbose=0
            )
        ]
        
        print("✅ Callbacks configured:")
        print(f"  - Early stopping: patience={self.config['patience']}")
        print("  - LR reduction: factor=0.5, patience=6")
        print("  - Model checkpointing enabled")
        print("  - LR scheduling enabled")
        
        return callbacks_list
    
    def train_model(self, X_train, y_train, X_val, y_val, experiment_dir):
        """
        Single-phase training για improved performance
        """
        print("\n🚀 Starting Improved Dual System Training...")
        print("🎯 Single-phase training για optimal convergence")
        print(f"📊 Training samples: {len(X_train):,}")
        print(f"📊 Validation samples: {len(X_val):,}")
        
        # Data augmentation για medical imaging
        from tensorflow.keras.preprocessing.image import ImageDataGenerator
        
        train_datagen = ImageDataGenerator(
            rescale=1./255,
            rotation_range=12,           # Conservative για medical
            width_shift_range=0.08,      
            height_shift_range=0.08,
            horizontal_flip=True,        # Brain symmetry
            zoom_range=0.08,            
            brightness_range=[0.92, 1.08],
            fill_mode='nearest'
        )
        
        val_datagen = ImageDataGenerator(rescale=1./255)
        
        # Calculate class weights
        from sklearn.utils.class_weight import compute_class_weight
        y_train_labels = y_train.argmax(axis=1)
        class_weights = compute_class_weight(
            'balanced',
            classes=np.unique(y_train_labels),
            y=y_train_labels
        )
        class_weight_dict = dict(enumerate(class_weights))
        
        print(f"\n⚖️ Balanced Class Weights:")
        for i, weight in class_weight_dict.items():
            print(f"  {self.class_names[i]}: {weight:.3f}")
        
        # Setup generators
        train_generator = train_datagen.flow(
            X_train, y_train,
            batch_size=self.config['batch_size'],
            shuffle=True
        )
        
        val_generator = val_datagen.flow(
            X_val, y_val,
            batch_size=self.config['batch_size'],
            shuffle=False
        )
        
        # Prepare callbacks
        callbacks_list = self.prepare_callbacks(experiment_dir)
        
        # Start training
        start_time = datetime.now()
        print(f"⏰ Training started at: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
        print("🎯 Expected improved performance over complex dual systems...")
        
        self.history = self.model.fit(
            train_generator,
            epochs=self.config['epochs'],
            validation_data=val_generator,
            callbacks=callbacks_list,
            class_weight=class_weight_dict,
            verbose=1
        )
        
        end_time = datetime.now()
        training_time = end_time - start_time
        print(f"\n✅ Training completed!")
        print(f"⏱️ Total training time: {training_time}")
        
        return self.history
    
    def save_model_and_config(self, experiment_dir):
        """Save model και configuration"""
        print("\n💾 Saving Improved Dual System...")
        
        # Save final model
        final_model_path = os.path.join(experiment_dir, 'improved_dual_final_model.h5')
        self.model.save(final_model_path)
        
        # Save configuration
        config_data = {
            'model_type': 'Improved Dual Branch System',
            'architecture': 'Medical-optimized dual CNN με simple fusion',
            'improvements': [
                'Single-phase training',
                'Medical-optimized branches',
                'Simple concatenation fusion',
                'Balanced parameter count',
                'Proper regularization'
            ],
            'training_config': self.config,
            'total_parameters': self.model.count_params(),
            'training_time': datetime.now().isoformat()
        }
        
        config_path = os.path.join(experiment_dir, 'improved_dual_config.json')
        with open(config_path, 'w') as f:
            json.dump(config_data, f, indent=2)
        
        print("✅ Model and configuration saved!")
        print(f"📁 Final model: {final_model_path}")
        print(f"📁 Config: {config_path}")
        
        return final_model_path


def main():
    """Main function για testing improved dual system"""
    print("🚀 Improved Dual Branch System για Brain Tumor Classification")
    print("=" * 70)
    
    # Initialize model
    model = ImprovedDualBranchModel()
    
    # Create architecture
    model.create_improved_dual_system()
    
    # Compile model
    model.compile_model()
    
    print("\n✅ Improved Dual System Implementation Complete!")
    print("📋 Ready για training με medical data")
    print("🎯 Expected performance: 95%+ (better than complex dual systems)")

if __name__ == "__main__":
    main()