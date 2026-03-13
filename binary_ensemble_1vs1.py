import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models, callbacks, utils
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import cv2
from datetime import datetime
import json
import itertools

class BinaryTumorModel:
    """
    Binary classification model for tumor type pairs
    Optimized for excellent binary discrimination
    """
    
    def __init__(self, class1, class2, input_shape=(224, 224, 3)):
        self.class1 = class1
        self.class2 = class2
        self.input_shape = input_shape
        self.model = None
        self.history = None
        self.model_name = f"{class1}_vs_{class2}"
        
        # Medical-specific parameters
        self.medical_config = {
            'batch_size': 32,
            'epochs': 50,
            'patience': 15,
            'learning_rate': 0.001,
            'validation_split': 0.2,
            'test_split': 0.1
        }
    
    def create_model(self):
        """
        Create optimized binary classification model
        Based on proven binary architecture
        """
        print(f"🏗️ Creating Binary Model: {self.model_name}")
        
        model = models.Sequential([
            # Input layer
            layers.Input(shape=self.input_shape),
            
            # Block 1: Initial Feature Extraction
            layers.Conv2D(32, (3, 3), activation='relu', padding='same', name='conv1_1'),
            layers.BatchNormalization(name='bn1_1'),
            layers.Conv2D(32, (3, 3), activation='relu', padding='same', name='conv1_2'),
            layers.BatchNormalization(name='bn1_2'),
            layers.MaxPooling2D(pool_size=(2, 2), name='pool1'),
            layers.Dropout(0.25, name='dropout1'),
            
            # Block 2: Mid-Level Features
            layers.Conv2D(64, (3, 3), activation='relu', padding='same', name='conv2_1'),
            layers.BatchNormalization(name='bn2_1'),
            layers.Conv2D(64, (3, 3), activation='relu', padding='same', name='conv2_2'),
            layers.BatchNormalization(name='bn2_2'),
            layers.MaxPooling2D(pool_size=(2, 2), name='pool2'),
            layers.Dropout(0.25, name='dropout2'),
            
            # Block 3: High-Level Features
            layers.Conv2D(128, (3, 3), activation='relu', padding='same', name='conv3_1'),
            layers.BatchNormalization(name='bn3_1'),
            layers.Conv2D(128, (3, 3), activation='relu', padding='same', name='conv3_2'),
            layers.BatchNormalization(name='bn3_2'),
            layers.MaxPooling2D(pool_size=(2, 2), name='pool3'),
            layers.Dropout(0.3, name='dropout3'),
            
            # Block 4: Abstract Features
            layers.Conv2D(256, (3, 3), activation='relu', padding='same', name='conv4_1'),
            layers.BatchNormalization(name='bn4_1'),
            layers.Conv2D(256, (3, 3), activation='relu', padding='same', name='conv4_2'),
            layers.BatchNormalization(name='bn4_2'),
            layers.MaxPooling2D(pool_size=(2, 2), name='pool4'),
            layers.Dropout(0.3, name='dropout4'),
            
            # Binary Classification Head
            layers.GlobalAveragePooling2D(name='global_avg_pool'),
            layers.Dense(256, activation='relu', name='dense1'),
            layers.BatchNormalization(name='bn_dense1'),
            layers.Dropout(0.5, name='dropout_dense1'),
            
            layers.Dense(128, activation='relu', name='dense2'),
            layers.BatchNormalization(name='bn_dense2'),
            layers.Dropout(0.4, name='dropout_dense2'),
            
            # Binary output with sigmoid
            layers.Dense(1, activation='sigmoid', name=f'binary_{self.model_name}')
        ])
        
        self.model = model
        
        # Calculate parameters
        total_params = self.model.count_params()
        print(f"📊 {self.model_name} Parameters: {total_params:,}")
        
        return model
    
    def compile_model(self):
        """
        Compile model for binary classification
        """
        print(f"⚙️ Compiling {self.model_name}...")
        
        self.model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=self.medical_config['learning_rate']),
            loss='binary_crossentropy',
            metrics=[
                'accuracy',
                keras.metrics.Precision(name='precision'),
                keras.metrics.Recall(name='recall'),
                keras.metrics.AUC(name='auc')
            ]
        )
        
        print(f"✅ {self.model_name} compiled successfully!")
    
    def prepare_callbacks(self, output_dir="."):
        """
        Setup training callbacks for binary model
        """
        callbacks_list = [
            # Early stopping
            callbacks.EarlyStopping(
                monitor='val_accuracy',
                patience=self.medical_config['patience'],
                restore_best_weights=True,
                verbose=1
            ),
            
            # Reduce learning rate on plateau
            callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=8,
                min_lr=1e-7,
                verbose=1
            ),
            
            # Model checkpoint
            callbacks.ModelCheckpoint(
                filepath=os.path.join(output_dir, f'best_{self.model_name}_model.h5'),
                monitor='val_accuracy',
                save_best_only=True,
                save_weights_only=False,
                verbose=1
            ),
            
            # CSV logger
            callbacks.CSVLogger(
                os.path.join(output_dir, f'{self.model_name}_training_log.csv'),
                append=False
            )
        ]
        
        return callbacks_list
    
    def train_model(self, X_train, y_train, X_val, y_val, output_dir="."):
        """
        Train binary classification model
        """
        print(f"\n🚀 Training {self.model_name}...")
        print(f"📊 Training samples: {len(X_train):,}")
        print(f"📊 Validation samples: {len(X_val):,}")
        
        # Setup data augmentation
        train_datagen = ImageDataGenerator(
            rescale=1./255,
            rotation_range=15,
            width_shift_range=0.1,
            height_shift_range=0.1,
            horizontal_flip=True,
            zoom_range=0.1,
            brightness_range=[0.9, 1.1],
            fill_mode='nearest'
        )
        
        val_datagen = ImageDataGenerator(rescale=1./255)
        
        # Prepare callbacks
        callbacks_list = self.prepare_callbacks(output_dir)
        
        # Training generators
        train_generator = train_datagen.flow(
            X_train, y_train,
            batch_size=self.medical_config['batch_size'],
            shuffle=True
        )
        
        val_generator = val_datagen.flow(
            X_val, y_val,
            batch_size=self.medical_config['batch_size'],
            shuffle=False
        )
        
        # Start training
        start_time = datetime.now()
        print(f"⏰ {self.model_name} training started at: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
        
        self.history = self.model.fit(
            train_generator,
            epochs=self.medical_config['epochs'],
            validation_data=val_generator,
            callbacks=callbacks_list,
            verbose=1
        )
        
        end_time = datetime.now()
        training_time = end_time - start_time
        print(f"✅ {self.model_name} training completed in {training_time}")
        
        return self.history


class OneVsOneEnsemble:
    """
    1-vs-1 Binary Ensemble for 3-class tumor classification
    Creates 3 binary models and combines their predictions
    """
    
    def __init__(self, input_shape=(224, 224, 3)):
        self.input_shape = input_shape
        self.class_names = ['glioma', 'meningioma', 'pituitary']
        self.binary_models = {}
        self.model_pairs = [
            ('glioma', 'meningioma'),
            ('glioma', 'pituitary'),
            ('meningioma', 'pituitary')
        ]
        
        # Initialize binary models
        for class1, class2 in self.model_pairs:
            model_key = f"{class1}_vs_{class2}"
            self.binary_models[model_key] = BinaryTumorModel(class1, class2, input_shape)
    
    def create_all_models(self):
        """
        Create all binary classification models
        """
        print("🏗️ Creating 1-vs-1 Binary Ensemble...")
        print("📊 Models: Glioma vs Meningioma, Glioma vs Pituitary, Meningioma vs Pituitary")
        
        for model_key, binary_model in self.binary_models.items():
            binary_model.create_model()
            binary_model.compile_model()
        
        total_params = sum(model.model.count_params() for model in self.binary_models.values())
        print(f"\n📊 Total Ensemble Parameters: {total_params:,}")
        print(f"💾 Total Model Size: ~{total_params * 4 / 1024 / 1024:.1f} MB")
    
    def prepare_binary_data(self, X, y, class_counts, class1, class2):
        """
        Prepare data for specific binary classification
        """
        # Map class names to indices
        class_to_idx = {name: idx for idx, name in enumerate(self.class_names)}
        idx1, idx2 = class_to_idx[class1], class_to_idx[class2]
        
        # Filter data for these two classes
        mask = (y.argmax(axis=1) == idx1) | (y.argmax(axis=1) == idx2)
        X_binary = X[mask]
        y_labels = y.argmax(axis=1)[mask]
        
        # Convert to binary labels (0 for class1, 1 for class2)
        y_binary = (y_labels == idx2).astype(np.float32)
        
        print(f"📊 {class1} vs {class2}: {len(X_binary):,} samples")
        print(f"  {class1}: {np.sum(y_binary == 0):,} samples")
        print(f"  {class2}: {np.sum(y_binary == 1):,} samples")
        
        return X_binary, y_binary
    
    def load_and_preprocess_data(self, data_path):
        """
        Load and preprocess data for all binary models
        """
        print(f"\n📁 Loading Data for 1-vs-1 Ensemble from: {data_path}")
        
        images = []
        labels = []
        class_counts = {}
        
        # Define class mapping
        class_to_idx = {
            'glioma': 0,
            'meningioma': 1,
            'pituitary': 2
        }
        
        for class_name, class_idx in class_to_idx.items():
            class_path = os.path.join(data_path, class_name)
            if not os.path.exists(class_path):
                print(f"⚠️ Warning: {class_name} folder not found at {class_path}")
                continue
            
            image_files = [f for f in os.listdir(class_path) 
                          if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
            
            print(f"📊 Loading {class_name}: {len(image_files)} images")
            
            for img_file in image_files:
                img_path = os.path.join(class_path, img_file)
                try:
                    # Load and preprocess image
                    img = cv2.imread(img_path)
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    img = cv2.resize(img, (224, 224))
                    
                    images.append(img)
                    labels.append(class_idx)
                    
                except Exception as e:
                    print(f"⚠️ Error loading {img_path}: {e}")
                    continue
            
            class_counts[class_name] = len([l for l in labels if l == class_idx])
        
        # Convert to numpy arrays
        X = np.array(images, dtype=np.float32)
        y = np.array(labels)
        
        # Convert labels to categorical for consistency
        y_categorical = utils.to_categorical(y, num_classes=3)
        
        print(f"\n📊 Total Dataset: {len(X):,} images")
        for class_name, count in class_counts.items():
            percentage = (count / len(X)) * 100
            print(f"  {class_name.capitalize()}: {count:,} images ({percentage:.1f}%)")
        
        return X, y_categorical, class_counts
    
    def create_data_splits(self, X, y, test_size=0.1, val_size=0.2):
        """
        Create train/validation/test splits with stratification
        """
        print(f"\n✂️ Creating Data Splits for Ensemble:")
        
        # First split: separate test set
        X_temp, X_test, y_temp, y_test = train_test_split(
            X, y, test_size=test_size, stratify=y.argmax(axis=1), random_state=42
        )
        
        # Second split: training and validation
        val_size_adjusted = val_size / (1 - test_size)
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp, test_size=val_size_adjusted, 
            stratify=y_temp.argmax(axis=1), random_state=42
        )
        
        print(f"📊 Split Sizes:")
        print(f"  Training: {len(X_train):,} images")
        print(f"  Validation: {len(X_val):,} images")
        print(f"  Test: {len(X_test):,} images")
        
        return X_train, X_val, X_test, y_train, y_val, y_test
    
    def train_all_models(self, X_train, y_train, X_val, y_val, output_dir="."):
        """
        Train all binary models in the ensemble
        """
        print("\n" + "="*80)
        print("🚀 TRAINING 1-VS-1 BINARY ENSEMBLE")
        print("="*80)
        
        training_results = {}
        
        for i, (class1, class2) in enumerate(self.model_pairs, 1):
            print(f"\n🥊 BINARY MODEL {i}/3: {class1.upper()} vs {class2.upper()}")
            print("-" * 60)
            
            # Prepare binary data for this pair
            X_binary_train, y_binary_train = self.prepare_binary_data(
                X_train, y_train, None, class1, class2
            )
            X_binary_val, y_binary_val = self.prepare_binary_data(
                X_val, y_val, None, class1, class2
            )
            
            # Train binary model
            model_key = f"{class1}_vs_{class2}"
            binary_model = self.binary_models[model_key]
            
            start_time = datetime.now()
            history = binary_model.train_model(
                X_binary_train, y_binary_train,
                X_binary_val, y_binary_val,
                output_dir
            )
            training_time = datetime.now() - start_time
            
            training_results[model_key] = {
                'history': history,
                'training_time': training_time,
                'train_samples': len(X_binary_train),
                'val_samples': len(X_binary_val)
            }
        
        print("\n" + "="*80)
        print("✅ ALL BINARY MODELS TRAINED SUCCESSFULLY!")
        print("="*80)
        
        return training_results
    
    def predict_ensemble(self, X_test, y_test, output_dir="."):
        """
        Make ensemble predictions using probability voting
        """
        print("\n🔮 Making Ensemble Predictions...")
        
        # Normalize test data
        X_test_norm = X_test / 255.0
        n_samples = len(X_test)
        
        # Initialize probability matrix: [samples, classes]
        ensemble_probs = np.zeros((n_samples, 3))
        
        # Collect predictions from each binary model
        binary_predictions = {}
        
        for i, (class1, class2) in enumerate(self.model_pairs):
            model_key = f"{class1}_vs_{class2}"
            print(f"🎯 Getting predictions from {model_key}...")
            
            # Load best model
            model_path = os.path.join(output_dir, f'best_{model_key}_model.h5')
            if os.path.exists(model_path):
                binary_model = keras.models.load_model(model_path)
            else:
                binary_model = self.binary_models[model_key].model
            
            # Get binary predictions
            binary_probs = binary_model.predict(X_test_norm, verbose=0)
            
            # Convert to class probabilities
            class_to_idx = {'glioma': 0, 'meningioma': 1, 'pituitary': 2}
            idx1, idx2 = class_to_idx[class1], class_to_idx[class2]
            
            # Binary prob: 0 = class1, 1 = class2
            prob_class1 = 1 - binary_probs.flatten()
            prob_class2 = binary_probs.flatten()
            
            # Add to ensemble probability matrix
            ensemble_probs[:, idx1] += prob_class1
            ensemble_probs[:, idx2] += prob_class2
            
            binary_predictions[model_key] = {
                'probabilities': binary_probs,
                'class1': class1,
                'class2': class2
            }
        
        # Normalize ensemble probabilities
        ensemble_probs = ensemble_probs / len(self.model_pairs)
        
        # Get final predictions
        ensemble_predictions = np.argmax(ensemble_probs, axis=1)
        y_true = np.argmax(y_test, axis=1)
        
        # Calculate accuracy
        accuracy = np.mean(ensemble_predictions == y_true)
        
        print(f"\n🎯 Ensemble Results:")
        print(f"  Test Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
        
        # Detailed classification report
        class_names_display = ['Glioma', 'Meningioma', 'Pituitary']
        class_report = classification_report(
            y_true, ensemble_predictions,
            target_names=class_names_display,
            digits=4
        )
        print(f"\n📋 Detailed Classification Report:")
        print(class_report)
        
        # Confusion matrix
        cm = confusion_matrix(y_true, ensemble_predictions)
        
        return {
            'ensemble_accuracy': accuracy,
            'ensemble_predictions': ensemble_predictions,
            'ensemble_probabilities': ensemble_probs,
            'binary_predictions': binary_predictions,
            'true_labels': y_true,
            'confusion_matrix': cm,
            'classification_report': class_report
        }
    
    def plot_ensemble_results(self, results, output_dir="."):
        """
        Create comprehensive visualizations for ensemble results
        """
        print("\n📊 Creating Ensemble Visualizations...")
        
        cm = results['confusion_matrix']
        class_names_display = ['Glioma', 'Meningioma', 'Pituitary']
        
        # Enhanced confusion matrix
        plt.figure(figsize=(10, 8))
        
        # Calculate percentages
        cm_percentage = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100
        
        # Create heatmap
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=class_names_display, yticklabels=class_names_display,
                   cbar_kws={'label': 'Number of Samples'})
        
        plt.title('Confusion Matrix - 1-vs-1 Binary Ensemble', 
                 fontsize=16, fontweight='bold', pad=20)
        plt.xlabel('Predicted Label', fontsize=12, fontweight='bold')
        plt.ylabel('True Label', fontsize=12, fontweight='bold')
        
        # Add accuracy text
        accuracy = results['ensemble_accuracy']
        plt.figtext(0.5, 0.02, f'Ensemble Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)', 
                   ha='center', fontsize=12, fontweight='bold')
        
        plt.tight_layout()
        save_path = os.path.join(output_dir, 'ensemble_confusion_matrix.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"💾 Ensemble confusion matrix saved: {save_path}")
        
        plt.close()


def main():
    """
    Main function to demonstrate 1-vs-1 ensemble training
    """
    print("🥊 1-VS-1 BINARY ENSEMBLE BRAIN TUMOR CLASSIFICATION")
    print("=" * 80)
    
    # Initialize ensemble
    ensemble = OneVsOneEnsemble()
    
    # Create all models
    ensemble.create_all_models()
    
    # Note: You would need to provide the actual data path
    # data_path = "/path/to/your/tumor/dataset"
    # X, y, class_counts = ensemble.load_and_preprocess_data(data_path)
    # X_train, X_val, X_test, y_train, y_val, y_test = ensemble.create_data_splits(X, y)
    # 
    # # Train all models
    # training_results = ensemble.train_all_models(X_train, y_train, X_val, y_val)
    # 
    # # Make ensemble predictions
    # ensemble_results = ensemble.predict_ensemble(X_test, y_test)
    # 
    # # Create visualizations
    # ensemble.plot_ensemble_results(ensemble_results)
    
    print("\n✅ 1-vs-1 Binary Ensemble Implementation Complete!")
    print("📁 Ready for training with 3 binary models")

if __name__ == "__main__":
    main()