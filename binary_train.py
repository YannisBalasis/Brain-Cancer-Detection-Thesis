

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras.preprocessing.image import ImageDataGenerator

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve
from sklearn.utils.class_weight import compute_class_weight
import os

class CompleteBinaryTraining:
    """
    Complete training pipeline για binary brain tumor classification
    """
    
    def __init__(self, data_dir, target_size=(224, 224), batch_size=32):
        """
        Αρχικοποίηση του training pipeline
        """
        self.data_dir = data_dir
        self.target_size = target_size
        self.batch_size = batch_size
        
        # Model parameters
        self.learning_rate = 0.001
        self.epochs = 50
        
        # Data splits (70/20/10)
        self.train_split = 0.7
        self.val_split = 0.2
        self.test_split = 0.1
        
        # Binary mapping
        self.tumor_classes = ['glioma', 'meningioma', 'pituitary']
        self.no_tumor_classes = ['notumor']
        
        # Placeholders
        self.model = None
        self.history = None
        self.class_weights = None
        
        print(" Complete Binary Training Pipeline Initialized")
        print(f" Target size: {target_size}")
        print(f" Batch size: {batch_size}")
        print(f" Data splits: {self.train_split*100:.0f}% / {self.val_split*100:.0f}% / {self.test_split*100:.0f}%")
    
    def create_binary_dataframe(self):
        """
        Δημιουργεί DataFrame με τα δεδομένα
        """
        print("\n Creating binary classification DataFrame...")
        
        image_paths = []
        binary_labels = []
        original_classes = []
        
        # Επεξεργασία Training και Testing
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
        
        # Δημιουργία DataFrame
        df = pd.DataFrame({
            'image_path': image_paths,
            'binary_label': binary_labels,
            'original_class': original_classes
        })
        
        print(f" DataFrame created with {len(df)} images")
        print(f" Binary distribution:")
        print(df['binary_label'].value_counts())
        
        return df
    
    def create_data_splits(self, df):
        """
        Δημιουργεί τα data splits 70/20/10
        """
        print(f"\n Creating data splits (70/20/10)...")
        
        # Πρώτο split: 70% train, 30% temp
        train_df, temp_df = train_test_split(
            df, 
            test_size=0.3, 
            stratify=df['binary_label'], 
            random_state=42
        )
        
        # Δεύτερο split: 20% validation, 10% test (από το 30% temp)
        val_df, test_df = train_test_split(
            temp_df, 
            test_size=0.33,  # 0.33 * 0.3 = 0.1 (10% του συνόλου)
            stratify=temp_df['binary_label'], 
            random_state=42
        )
        
        print(f" Data splits created:")
        print(f"    Training: {len(train_df)} images ({len(train_df)/len(df)*100:.1f}%)")
        print(f"    Validation: {len(val_df)} images ({len(val_df)/len(df)*100:.1f}%)")
        print(f"    Testing: {len(test_df)} images ({len(test_df)/len(df)*100:.1f}%)")
        
        # Check balance σε κάθε split
        print(f"\n Class distribution per split:")
        for name, split_df in [("Train", train_df), ("Validation", val_df), ("Test", test_df)]:
            counts = split_df['binary_label'].value_counts()
            total = len(split_df)
            print(f"   {name}: {counts.get('tumor', 0)} tumor ({counts.get('tumor', 0)/total*100:.1f}%), {counts.get('no_tumor', 0)} no_tumor ({counts.get('no_tumor', 0)/total*100:.1f}%)")
        
        return train_df, val_df, test_df
    
    def calculate_class_weights(self, train_df):
        """
        Υπολογίζει class weights για το imbalanced dataset
        """
        print(f"\n Calculating class weights for imbalanced data...")
        
        # Count classes
        class_counts = train_df['binary_label'].value_counts()
        print(f" Training class counts: {class_counts.to_dict()}")
        
        # Convert to numeric labels
        y_train = (train_df['binary_label'] == 'tumor').astype(int)
        
        # Calculate class weights
        class_weights_array = compute_class_weight(
            'balanced',
            classes=np.unique(y_train),
            y=y_train
        )
        
        self.class_weights = {0: class_weights_array[0], 1: class_weights_array[1]}
        
        print(f" Class weights calculated:")
        print(f"   No Tumor (0): {self.class_weights[0]:.3f}")
        print(f"   Tumor (1): {self.class_weights[1]:.3f}")
        
        return self.class_weights
    
    def create_data_generators(self, train_df, val_df, test_df):
        """
        Δημιουργεί data generators με augmentation
        """
        print(f"\n Creating data generators...")
        
        # Training generator με augmentation
        train_datagen = ImageDataGenerator(
            rescale=1./255,
            
            # MRI-safe augmentations
            rotation_range=15,
            width_shift_range=0.1,
            height_shift_range=0.1,
            shear_range=0.1,
            zoom_range=0.1,
            horizontal_flip=True,
            vertical_flip=False,  # Όχι για brain MRI
            fill_mode='nearest',
            
            # Intensity augmentations
            brightness_range=[0.8, 1.2],
            channel_shift_range=0.1
        )
        
        # Validation/Test generators (μόνο normalization)
        val_test_datagen = ImageDataGenerator(rescale=1./255)
        
        # Create generators
        train_generator = train_datagen.flow_from_dataframe(
            train_df,
            x_col='image_path',
            y_col='binary_label',
            target_size=self.target_size,
            batch_size=self.batch_size,
            class_mode='binary',
            shuffle=True,
            seed=42
        )
        
        val_generator = val_test_datagen.flow_from_dataframe(
            val_df,
            x_col='image_path',
            y_col='binary_label',
            target_size=self.target_size,
            batch_size=self.batch_size,
            class_mode='binary',
            shuffle=False,
            seed=42
        )
        
        test_generator = val_test_datagen.flow_from_dataframe(
            test_df,
            x_col='image_path',
            y_col='binary_label',
            target_size=self.target_size,
            batch_size=self.batch_size,
            class_mode='binary',
            shuffle=False,
            seed=42
        )
        
        print(f" Data generators created successfully!")
        print(f" Class indices: {train_generator.class_indices}")
        
        return train_generator, val_generator, test_generator
    
    def build_model(self):
        """
        Δημιουργεί το CNN model
        """
        print(f"\n Building CNN model...")
        
        self.model = models.Sequential([
            # Block 1: 32 filters
            layers.Conv2D(32, (3, 3), activation='relu', input_shape=(*self.target_size, 3)),
            layers.BatchNormalization(),
            layers.Conv2D(32, (3, 3), activation='relu'),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.25),
            
            # Block 2: 64 filters
            layers.Conv2D(64, (3, 3), activation='relu'),
            layers.BatchNormalization(),
            layers.Conv2D(64, (3, 3), activation='relu'),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.35),
            
            # Block 3: 128 filters
            layers.Conv2D(128, (3, 3), activation='relu'),
            layers.BatchNormalization(),
            layers.Conv2D(128, (3, 3), activation='relu'),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.4),
            
            # Block 4: 256 filters
            layers.Conv2D(256, (3, 3), activation='relu'),
            layers.BatchNormalization(),
            layers.Conv2D(256, (3, 3), activation='relu'),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.5),
            
            # Global Average Pooling
            layers.GlobalAveragePooling2D(),
            
            # Dense layers
            layers.Dense(512, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.5),
            
            layers.Dense(256, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.5),
            
            # Binary output
            layers.Dense(1, activation='sigmoid')
        ])
        
        # Compile
        self.model.compile(
            optimizer=Adam(learning_rate=self.learning_rate),
            loss='binary_crossentropy',
            metrics=['accuracy', 'precision', 'recall']
        )
        
        print(f" Model built and compiled!")
        print(f" Total parameters: {self.model.count_params():,}")
        
        return self.model
    
    def get_callbacks(self):
        """
        Δημιουργεί training callbacks
        """
        callbacks = [
            EarlyStopping(
                monitor='val_loss',
                patience=10,
                restore_best_weights=True,
                verbose=1
            ),
            ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=5,
                min_lr=1e-7,
                verbose=1
            ),
            ModelCheckpoint(
                'best_binary_model.h5',
                monitor='val_accuracy',
                save_best_only=True,
                verbose=1
            )
        ]
        
        return callbacks
    
    def train_model(self, train_gen, val_gen):
        """
        Τρέχει το training
        """
        print(f"\n Starting model training...")
        print(f" Training parameters:")
        print(f"   Epochs: {self.epochs}")
        print(f"   Learning rate: {self.learning_rate}")
        print(f"   Class weights: {self.class_weights}")
        
        callbacks = self.get_callbacks()
        
        self.history = self.model.fit(
            train_gen,
            validation_data=val_gen,
            epochs=self.epochs,
            callbacks=callbacks,
            class_weight=self.class_weights,  # Για το imbalanced dataset
            verbose=1
        )
        
        print(f" Training completed!")
        return self.history
    
    def plot_training_history(self):
        """
        Plot training history
        """
        if self.history is None:
            print(" No training history found")
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Training History', fontsize=16)
        
        # Accuracy
        axes[0, 0].plot(self.history.history['accuracy'], label='Training')
        axes[0, 0].plot(self.history.history['val_accuracy'], label='Validation')
        axes[0, 0].set_title('Model Accuracy')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Accuracy')
        axes[0, 0].legend()
        axes[0, 0].grid(True)
        
        # Loss
        axes[0, 1].plot(self.history.history['loss'], label='Training')
        axes[0, 1].plot(self.history.history['val_loss'], label='Validation')
        axes[0, 1].set_title('Model Loss')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Loss')
        axes[0, 1].legend()
        axes[0, 1].grid(True)
        
        # Precision
        axes[1, 0].plot(self.history.history['precision'], label='Training')
        axes[1, 0].plot(self.history.history['val_precision'], label='Validation')
        axes[1, 0].set_title('Model Precision')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('Precision')
        axes[1, 0].legend()
        axes[1, 0].grid(True)
        
        # Recall
        axes[1, 1].plot(self.history.history['recall'], label='Training')
        axes[1, 1].plot(self.history.history['val_recall'], label='Validation')
        axes[1, 1].set_title('Model Recall')
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('Recall')
        axes[1, 1].legend()
        axes[1, 1].grid(True)
        
        plt.tight_layout()
        plt.savefig('training_history.png', dpi=300, bbox_inches='tight')
        print(" Training history saved as 'training_history.png'")
        plt.show()


def main():
    """
    Κύρια συνάρτηση training pipeline
    """
    print(" Complete Binary Training Pipeline - Step 3")
    print("=" * 60)
    
    # Dataset path - ΕΔΩ αλλάζεις το path σου
    data_dir = "/Users/yannisbalasis/Documents/thesis/dataset_binary_class"
    
    # Initialize training pipeline
    trainer = CompleteBinaryTraining(
        data_dir=data_dir,
        target_size=(224, 224),
        batch_size=32
    )
    
    # Step 1: Create DataFrame
    df = trainer.create_binary_dataframe()
    
    # Step 2: Create data splits (70/20/10)
    train_df, val_df, test_df = trainer.create_data_splits(df)
    
    # Step 3: Calculate class weights
    class_weights = trainer.calculate_class_weights(train_df)
    
    # Step 4: Create data generators
    train_gen, val_gen, test_gen = trainer.create_data_generators(train_df, val_df, test_df)
    
    # Step 5: Build model
    model = trainer.build_model()
    
    # Step 6: Train model
    history = trainer.train_model(train_gen, val_gen)
    
    # Step 7: Plot results
    trainer.plot_training_history()
    
    print(f"\n Step 3 Complete!")
    print(f" Generated files:")
    print(f"    best_binary_model.h5")
    print(f"    training_history.png")
    print(f" Next: Model evaluation on test set (Step 4)")


if __name__ == "__main__":
    main()