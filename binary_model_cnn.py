

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models
from tensorflow.keras.optimizers import Adam
import numpy as np

class SimpleBinaryTumorCNN:
    """
    Απλό CNN για Binary Brain Tumor Classification
    """
    
    def __init__(self):
        """
        Αρχικοποίηση του model
        """
        self.model = None
        self.input_shape = (224, 224, 3)  # Μέγεθος εικόνας
        self.learning_rate = 0.001
        
    def build_model(self):
        """
        Δημιουργία της CNN αρχιτεκτονικής
        """
        print(" Building Binary Classification CNN...")
        
        self.model = models.Sequential([
            # Block 1: 32 filters
            layers.Conv2D(32, (3, 3), activation='relu', input_shape=self.input_shape),
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
            
            # Global Average Pooling (πιο έξυπνο από Flatten)
            layers.GlobalAveragePooling2D(),
            
            # Dense layers
            layers.Dense(512, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.5),
            
            layers.Dense(256, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.5),
            
            # Output layer - Binary Classification
            layers.Dense(1, activation='sigmoid')  # 0 = No Tumor, 1 = Tumor
        ])
        
        # Compile το model
        self.model.compile(
            optimizer=Adam(learning_rate=self.learning_rate),
            loss='binary_crossentropy',
            metrics=['accuracy', 'precision', 'recall']
        )
        
        print(" Model built successfully!")
        return self.model
    
    def show_model_info(self):
        """
        Εμφάνιση πληροφοριών του model
        """
        if self.model is None:
            print(" Model not built yet. Call build_model() first.")
            return
        
        print("\n Model Architecture:")
        print("=" * 50)
        self.model.summary()
        
        # Count parameters
        total_params = self.model.count_params()
        print(f"\n Total Parameters: {total_params:,}")
        print(f" Estimated Model Size: {total_params * 4 / 1024 / 1024:.1f} MB")
        
    def test_model(self):
        """
        Τεστ του model με dummy data
        """
        if self.model is None:
            print(" Model not built yet. Call build_model() first.")
            return
        
        print("\n Testing model with dummy data...")
        
        # Δημιουργία dummy input
        dummy_input = np.random.random((1, 224, 224, 3))
        
        try:
            # Prediction
            prediction = self.model.predict(dummy_input, verbose=0)
            probability = prediction[0][0]
            predicted_class = "Tumor" if probability > 0.5 else "No Tumor"
            
            print(f" Model test successful!")
            print(f" Input shape: {dummy_input.shape}")
            print(f" Output probability: {probability:.4f}")
            print(f" Predicted class: {predicted_class}")
            print(f" Confidence: {max(probability, 1-probability):.1%}")
            
        except Exception as e:
            print(f" Model test failed: {e}")


def main():
    """
    Κύρια συνάρτηση για testing του model
    """
    print(" Binary Brain Tumor Classification - Step 1")
    print("=" * 60)
    
    # Δημιουργία model instance
    cnn_model = SimpleBinaryTumorCNN()
    
    # Build το model
    model = cnn_model.build_model()
    
    # Εμφάνιση πληροφοριών
    cnn_model.show_model_info()
    
    # Τεστ του model
    cnn_model.test_model()
    


if __name__ == "__main__":
    main()