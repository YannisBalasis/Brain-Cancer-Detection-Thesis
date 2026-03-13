

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models
from tensorflow.keras.optimizers import Adam
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyBboxPatch
import os

class VisualBinaryTumorCNN:
    """
    Binary CNN με visualization capabilities
    """
    
    def __init__(self):
        """
        Αρχικοποίηση του model
        """
        self.model = None
        self.input_shape = (224, 224, 3)
        self.learning_rate = 0.001
        
    def build_model(self):
        """
        Δημιουργία της CNN αρχιτεκτονικής
        """
        print(" Building Binary Classification CNN...")
        
        self.model = models.Sequential([
            # Block 1: 32 filters
            layers.Conv2D(32, (3, 3), activation='relu', input_shape=self.input_shape, name='conv_block1_1'),
            layers.BatchNormalization(name='bn_block1_1'),
            layers.Conv2D(32, (3, 3), activation='relu', name='conv_block1_2'),
            layers.BatchNormalization(name='bn_block1_2'),
            layers.MaxPooling2D((2, 2), name='maxpool_block1'),
            layers.Dropout(0.25, name='dropout_block1'),
            
            # Block 2: 64 filters
            layers.Conv2D(64, (3, 3), activation='relu', name='conv_block2_1'),
            layers.BatchNormalization(name='bn_block2_1'),
            layers.Conv2D(64, (3, 3), activation='relu', name='conv_block2_2'),
            layers.BatchNormalization(name='bn_block2_2'),
            layers.MaxPooling2D((2, 2), name='maxpool_block2'),
            layers.Dropout(0.35, name='dropout_block2'),
            
            # Block 3: 128 filters
            layers.Conv2D(128, (3, 3), activation='relu', name='conv_block3_1'),
            layers.BatchNormalization(name='bn_block3_1'),
            layers.Conv2D(128, (3, 3), activation='relu', name='conv_block3_2'),
            layers.BatchNormalization(name='bn_block3_2'),
            layers.MaxPooling2D((2, 2), name='maxpool_block3'),
            layers.Dropout(0.4, name='dropout_block3'),
            
            # Block 4: 256 filters
            layers.Conv2D(256, (3, 3), activation='relu', name='conv_block4_1'),
            layers.BatchNormalization(name='bn_block4_1'),
            layers.Conv2D(256, (3, 3), activation='relu', name='conv_block4_2'),
            layers.BatchNormalization(name='bn_block4_2'),
            layers.MaxPooling2D((2, 2), name='maxpool_block4'),
            layers.Dropout(0.5, name='dropout_block4'),
            
            # Global Average Pooling
            layers.GlobalAveragePooling2D(name='global_avg_pooling'),
            
            # Dense layers
            layers.Dense(512, activation='relu', name='dense_1'),
            layers.BatchNormalization(name='bn_dense_1'),
            layers.Dropout(0.5, name='dropout_dense_1'),
            
            layers.Dense(256, activation='relu', name='dense_2'),
            layers.BatchNormalization(name='bn_dense_2'),
            layers.Dropout(0.5, name='dropout_dense_2'),
            
            # Output layer
            layers.Dense(1, activation='sigmoid', name='binary_output')
        ])
        
        # Compile το model
        self.model.compile(
            optimizer=Adam(learning_rate=self.learning_rate),
            loss='binary_crossentropy',
            metrics=['accuracy', 'precision', 'recall']
        )
        
        print(" Model built successfully!")
        return self.model
    
    def create_architecture_diagram(self, save_path='our_cnn_architecture.png'):
        """
        Δημιουργεί diagram της αρχιτεκτονικής μας
        """
        print(" Creating architecture diagram...")
        
        fig, ax = plt.subplots(figsize=(16, 10))
        ax.set_xlim(0, 14)
        ax.set_ylim(0, 10)
        ax.axis('off')
        
        # Title
        plt.title('Our Advanced Binary CNN Architecture', fontsize=20, fontweight='bold', pad=20)
        
        # Input
        input_box = FancyBboxPatch((0.5, 4), 1.5, 2, boxstyle="round,pad=0.1", 
                                   facecolor='lightblue', edgecolor='black', linewidth=2)
        ax.add_patch(input_box)
        ax.text(1.25, 5, 'Input\n(224×224×3)\nBrain MRI', ha='center', va='center', 
               fontsize=10, fontweight='bold')
        
        # Conv Blocks
        block_positions = [(2.5, 4), (4.5, 4), (6.5, 4), (8.5, 4)]
        block_info = [
            ('Block 1', '32 filters', 'lightgreen'),
            ('Block 2', '64 filters', 'mediumseagreen'),
            ('Block 3', '128 filters', 'seagreen'),
            ('Block 4', '256 filters', 'darkseagreen')
        ]
        
        for (x, y), (name, filters, color) in zip(block_positions, block_info):
            # Conv block
            block_box = FancyBboxPatch((x, y), 1.5, 2, boxstyle="round,pad=0.1", 
                                       facecolor=color, edgecolor='black', linewidth=2)
            ax.add_patch(block_box)
            
            ax.text(x + 0.75, y + 1.5, name, ha='center', va='center', 
                   fontsize=11, fontweight='bold')
            ax.text(x + 0.75, y + 1.1, filters, ha='center', va='center', fontsize=9)
            ax.text(x + 0.75, y + 0.7, '2×Conv2D\nBatchNorm\nMaxPool\nDropout', 
                   ha='center', va='center', fontsize=8)
        
        # Global Average Pooling
        gap_box = FancyBboxPatch((10, 4.5), 1.5, 1, boxstyle="round,pad=0.1", 
                                 facecolor='yellow', edgecolor='red', linewidth=3)
        ax.add_patch(gap_box)
        ax.text(10.75, 5, 'Global Avg\nPooling', ha='center', va='center', 
               fontsize=10, fontweight='bold', color='red')
        
        # Dense layers
        dense_positions = [(10, 2.5), (10, 1)]
        dense_info = [('Dense 512\n+ BatchNorm\n+ Dropout', 'plum'), 
                      ('Dense 256\n+ BatchNorm\n+ Dropout', 'plum')]
        
        for (x, y), (text, color) in zip(dense_positions, dense_info):
            dense_box = FancyBboxPatch((x, y), 1.5, 1, boxstyle="round,pad=0.05", 
                                       facecolor=color, edgecolor='black')
            ax.add_patch(dense_box)
            ax.text(x + 0.75, y + 0.5, text, ha='center', va='center', fontsize=9)
        
        # Output
        output_box = FancyBboxPatch((12, 4.2), 1.2, 1.6, boxstyle="round,pad=0.1", 
                                    facecolor='gold', edgecolor='black', linewidth=2)
        ax.add_patch(output_box)
        ax.text(12.6, 5, 'Binary\nOutput\n(Sigmoid)\n\n0: No Tumor\n1: Tumor', 
               ha='center', va='center', fontsize=9, fontweight='bold')
        
        # Arrows
        arrow_props = dict(arrowstyle='->', lw=2.5, color='darkblue')
        
        # Input to Block 1
        ax.annotate('', xy=(2.5, 5), xytext=(2, 5), arrowprops=arrow_props)
        
        # Between blocks
        for i in range(len(block_positions) - 1):
            start_x = block_positions[i][0] + 1.5
            end_x = block_positions[i + 1][0]
            ax.annotate('', xy=(end_x, 5), xytext=(start_x, 5), arrowprops=arrow_props)
        
        # Block 4 to GAP
        ax.annotate('', xy=(10, 5), xytext=(10, 5), arrowprops=arrow_props)
        
        # GAP to Dense
        ax.annotate('', xy=(10.75, 3.5), xytext=(10.75, 4.5), arrowprops=arrow_props)
        
        # Between Dense layers
        ax.annotate('', xy=(10.75, 2), xytext=(10.75, 2.5), arrowprops=arrow_props)
        
        # Dense to Output
        ax.annotate('', xy=(12, 5), xytext=(11.5, 1.5), arrowprops=arrow_props)
        
        # Feature maps annotations
        feature_sizes = ['222×222', '110×110', '53×53', '24×24', '10×10']
        for i, (pos, size) in enumerate(zip(block_positions, feature_sizes)):
            ax.text(pos[0] + 0.75, pos[1] - 0.5, f'Feature: {size}', 
                   ha='center', va='center', fontsize=8, style='italic', color='blue')
        
        # Model stats
        stats_text = """
 Model Statistics:
• Total Parameters: 1,442,337
• Model Size: 5.5 MB
• Convolutional Layers: 8
• BatchNorm Layers: 10
• Dropout Layers: 6
• Training Time: ~30-60 min (GPU)
        """
        ax.text(0.5, 8.5, stats_text, fontsize=10, va='top', 
               bbox=dict(boxstyle="round,pad=0.5", facecolor="lightgray", alpha=0.8))
        
        # Advantages
        advantages_text = """
 Advanced Features:
• Double Convolutions per block
• Batch Normalization for stability
• Progressive Dropout (0.25→0.5)
• Global Average Pooling
• Optimized for Medical Imaging
        """
        ax.text(12.5, 2.5, advantages_text, fontsize=10, va='top',
               bbox=dict(boxstyle="round,pad=0.5", facecolor="lightgreen", alpha=0.8))
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f" Architecture diagram saved: {save_path}")
        plt.show()
        
        return save_path
    
    def compare_with_basic_cnn(self, save_path='cnn_comparison.png'):
        """
        Σύγκριση με τα βασικά CNN από τις εικόνες
        """
        print(" Creating comparison with basic CNNs...")
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
        
        # Basic CNN (αριστερά)
        ax1.set_title('Basic CNN\n(από τις εικόνες)', fontsize=14, fontweight='bold')
        ax1.set_xlim(0, 8)
        ax1.set_ylim(0, 10)
        ax1.axis('off')
        
        # Simple layers for basic CNN
        basic_layers = [
            (1, 8, 'Input', 'lightblue'),
            (1, 6.5, 'Conv+Pool', 'lightgreen'),
            (1, 5, 'Conv+Pool', 'lightgreen'),
            (1, 3.5, 'Conv+Pool', 'lightgreen'),
            (1, 2, 'FC', 'pink'),
            (1, 0.5, 'Output', 'gold')
        ]
        
        for x, y, text, color in basic_layers:
            box = FancyBboxPatch((x, y), 1.5, 1, boxstyle="round,pad=0.05", 
                                facecolor=color, edgecolor='black')
            ax1.add_patch(box)
            ax1.text(x + 0.75, y + 0.5, text, ha='center', va='center', fontsize=10)
        
        # Add arrows
        for i in range(len(basic_layers) - 1):
            y_start = basic_layers[i][1]
            y_end = basic_layers[i + 1][1] + 1
            ax1.annotate('', xy=(1.75, y_end), xytext=(1.75, y_start), 
                        arrowprops=dict(arrowstyle='->', lw=2))
        
        ax1.text(4, 5, ' Issues:\n• No BatchNorm\n• No Dropout\n• Simple architecture\n• Prone to overfitting\n• ~5-10M parameters', 
                fontsize=11, va='center', bbox=dict(boxstyle="round", facecolor="mistyrose"))
        
        # Our CNN (δεξιά)
        ax2.set_title('Our Advanced CNN\n(Διπλωματική)', fontsize=14, fontweight='bold')
        ax2.set_xlim(0, 8)
        ax2.set_ylim(0, 10)
        ax2.axis('off')
        
        # Our layers
        our_layers = [
            (1, 8.5, 'Input\n(224×224×3)', 'lightblue'),
            (1, 7, 'Conv Block 1\n32 filters', 'lightgreen'),
            (1, 5.5, 'Conv Block 2\n64 filters', 'mediumseagreen'),
            (1, 4, 'Conv Block 3\n128 filters', 'seagreen'),
            (1, 2.5, 'Conv Block 4\n256 filters', 'darkseagreen'),
            (1, 1, 'Global Avg Pool', 'yellow'),
            (3.5, 2, 'Dense Layers', 'plum'),
            (3.5, 0.5, 'Binary Output', 'gold')
        ]
        
        for x, y, text, color in our_layers:
            width = 2 if 'Conv Block' in text else 1.5
            box = FancyBboxPatch((x, y), width, 1, boxstyle="round,pad=0.05", 
                                facecolor=color, edgecolor='black', linewidth=1.5)
            ax2.add_patch(box)
            ax2.text(x + width/2, y + 0.5, text, ha='center', va='center', fontsize=9, fontweight='bold')
        
        ax2.text(5.5, 5, ' Advantages:\n• BatchNormalization\n• Progressive Dropout\n• Double Convolutions\n• Global Avg Pooling\n• 1.4M parameters\n• Medical optimized', 
                fontsize=11, va='center', bbox=dict(boxstyle="round", facecolor="lightgreen", alpha=0.8))
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f" Comparison diagram saved: {save_path}")
        plt.show()
        
        return save_path
    
    def show_model_info(self):
        """
        Εμφάνιση πληροφοριών του model με visualizations
        """
        if self.model is None:
            print(" Model not built yet. Call build_model() first.")
            return
        
        print("\n Model Architecture Summary:")
        print("=" * 50)
        self.model.summary()
        
        # Count parameters
        total_params = self.model.count_params()
        print(f"\n Total Parameters: {total_params:,}")
        print(f" Estimated Model Size: {total_params * 4 / 1024 / 1024:.1f} MB")
        
        # Create visualizations
        print("\n Creating Architecture Visualizations...")
        arch_path = self.create_architecture_diagram()
        comp_path = self.compare_with_basic_cnn()
        
        return arch_path, comp_path


def main():
    """
    Κύρια συνάρτηση με visualizations
    """
    print(" Binary Brain Tumor Classification - Step 1.5 (With Visualizations)")
    print("=" * 70)
    
    # Δημιουργία model instance
    visual_cnn = VisualBinaryTumorCNN()
    
    # Build το model
    model = visual_cnn.build_model()
    
    # Εμφάνιση πληροφοριών και visualizations
    arch_path, comp_path = visual_cnn.show_model_info()
    
    # Test prediction
    print("\n🧪 Testing model with dummy data...")
    dummy_input = np.random.random((1, 224, 224, 3))
    prediction = model.predict(dummy_input, verbose=0)
    probability = prediction[0][0]
    predicted_class = "Tumor" if probability > 0.5 else "No Tumor"
    
    print(f" Model test successful!")
    print(f" Prediction: {predicted_class} ({probability:.3f})")
    print(f" Confidence: {max(probability, 1-probability):.1%}")
    
    print(f"\n Step 1.5 Complete!")
    print(f" Created files:")
    print(f"    {arch_path}")
    print(f"    {comp_path}")
    print(f" Next: Data Preprocessing (Step 2)")


if __name__ == "__main__":
    main()