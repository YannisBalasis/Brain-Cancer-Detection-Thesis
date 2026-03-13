

import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

class SmartBrainMRIPreprocessor:
    """
    Smart preprocessor για brain MRI images που μετατρέπει multiclass σε binary
    """
    
    def __init__(self, target_size=(224, 224), batch_size=32):
        """
        Αρχικοποίηση του preprocessor
        
        Args:
            target_size: Μέγεθος εικόνων (height, width)
            batch_size: Batch size για training
        """
        self.target_size = target_size
        self.batch_size = batch_size
        
        # Ορισμός binary mapping
        self.tumor_classes = ['glioma', 'meningioma', 'pituitary']  # → tumor
        self.no_tumor_classes = ['notumor']  # → no_tumor
        
        print(" Smart Brain MRI Preprocessor Initialized")
        print(f" Target size: {target_size}")
        print(f" Batch size: {batch_size}")
        print(f" Binary mapping:")
        print(f"   Tumor: {self.tumor_classes}")
        print(f"   No Tumor: {self.no_tumor_classes}")
    
    def analyze_dataset(self, data_dir):
        """
        Αναλύει τη δομή του dataset και μετράει εικόνες
        """
        print(f"\n Analyzing dataset: {data_dir}")
        print("=" * 50)
        
        if not os.path.exists(data_dir):
            print(f" Dataset directory not found: {data_dir}")
            return None
        
        dataset_stats = {
            'Training': {},
            'Testing': {},
            'total_tumor': 0,
            'total_no_tumor': 0
        }
        
        # Ανάλυση Training και Testing
        for split in ['Training', 'Testing']:
            split_dir = os.path.join(data_dir, split)
            
            if not os.path.exists(split_dir):
                print(f" {split} directory not found")
                continue
            
            print(f"\n {split} Data:")
            split_tumor = 0
            split_no_tumor = 0
            
            # Έλεγχος κάθε κλάσης
            for class_name in os.listdir(split_dir):
                class_path = os.path.join(split_dir, class_name)
                
                if not os.path.isdir(class_path):
                    continue
                
                # Μέτρημα εικόνων
                images = [f for f in os.listdir(class_path) 
                         if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))]
                count = len(images)
                
                # Binary mapping
                if class_name.lower() in [c.lower() for c in self.tumor_classes]:
                    binary_label = "tumor"
                    split_tumor += count
                    dataset_stats['total_tumor'] += count
                elif class_name.lower() in [c.lower() for c in self.no_tumor_classes]:
                    binary_label = "no_tumor"
                    split_no_tumor += count
                    dataset_stats['total_no_tumor'] += count
                else:
                    binary_label = "unknown"
                    print(f"⚠️ Unknown class: {class_name}")
                
                dataset_stats[split][class_name] = count
                print(f"    {class_name}: {count} images → {binary_label}")
            
            print(f"    {split} Summary: {split_tumor} tumor, {split_no_tumor} no_tumor")
        
        # Συνολικά αποτελέσματα
        total_images = dataset_stats['total_tumor'] + dataset_stats['total_no_tumor']
        tumor_percentage = (dataset_stats['total_tumor'] / total_images * 100) if total_images > 0 else 0
        
        print(f"\n Overall Dataset Summary:")
        print(f"    Tumor: {dataset_stats['total_tumor']} images ({tumor_percentage:.1f}%)")
        print(f"    No Tumor: {dataset_stats['total_no_tumor']} images ({100-tumor_percentage:.1f}%)")
        print(f"    Total: {total_images} images")
        
        # Visualization
        self._plot_dataset_distribution(dataset_stats)
        
        return dataset_stats
    
    def _plot_dataset_distribution(self, stats):
        """
        Visualization του dataset distribution
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
        
        # Binary distribution
        binary_counts = [stats['total_tumor'], stats['total_no_tumor']]
        binary_labels = ['Tumor', 'No Tumor']
        colors = ['lightcoral', 'lightblue']
        
        bars1 = ax1.bar(binary_labels, binary_counts, color=colors, alpha=0.8)
        ax1.set_title('Binary Classification Distribution', fontsize=14, fontweight='bold')
        ax1.set_ylabel('Number of Images')
        
        # Προσθήκη αριθμών στα bars
        for bar, count in zip(bars1, binary_counts):
            ax1.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 20,
                    f'{count}', ha='center', va='bottom', fontweight='bold')
        
        # Detailed class distribution
        all_classes = []
        all_counts = []
        
        for split in ['Training', 'Testing']:
            if split in stats:
                for class_name, count in stats[split].items():
                    all_classes.append(f"{split}\n{class_name}")
                    all_counts.append(count)
        
        bars2 = ax2.bar(range(len(all_classes)), all_counts, 
                       color=['lightgreen' if 'Training' in cls else 'lightsteelblue' for cls in all_classes])
        ax2.set_title('Detailed Class Distribution', fontsize=14, fontweight='bold')
        ax2.set_ylabel('Number of Images')
        ax2.set_xticks(range(len(all_classes)))
        ax2.set_xticklabels(all_classes, rotation=45, ha='right')
        
        # Προσθήκη αριθμών
        for bar, count in zip(bars2, all_counts):
            ax2.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 5,
                    f'{count}', ha='center', va='bottom', fontsize=9)
        
        plt.tight_layout()
        plt.savefig('dataset_distribution.png', dpi=300, bbox_inches='tight')
        print(" Dataset distribution saved as 'dataset_distribution.png'")
        plt.show()
    
    def create_binary_dataframe(self, data_dir):
        """
        Δημιουργεί DataFrame με image paths και binary labels
        """
        print("\n Creating binary classification DataFrame...")
        
        image_paths = []
        binary_labels = []
        original_classes = []
        splits = []
        
        # Επεξεργασία Training και Testing
        for split in ['Training', 'Testing']:
            split_dir = os.path.join(data_dir, split)
            
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
                    binary_label = 'tumor'  # Default για άγνωστες κλάσεις
                
                # Collect image paths
                for image_file in os.listdir(class_path):
                    if image_file.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
                        image_path = os.path.join(class_path, image_file)
                        image_paths.append(image_path)
                        binary_labels.append(binary_label)
                        original_classes.append(class_name)
                        splits.append(split)
        
        # Δημιουργία DataFrame
        df = pd.DataFrame({
            'image_path': image_paths,
            'binary_label': binary_labels,
            'original_class': original_classes,
            'split': splits
        })
        
        print(f" DataFrame created with {len(df)} images")
        print(f" Binary distribution:")
        print(df['binary_label'].value_counts())
        
        return df
    
    def create_data_generators(self, data_dir):
        """
        Δημιουργεί training, validation, και test generators
        """
        print("\n Creating data generators...")
        
        # Δημιουργία DataFrame
        df = self.create_binary_dataframe(data_dir)
        
        # Split σε train/val/test (70/20/10)
        train_df, temp_df = train_test_split(
            df, test_size=0.3, stratify=df['binary_label'], random_state=42
        )
        val_df, test_df = train_test_split(
            temp_df, test_size=0.33, stratify=temp_df['binary_label'], random_state=42
        )
        
        print(f" Data splits:")
        print(f"    Training: {len(train_df)} images")
        print(f"    Validation: {len(val_df)} images")
        print(f"    Testing: {len(test_df)} images")
        
        # Data Augmentation για training
        train_datagen = ImageDataGenerator(
            rescale=1./255,
            
            # Geometric augmentations (ασφαλείς για brain MRI)
            rotation_range=15,           # Μικρές περιστροφές
            width_shift_range=0.1,       # Horizontal shifts
            height_shift_range=0.1,      # Vertical shifts
            shear_range=0.1,            # Shear transformations
            zoom_range=0.1,             # Zoom in/out
            horizontal_flip=True,        # OK για brain (συμμετρία)
            vertical_flip=False,         # ΌΧΙ για brain MRI
            fill_mode='nearest',
            
            # Intensity augmentations (για MRI scanner variations)
            brightness_range=[0.8, 1.2],
            channel_shift_range=0.1
        )
        
        # Validation και Test (μόνο normalization)
        val_test_datagen = ImageDataGenerator(rescale=1./255)
        
        # Δημιουργία generators
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
    
    def test_generators(self, train_gen, val_gen, test_gen):
        """
        Τεστάρει τους generators και δείχνει sample εικόνες
        """
        print("\n Testing data generators...")
        
        # Test ενός batch
        sample_batch_x, sample_batch_y = next(train_gen)
        
        print(f" Generator test successful!")
        print(f" Batch shape: {sample_batch_x.shape}")
        print(f" Labels shape: {sample_batch_y.shape}")
        print(f" Pixel value range: [{sample_batch_x.min():.3f}, {sample_batch_x.max():.3f}]")
        print(f" Sample labels: {sample_batch_y[:5]}")
        
        # Visualization του πρώτου batch
        self._visualize_sample_batch(sample_batch_x, sample_batch_y)
        
    def _visualize_sample_batch(self, batch_x, batch_y):
        """
        Visualize sample εικόνες από έναν batch
        """
        fig, axes = plt.subplots(2, 4, figsize=(16, 8))
        axes = axes.ravel()
        
        for i in range(min(8, len(batch_x))):
            axes[i].imshow(batch_x[i])
            label = "Tumor" if batch_y[i] > 0.5 else "No Tumor"
            axes[i].set_title(f"{label} (conf: {batch_y[i]:.3f})")
            axes[i].axis('off')
        
        plt.suptitle('Sample Training Images (After Preprocessing)', fontsize=16)
        plt.tight_layout()
        plt.savefig('sample_training_images.png', dpi=300, bbox_inches='tight')
        print("📸 Sample images saved as 'sample_training_images.png'")
        plt.show()


def main():
    """
    Κύρια συνάρτηση για testing του preprocessor
    """
    print(" Brain MRI Data Preprocessing - Step 2")
    print("=" * 60)
    
    # Το path του dataset σου
    data_dir = "/Users/yannisbalasis/Documents/thesis/dataset_binary_class"
    
    # Δημιουργία preprocessor
    preprocessor = SmartBrainMRIPreprocessor(target_size=(224, 224), batch_size=32)
    
    # Ανάλυση dataset
    dataset_stats = preprocessor.analyze_dataset(data_dir)
    
    if dataset_stats is None:
        print(" Cannot proceed without valid dataset")
        print(" Update the 'data_dir' path in this script to your dataset location")
        return
    
    # Δημιουργία data generators
    train_gen, val_gen, test_gen = preprocessor.create_data_generators(data_dir)
    
    # Test των generators
    preprocessor.test_generators(train_gen, val_gen, test_gen)
    
    print(f"\n Step 2 Complete!")
    print(f" Ready for training!")
    print(f" Generated files:")
    print(f"    dataset_distribution.png")
    print(f"    sample_training_images.png")
    print(f" Next: Training the model (Step 3)")


if __name__ == "__main__":
    main()