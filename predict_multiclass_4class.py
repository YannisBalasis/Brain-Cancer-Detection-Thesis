#!/usr/bin/env python3
"""
🔮 4-Class Brain Tumor Classification - Prediction Script
Simple script για testing trained multiclass model με individual images

Usage:
python predict_multiclass_4class.py --model path/to/model.h5 --image path/to/image.jpg
"""

import os
import argparse
import numpy as np
import cv2
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
from datetime import datetime

class MultiClass4ClassPredictor:
    """
    Simple prediction interface για 4-class multiclass model
    """
    
    def __init__(self, model_path):
        self.model = None
        self.class_names = ['Glioma', 'Meningioma', 'No Tumor', 'Pituitary']
        self.class_colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4']
        
        # Medical information για each class
        self.class_info = {
            'Glioma': {
                'description': 'Aggressive brain tumor από glial cells',
                'urgency': 'URGENT - Immediate Action Required',
                'treatment': 'Immediate oncology consultation, surgical evaluation',
                'risk_level': 'HIGH RISK'
            },
            'Meningioma': {
                'description': 'Usually benign tumor από brain membranes',
                'urgency': 'HIGH PRIORITY - Specialist Consultation Needed', 
                'treatment': 'Neurosurgical consultation, MRI monitoring',
                'risk_level': 'MODERATE RISK'
            },
            'Pituitary': {
                'description': 'Tumor στον pituitary gland',
                'urgency': 'HIGH PRIORITY - Specialist Consultation Needed',
                'treatment': 'Endocrinology referral, hormone assessment',
                'risk_level': 'MODERATE RISK'
            },
            'No Tumor': {
                'description': 'Healthy brain tissue detected',
                'urgency': 'ROUTINE - Standard Follow-up',
                'treatment': 'Continue routine medical care',
                'risk_level': 'LOW RISK'
            }
        }
        
        self.load_model(model_path)
    
    def load_model(self, model_path):
        """Load trained model"""
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")
        
        print(f"📁 Loading model από: {model_path}")
        try:
            self.model = keras.models.load_model(model_path)
            print("✅ Model loaded successfully!")
            print(f"📊 Model parameters: {self.model.count_params():,}")
        except Exception as e:
            raise RuntimeError(f"Error loading model: {e}")
    
    def preprocess_image(self, image_path):
        """Preprocess image για prediction"""
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image file not found: {image_path}")
        
        # Load image
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError(f"Could not load image: {image_path}")
        
        # Convert BGR to RGB
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Resize to model input size
        img_resized = cv2.resize(img_rgb, (224, 224))
        
        # Normalize
        img_normalized = img_resized.astype(np.float32) / 255.0
        
        # Add batch dimension
        img_batch = np.expand_dims(img_normalized, axis=0)
        
        return img_batch, img_rgb
    
    def predict(self, image_path, show_confidence=True):
        """Make prediction on single image"""
        print(f"🔮 Making prediction on: {os.path.basename(image_path)}")
        
        # Preprocess image
        img_batch, original_img = self.preprocess_image(image_path)
        
        # Make prediction
        predictions = self.model.predict(img_batch, verbose=0)
        probabilities = predictions[0]
        
        # Get predicted class
        predicted_class_idx = np.argmax(probabilities)
        predicted_class = self.class_names[predicted_class_idx]
        confidence = probabilities[predicted_class_idx]
        
        # Create results dictionary
        results = {
            'predicted_class': predicted_class,
            'confidence': float(confidence),
            'probabilities': {name: float(prob) for name, prob in zip(self.class_names, probabilities)},
            'image_path': image_path,
            'prediction_time': datetime.now().isoformat()
        }
        
        # Print results
        if show_confidence:
            self._print_prediction_results(results)
        
        return results, original_img
    
    def _print_prediction_results(self, results):
        """Print formatted prediction results"""
        print("\n" + "="*60)
        print("🧠 4-CLASS BRAIN TUMOR CLASSIFICATION RESULTS")
        print("="*60)
        
        predicted = results['predicted_class']
        confidence = results['confidence']
        
        print(f"🎯 PREDICTION: {predicted}")
        print(f"📊 CONFIDENCE: {confidence:.4f} ({confidence*100:.2f}%)")
        
        # Confidence level assessment
        if confidence >= 0.95:
            conf_level = "Very High Confidence"
            conf_icon = "🟢"
        elif confidence >= 0.85:
            conf_level = "High Confidence"  
            conf_icon = "🟡"
        elif confidence >= 0.70:
            conf_level = "Moderate Confidence"
            conf_icon = "🟠"
        else:
            conf_level = "Low Confidence - Expert Review Recommended"
            conf_icon = "🔴"
        
        print(f"{conf_icon} RELIABILITY: {conf_level}")
        
        # Medical information
        info = self.class_info[predicted]
        print(f"\n🏥 MEDICAL INFORMATION:")
        print(f"  📝 Description: {info['description']}")
        print(f"  ⚠️ Risk Level: {info['risk_level']}")
        print(f"  🚨 Urgency: {info['urgency']}")
        print(f"  💊 Recommended Action: {info['treatment']}")
        
        # All probabilities
        print(f"\n📊 ALL CLASS PROBABILITIES:")
        sorted_probs = sorted(results['probabilities'].items(), key=lambda x: x[1], reverse=True)
        for i, (class_name, prob) in enumerate(sorted_probs):
            icon = "🥇" if i == 0 else "🥈" if i == 1 else "🥉" if i == 2 else "4️⃣"
            print(f"  {icon} {class_name}: {prob:.4f} ({prob*100:.2f}%)")
        
        print("\n⚠️ MEDICAL DISCLAIMER:")
        print("This AI system is for research and consultation support only.")
        print("Always consult με qualified medical professionals για diagnosis and treatment.")
        print("="*60)
    
    def visualize_prediction(self, results, original_img, save_path=None):
        """Create visualization of prediction results"""
        predicted = results['predicted_class']
        confidence = results['confidence']
        probabilities = results['probabilities']
        
        # Create figure
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Display original image
        ax1.imshow(original_img)
        ax1.set_title(f'Brain MRI Scan\nPredicted: {predicted} ({confidence*100:.1f}%)', 
                     fontsize=14, fontweight='bold')
        ax1.axis('off')
        
        # Add prediction info as text overlay
        info = self.class_info[predicted]
        text_info = f"Risk: {info['risk_level']}\n{info['urgency']}"
        ax1.text(0.02, 0.98, text_info, transform=ax1.transAxes, 
                verticalalignment='top', fontsize=10,
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        # Probability bar chart
        classes = list(probabilities.keys())
        probs = list(probabilities.values())
        
        bars = ax2.barh(classes, probs, color=self.class_colors)
        ax2.set_xlabel('Probability', fontweight='bold')
        ax2.set_title('Classification Probabilities', fontsize=14, fontweight='bold')
        ax2.set_xlim(0, 1.0)
        
        # Add probability labels on bars
        for bar, prob in zip(bars, probs):
            width = bar.get_width()
            ax2.text(width + 0.01, bar.get_y() + bar.get_height()/2, 
                    f'{prob:.3f}', ha='left', va='center', fontweight='bold')
        
        # Highlight predicted class
        predicted_idx = classes.index(predicted)
        bars[predicted_idx].set_edgecolor('red')
        bars[predicted_idx].set_linewidth(3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"💾 Visualization saved: {save_path}")
        else:
            plt.show()
        
        return fig
    
    def batch_predict(self, image_folder, output_file=None):
        """Make predictions on batch of images"""
        print(f"📁 Processing images από folder: {image_folder}")
        
        if not os.path.exists(image_folder):
            raise FileNotFoundError(f"Folder not found: {image_folder}")
        
        # Find all image files
        image_extensions = ('.png', '.jpg', '.jpeg', '.bmp', '.tiff')
        image_files = [f for f in os.listdir(image_folder) 
                      if f.lower().endswith(image_extensions)]
        
        if not image_files:
            print("❌ No image files found")
            return []
        
        print(f"📊 Found {len(image_files)} images")
        
        # Process each image
        results_list = []
        for i, image_file in enumerate(image_files, 1):
            image_path = os.path.join(image_folder, image_file)
            print(f"\n[{i}/{len(image_files)}] Processing: {image_file}")
            
            try:
                results, _ = self.predict(image_path, show_confidence=False)
                results_list.append(results)
                
                # Brief summary
                pred = results['predicted_class']
                conf = results['confidence']
                print(f"  🎯 {pred} ({conf*100:.1f}%)")
                
            except Exception as e:
                print(f"  ❌ Error: {e}")
                continue
        
        # Save results if requested
        if output_file and results_list:
            import json
            with open(output_file, 'w') as f:
                json.dump(results_list, f, indent=2)
            print(f"\n💾 Batch results saved: {output_file}")
        
        # Summary statistics
        if results_list:
            print(f"\n📊 BATCH SUMMARY:")
            class_counts = {}
            total_confidence = 0
            
            for result in results_list:
                pred = result['predicted_class']
                conf = result['confidence']
                
                class_counts[pred] = class_counts.get(pred, 0) + 1
                total_confidence += conf
            
            print(f"  Total processed: {len(results_list)}")
            print(f"  Average confidence: {total_confidence/len(results_list):.3f}")
            print("  Class distribution:")
            for class_name, count in class_counts.items():
                percentage = count / len(results_list) * 100
                print(f"    {class_name}: {count} ({percentage:.1f}%)")
        
        return results_list


def main():
    """Main function για command line usage"""
    parser = argparse.ArgumentParser(description='4-Class Brain Tumor Prediction')
    parser.add_argument('--model', type=str, required=True,
                       help='Path to trained model (.h5 file)')
    parser.add_argument('--image', type=str,
                       help='Path to single image για prediction')
    parser.add_argument('--folder', type=str,
                       help='Path to folder containing images για batch prediction')
    parser.add_argument('--output', type=str,
                       help='Output file για saving results (JSON format)')
    parser.add_argument('--visualize', action='store_true',
                       help='Show visualization of prediction')
    parser.add_argument('--save-viz', type=str,
                       help='Save visualization to file')
    
    args = parser.parse_args()
    
    # Validate arguments
    if not args.image and not args.folder:
        print("❌ Must provide either --image ή --folder")
        return
    
    if args.image and args.folder:
        print("❌ Cannot use both --image και --folder simultaneously")
        return
    
    # Print header
    print("🧠 4-CLASS BRAIN TUMOR CLASSIFICATION - PREDICTION")
    print("=" * 60)
    print(f"📅 Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    try:
        # Initialize predictor
        predictor = MultiClass4ClassPredictor(args.model)
        
        if args.image:
            # Single image prediction
            results, original_img = predictor.predict(args.image)
            
            # Save results if requested
            if args.output:
                import json
                with open(args.output, 'w') as f:
                    json.dump(results, f, indent=2)
                print(f"💾 Results saved: {args.output}")
            
            # Visualization
            if args.visualize or args.save_viz:
                predictor.visualize_prediction(results, original_img, args.save_viz)
        
        elif args.folder:
            # Batch prediction
            results_list = predictor.batch_predict(args.folder, args.output)
            
            if not results_list:
                print("❌ No successful predictions")
                return
        
        print("\n✅ Prediction completed successfully!")
        
    except Exception as e:
        print(f"❌ Error during prediction: {e}")
        return


if __name__ == "__main__":
    main()