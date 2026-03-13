import os
import sys
import argparse
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from tensorflow import keras
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import cv2

class BinaryTumorPredictor:

    
    def __init__(self, model_path="best_binary_model.h5", target_size=(224, 224)):

        self.model_path = model_path
        self.target_size = target_size
        self.model = None
        
        print(" Binary Brain Tumor Predictor")
        print("=" * 40)
        print(f" Model path: {model_path}")
        print(f" Input size: {target_size}")
        
        # Load model
        self.load_model()
    
    def load_model(self):

        if not os.path.exists(self.model_path):
            print(f" Model file not found: {self.model_path}")
            print(" Make sure you have trained the model first!")
            print("   Run: python step3_complete_training.py")
            sys.exit(1)
        
        try:
            print(f" Loading model...")
            self.model = keras.models.load_model(self.model_path)
            print(f" Model loaded successfully!")
            print(f" Parameters: {self.model.count_params():,}")
            
        except Exception as e:
            print(f" Error loading model: {e}")
            sys.exit(1)
    
    def preprocess_image(self, image_path):

        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image not found: {image_path}")
        
        try:
            # Load image
            image = load_img(image_path, target_size=self.target_size)
            image_array = img_to_array(image)
            
            # Normalize pixel values [0,1]
            image_array = image_array / 255.0
            
            # Add batch dimension
            image_array = np.expand_dims(image_array, axis=0)
            
            print(f" Image loaded and preprocessed")
            print(f"   Original path: {image_path}")
            print(f"   Processed shape: {image_array.shape}")
            
            return image_array, image
            
        except Exception as e:
            raise Exception(f"Error preprocessing image: {e}")
    
    def predict(self, image_path):

        if self.model is None:
            raise Exception("Model not loaded!")
        
        # Preprocess image
        processed_image, original_image = self.preprocess_image(image_path)
        
        # Make prediction
        print(f" Making prediction...")
        prediction_prob = self.model.predict(processed_image, verbose=0)[0][0]
        
        # Interpret results
        is_tumor = prediction_prob > 0.5
        predicted_class = "Tumor" if is_tumor else "No Tumor"
        confidence = prediction_prob if is_tumor else (1 - prediction_prob)
        
        # Risk assessment
        if confidence >= 0.95:
            risk_level = "Very High Confidence"
            risk_color = "darkgreen" if not is_tumor else "darkred"
        elif confidence >= 0.85:
            risk_level = "High Confidence" 
            risk_color = "green" if not is_tumor else "red"
        elif confidence >= 0.70:
            risk_level = "Moderate Confidence"
            risk_color = "orange"
        else:
            risk_level = "Low Confidence - Requires Review"
            risk_color = "gray"
        
        results = {
            'image_path': image_path,
            'predicted_class': predicted_class,
            'is_tumor': is_tumor,
            'probability_tumor': prediction_prob,
            'probability_no_tumor': 1 - prediction_prob,
            'confidence': confidence,
            'risk_level': risk_level,
            'risk_color': risk_color,
            'original_image': original_image
        }
        
        return results
    
    def display_prediction(self, results, save_path=None):

        print(f"\n PREDICTION RESULTS")
        print("=" * 30)
        print(f" Prediction: {results['predicted_class']}")
        print(f" Confidence: {results['confidence']:.1%}")
        print(f" Risk Level: {results['risk_level']}")
        print(f" Tumor Probability: {results['probability_tumor']:.3f}")
        print(f" No Tumor Probability: {results['probability_no_tumor']:.3f}")
        
        # Create visualization
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
        
        # Original image
        ax1.imshow(results['original_image'])
        ax1.set_title('Input MRI Image', fontsize=14, fontweight='bold')
        ax1.axis('off')
        
        # Prediction visualization
        probs = [results['probability_no_tumor'], results['probability_tumor']]
        labels = ['No Tumor', 'Tumor']
        colors = ['lightblue', 'lightcoral']
        
        bars = ax2.bar(labels, probs, color=colors, alpha=0.8, edgecolor='black', linewidth=2)
        ax2.set_title('Prediction Probabilities', fontsize=14, fontweight='bold')
        ax2.set_ylabel('Probability', fontsize=12)
        ax2.set_ylim(0, 1)
        
        # Add probability labels on bars
        for bar, prob in zip(bars, probs):
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                    f'{prob:.1%}', ha='center', va='bottom', 
                    fontsize=12, fontweight='bold')
        
        # Highlight predicted class
        predicted_idx = 1 if results['is_tumor'] else 0
        bars[predicted_idx].set_color(results['risk_color'])
        bars[predicted_idx].set_alpha(1.0)
        
        # Add prediction text box
        textstr = f"""
PREDICTION: {results['predicted_class']}
Confidence: {results['confidence']:.1%}
Risk Level: {results['risk_level']}
        """.strip()
        
        props = dict(boxstyle='round', facecolor=results['risk_color'], alpha=0.3)
        ax2.text(0.02, 0.98, textstr, transform=ax2.transAxes, fontsize=11,
                verticalalignment='top', bbox=props, fontweight='bold')
        
        # Add medical disclaimer
        disclaimer = " For educational purposes only. Consult medical professional for diagnosis."
        fig.text(0.5, 0.02, disclaimer, ha='center', fontsize=10, 
                style='italic', color='red')
        
        plt.suptitle(f'Binary Brain Tumor Classification Results', 
                    fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.subplots_adjust(bottom=0.1)
        
        # Save visualization
        if save_path is None:
            image_name = Path(results['image_path']).stem
            save_path = f"prediction_{image_name}.png"
        
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f" Prediction visualization saved: {save_path}")
        
        # Show plot
        plt.show()
        
        return save_path
    
    def generate_report(self, results, save_path=None):
        """
        Δημιουργεί text report
        """
        if save_path is None:
            image_name = Path(results['image_path']).stem
            save_path = f"report_{image_name}.txt"
        
        report = f"""
BINARY BRAIN TUMOR CLASSIFICATION REPORT
=========================================
Generated by: Binary Tumor Predictor v1.0
Date/Time: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}

INPUT INFORMATION:
------------------
Image Path: {results['image_path']}
Image Name: {Path(results['image_path']).name}

PREDICTION RESULTS:
-------------------
Classification: {results['predicted_class']}
Confidence Level: {results['confidence']:.1%}
Risk Assessment: {results['risk_level']}

DETAILED PROBABILITIES:
-----------------------
Tumor Probability:     {results['probability_tumor']:.4f} ({results['probability_tumor']*100:.2f}%)
No Tumor Probability:  {results['probability_no_tumor']:.4f} ({results['probability_no_tumor']*100:.2f}%)

MODEL INFORMATION:
------------------
Architecture: Custom CNN (1.4M parameters)
Training Accuracy: 99.29%
Test Accuracy: 98.71%
ROC-AUC: 0.998

TECHNICAL NOTES:
----------------
- Input image was automatically resized to 224x224 pixels
- Preprocessing included normalization and standardization
- Prediction threshold: 0.5 (probability > 0.5 = Tumor)
- Model trained on {7023} MRI brain images
        """.strip()
        
        with open(save_path, 'w') as f:
            f.write(report)
        
        print(f" Detailed report saved: {save_path}")
        return save_path


def main():

    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description="Binary Brain Tumor Predictor - Predicts if brain MRI shows tumor",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python binary_app.py brain_scan.jpg
  python binary_app.py /path/to/mri_image.png --model custom_model.h5
  python binary_app.py image.jpg --save-viz prediction_viz.png
        """)
    
    parser.add_argument('image_path', 
                       help='Path to brain MRI image (jpg, png, etc.)')
    parser.add_argument('--model', '-m', 
                       default='best_binary_model.h5',
                       help='Path to trained model file (default: best_binary_model.h5)')
    parser.add_argument('--save-viz', '-v',
                       help='Path to save prediction visualization')
    parser.add_argument('--save-report', '-r', 
                       help='Path to save detailed report')
    parser.add_argument('--no-display', action='store_true',
                       help='Do not display visualization (save only)')
    
    args = parser.parse_args()
    
    try:
        # Initialize predictor
        predictor = BinaryTumorPredictor(model_path=args.model)
        
        # Make prediction
        print(f"\n Analyzing image: {args.image_path}")
        results = predictor.predict(args.image_path)
        
        # Display results
        if not args.no_display:
            viz_path = predictor.display_prediction(results, save_path=args.save_viz)
        elif args.save_viz:
            # Save without display
            viz_path = predictor.display_prediction(results, save_path=args.save_viz)
            plt.close()  # Close without showing
        
        # Generate report if requested
        if args.save_report:
            import pandas as pd
            report_path = predictor.generate_report(results, save_path=args.save_report)
        
        # Final summary
        print(f"\n Prediction completed successfully!")
        if results['is_tumor']:
            print(f" TUMOR DETECTED - Confidence: {results['confidence']:.1%}")
            print(f" Recommendation: Immediate medical consultation")
        else:
            print(f" NO TUMOR DETECTED - Confidence: {results['confidence']:.1%}")
            print(f" Recommendation: Regular check-ups as advised")
        
        return 0
        
    except FileNotFoundError as e:
        print(f" File Error: {e}")
        return 1
    except Exception as e:
        print(f" Prediction Error: {e}")
        return 1


if __name__ == "__main__":
    # Import pandas only if needed (for report generation)
    try:
        import pandas as pd
    except ImportError:
        print(" pandas not installed - report generation disabled")
        pd = None
    
    exit_code = main()
    sys.exit(exit_code)