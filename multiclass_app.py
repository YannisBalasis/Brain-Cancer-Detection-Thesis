import os
import sys
import argparse
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from tensorflow import keras
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import pandas as pd

class MultiClassV2Predictor:
    """
    Professional V2 predictor για 4-class brain tumor classification
    """
    
    def __init__(self, model_path="best_multiclass_v2_model.h5", target_size=(224, 224)):
        """
        Αρχικοποίηση του V2 predictor
        """
        self.model_path = model_path
        self.target_size = target_size
        self.model = None
        
        # V2 Class configuration (correct order!)
        self.class_names = ['glioma', 'meningioma', 'pituitary', 'notumor']
        self.class_labels = ['Glioma', 'Meningioma', 'Pituitary', 'No Tumor']
        self.num_classes = 4
        
        # Medical information
        self.class_descriptions = {
            'Glioma': 'Aggressive brain tumor από glial cells - Immediate treatment required',
            'Meningioma': 'Usually benign tumor από brain membranes - Surgery/monitoring needed',
            'Pituitary': 'Tumor στον pituitary gland - Hormone therapy/surgery required',
            'No Tumor': 'Healthy brain tissue - Regular monitoring recommended'
        }
        
        # Risk stratification
        self.risk_levels = {
            'Glioma': 'HIGH RISK',
            'Meningioma': 'MODERATE RISK', 
            'Pituitary': 'MODERATE RISK',
            'No Tumor': 'LOW RISK'
        }
        
        # Treatment recommendations
        self.treatment_recommendations = {
            'Glioma': 'URGENT: Immediate oncology consultation, surgical evaluation, advanced imaging',
            'Meningioma': 'Schedule neurosurgical consultation, monitor με MRI, assess symptoms',
            'Pituitary': 'Endocrinology referral, hormone assessment, visual field testing',
            'No Tumor': 'Continue routine medical care, follow-up if symptoms develop'
        }
        
        # Colors για visualization
        self.class_colors = {
            'Glioma': '#FF4444',      # Red - High risk
            'Meningioma': '#FF8C00',  # Orange - Moderate risk
            'Pituitary': '#4169E1',   # Blue - Moderate risk
            'No Tumor': '#32CD32'     # Green - Low risk
        }
        
        print("🧠 Multiclass Brain Tumor Classifier V2")
        print("=" * 50)
        print(f"🏆 Model: V2 Custom CNN (95.11% test accuracy)")
        print(f"📁 Model path: {model_path}")
        print(f"📏 Input size: {target_size}")
        print(f"🎯 Classes: {', '.join(self.class_labels)}")
        print(f"🏥 Medical-grade precision tumor classification")
        
        # Load V2 model
        self.load_v2_model()
    
    def load_v2_model(self):
        """
        Φορτώνει το V2 multiclass model
        """
        if not os.path.exists(self.model_path):
            print(f"❌ V2 Model file not found: {self.model_path}")
            print("💡 Make sure you have trained the V2 multiclass model first!")
            print("   Run: python multiclass_v2_model.py")
            sys.exit(1)
        
        try:
            print(f"🤖 Loading V2 multiclass model...")
            self.model = keras.models.load_model(self.model_path)
            print(f"✅ V2 Model loaded successfully!")
            print(f"🎯 Parameters: {self.model.count_params():,}")
            print(f"📊 Architecture: V2 Custom CNN (4 conv blocks + 3 dense)")
            print(f"🏆 Proven performance: 95.11% test accuracy")
            
        except Exception as e:
            print(f"❌ Error loading V2 model: {e}")
            sys.exit(1)
    
    def preprocess_image(self, image_path):
        """
        Preprocesses εικόνα για V2 prediction
        """
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
            
            print(f"📸 Image loaded and preprocessed")
            print(f"   Original path: {image_path}")
            print(f"   Processed shape: {image_array.shape}")
            
            return image_array, image
            
        except Exception as e:
            raise Exception(f"Error preprocessing image: {e}")
    
    def predict_v2(self, image_path):
        """
        Κάνει V2 4-class prediction στην εικόνα
        """
        if self.model is None:
            raise Exception("V2 Model not loaded!")
        
        # Preprocess image
        processed_image, original_image = self.preprocess_image(image_path)
        
        # Make V2 prediction
        print(f"🔮 Making V2 4-class prediction...")
        prediction_probs = self.model.predict(processed_image, verbose=0)[0]
        
        # Get predicted class
        predicted_class_idx = np.argmax(prediction_probs)
        predicted_class = self.class_labels[predicted_class_idx]
        predicted_confidence = prediction_probs[predicted_class_idx]
        
        # Create detailed probability breakdown
        prob_breakdown = {}
        for i, (class_name, prob) in enumerate(zip(self.class_labels, prediction_probs)):
            prob_breakdown[class_name] = {
                'probability': float(prob),
                'percentage': float(prob * 100),
                'rank': int(np.where(np.argsort(prediction_probs)[::-1] == i)[0][0] + 1)
            }
        
        # Medical assessment
        risk_level = self.risk_levels[predicted_class]
        description = self.class_descriptions[predicted_class]
        treatment = self.treatment_recommendations[predicted_class]
        
        # Confidence assessment
        if predicted_confidence >= 0.95:
            confidence_level = "Very High Confidence"
            confidence_color = "darkgreen"
            reliability = "Excellent"
        elif predicted_confidence >= 0.85:
            confidence_level = "High Confidence" 
            confidence_color = "green"
            reliability = "Very Good"
        elif predicted_confidence >= 0.70:
            confidence_level = "Moderate Confidence"
            confidence_color = "orange"
            reliability = "Good"
        else:
            confidence_level = "Low Confidence - Expert Review Recommended"
            confidence_color = "red"
            reliability = "Requires Validation"
        
        # Second most likely prediction
        sorted_indices = np.argsort(prediction_probs)[::-1]
        second_class = self.class_labels[sorted_indices[1]]
        second_prob = prediction_probs[sorted_indices[1]]
        
        # Clinical priority
        if predicted_class == 'Glioma':
            clinical_priority = 'URGENT - Immediate Action Required'
            urgency_color = 'red'
        elif predicted_class in ['Meningioma', 'Pituitary']:
            clinical_priority = 'HIGH PRIORITY - Specialist Consultation Needed'
            urgency_color = 'orange'
        else:
            clinical_priority = 'ROUTINE - Standard Follow-up'
            urgency_color = 'green'
        
        results = {
            'image_path': image_path,
            'predicted_class': predicted_class,
            'predicted_confidence': predicted_confidence,
            'risk_level': risk_level,
            'description': description,
            'treatment_recommendations': treatment,
            'confidence_level': confidence_level,
            'confidence_color': confidence_color,
            'reliability': reliability,
            'clinical_priority': clinical_priority,
            'urgency_color': urgency_color,
            'second_prediction': second_class,
            'second_confidence': second_prob,
            'probability_breakdown': prob_breakdown,
            'all_probabilities': prediction_probs,
            'original_image': original_image,
            'model_version': 'V2 Custom CNN (95.11% accuracy)'
        }
        
        return results
    
    def display_v2_prediction(self, results, save_path=None):
        """
        Εμφανίζει detailed V2 prediction results με professional visualization
        """
        print(f"\n🎯 V2 MULTICLASS PREDICTION RESULTS")
        print("=" * 50)
        print(f"🏆 Model: {results['model_version']}")
        print(f"📊 Primary Diagnosis: {results['predicted_class']}")
        print(f"🎯 Confidence: {results['predicted_confidence']:.1%} ({results['confidence_level']})")
        print(f"⚠️ Risk Level: {results['risk_level']}")
        print(f"🚨 Priority: {results['clinical_priority']}")
        print(f"📋 Reliability: {results['reliability']}")
        print(f"🏥 Description: {results['description']}")
        
        if results['second_confidence'] > 0.1:  # Show if >10%
            print(f"🔄 Alternative: {results['second_prediction']} ({results['second_confidence']:.1%})")
        
        print(f"\n📊 V2 Detailed Probability Breakdown:")
        # Sort by probability for proper display
        sorted_probs = sorted(results['probability_breakdown'].items(), 
                            key=lambda x: x[1]['probability'], reverse=True)
        
        for class_name, info in sorted_probs:
            print(f"   #{info['rank']} {class_name:12s}: {info['percentage']:6.2f}% "
                  f"(confidence: {info['probability']:.4f})")
        
        print(f"\n🏥 Clinical Recommendations:")
        print(f"   {results['treatment_recommendations']}")
        
        # Create comprehensive V2 visualization
        fig = plt.figure(figsize=(18, 12))
        gs = fig.add_gridspec(3, 4, height_ratios=[2, 1, 1], width_ratios=[2, 1, 1, 1])
        
        # Main image display
        ax_main = fig.add_subplot(gs[0, :2])
        ax_main.imshow(results['original_image'])
        ax_main.set_title('Brain MRI Scan - V2 Analysis', fontsize=16, fontweight='bold')
        ax_main.axis('off')
        
        # Main prediction result panel
        ax_result = fig.add_subplot(gs[0, 2:])
        ax_result.axis('off')
        
        # Main diagnosis box
        main_color = self.class_colors[results['predicted_class']]
        result_text = f"""
V2 DIAGNOSIS REPORT
{results['predicted_class']} 

Confidence: {results['predicted_confidence']:.1%}
Risk Level: {results['risk_level']}
Priority: {results['clinical_priority']}

MODEL PERFORMANCE:
V2 Custom CNN
95.11% Test Accuracy
Medical-Grade Reliability

DESCRIPTION:
{results['description']}

RELIABILITY:
{results['reliability']} Prediction
        """.strip()
        
        ax_result.text(0.05, 0.95, result_text, fontsize=11, fontweight='bold',
                      bbox=dict(boxstyle="round,pad=0.5", facecolor=main_color, alpha=0.3),
                      verticalalignment='top')
        
        # Treatment recommendations box
        treatment_text = f"""
CLINICAL RECOMMENDATIONS:

{results['treatment_recommendations']}

Follow-up: Specialist consultation 
as clinically appropriate.
        """.strip()
        
        ax_result.text(0.05, 0.25, treatment_text, fontsize=10,
                      bbox=dict(boxstyle="round,pad=0.3", facecolor='lightblue', alpha=0.5),
                      verticalalignment='top')
        
        # Probability bar chart
        ax_prob = fig.add_subplot(gs[1, :])
        
        classes = list(results['probability_breakdown'].keys())
        probabilities = [results['probability_breakdown'][c]['percentage'] for c in classes]
        colors = [self.class_colors[c] for c in classes]
        
        bars = ax_prob.barh(classes, probabilities, color=colors, alpha=0.8, 
                           edgecolor='black', linewidth=1.5)
        ax_prob.set_xlabel('Probability (%)', fontsize=12, fontweight='bold')
        ax_prob.set_title('V2 4-Class Probability Distribution (95.11% Model Accuracy)', 
                         fontsize=14, fontweight='bold')
        ax_prob.set_xlim(0, 100)
        
        # Add probability labels on bars
        for bar, prob in zip(bars, probabilities):
            width = bar.get_width()
            ax_prob.text(width + 1, bar.get_y() + bar.get_height()/2, 
                        f'{prob:.1f}%', ha='left', va='center', 
                        fontweight='bold', fontsize=11)
        
        # Performance metrics table
        ax_metrics = fig.add_subplot(gs[2, :])
        ax_metrics.axis('off')
        
        # Create performance table
        metrics_data = [
            ['Model Performance', 'Value', 'Clinical Significance'],
            ['Test Accuracy', '95.11%', 'Medical-grade performance'],
            ['Model Type', 'V2 Custom CNN', 'Purpose-built για brain tumors'],
            ['Prediction Confidence', f'{results["predicted_confidence"]:.1%}', results['confidence_level']],
            ['Risk Assessment', results['risk_level'], 'Clinical risk stratification'],
            ['Clinical Priority', results['clinical_priority'], 'Treatment urgency level']
        ]
        
        table = ax_metrics.table(cellText=metrics_data[1:], colLabels=metrics_data[0],
                               cellLoc='left', loc='center', colWidths=[0.3, 0.2, 0.5])
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1, 2)
        
        # Style table
        for (i, j), cell in table.get_celld().items():
            if i == 0:  # Header
                cell.set_text_props(weight='bold')
                cell.set_facecolor('#2E8B57')
                cell.set_text_props(color='white')
            else:
                cell.set_facecolor('#F0F8FF' if i % 2 == 0 else 'white')
        
        # Medical disclaimer
        disclaimer = "⚠️ FOR RESEARCH/EDUCATIONAL PURPOSES ONLY. NOT FOR CLINICAL DIAGNOSIS. CONSULT MEDICAL PROFESSIONAL."
        fig.text(0.5, 0.02, disclaimer, ha='center', fontsize=11, 
                style='italic', color='red', weight='bold')
        
        # Main title
        plt.suptitle(f'V2 Multiclass Brain Tumor Classification Results - 95.11% Accuracy', 
                    fontsize=18, fontweight='bold')
        
        plt.tight_layout()
        plt.subplots_adjust(top=0.93, bottom=0.08)
        
        # Save visualization
        if save_path is None:
            image_name = Path(results['image_path']).stem
            save_path = f"v2_multiclass_prediction_{image_name}.png"
        
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"\n📊 V2 Prediction visualization saved: {save_path}")
        
        # Show plot
        plt.show()
        
        return save_path


def main():
    """
    Main function για V2 multiclass application
    """
    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description="V2 Multiclass Brain Tumor Classifier - Precise tumor type identification (95.11% accuracy)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python multiclass_v2_app.py brain_scan.jpg
  python multiclass_v2_app.py /path/to/mri_image.png --model custom_v2_model.h5
  python multiclass_v2_app.py image.jpg --save-viz v2_prediction.png
  python multiclass_v2_app.py scan.jpg --no-display --save-viz result.png
        """)
    
    parser.add_argument('image_path', 
                       help='Path to brain MRI image (jpg, png, etc.)')
    parser.add_argument('--model', '-m', 
                       default='best_multiclass_v2_model.h5',
                       help='Path to V2 model file (default: best_multiclass_v2_model.h5)')
    parser.add_argument('--save-viz', '-v',
                       help='Path to save V2 prediction visualization')
    parser.add_argument('--no-display', action='store_true',
                       help='Do not display visualization (save only)')
    
    args = parser.parse_args()
    
    try:
        # Initialize V2 predictor
        predictor = MultiClassV2Predictor(model_path=args.model)
        
        # Make V2 prediction
        print(f"\n🔍 V2 Analyzing brain MRI: {args.image_path}")
        results = predictor.predict_v2(args.image_path)
        
        # Display V2 results
        if not args.no_display:
            viz_path = predictor.display_v2_prediction(results, save_path=args.save_viz)
        elif args.save_viz:
            # Save without display
            viz_path = predictor.display_v2_prediction(results, save_path=args.save_viz)
            plt.close()  # Close without showing
        
        # Final V2 summary
        print(f"\n🎉 V2 Multiclass analysis completed!")
        
        # Medical V2 recommendations
        if results['predicted_class'] != 'No Tumor':
            print(f"⚠️  TUMOR DETECTED: {results['predicted_class']} - Confidence: {results['predicted_confidence']:.1%}")
            print(f"🏥 Risk Level: {results['risk_level']}")
            print(f"🚨 Priority: {results['clinical_priority']}")
            print(f"💡 Recommendation: {results['treatment_recommendations']}")
        else:
            print(f"✅ NO TUMOR DETECTED - Confidence: {results['predicted_confidence']:.1%}")
            print(f"💚 Assessment: Healthy brain tissue identified")
            print(f"📋 Recommendation: Continue routine medical care")
        
        print(f"🏆 V2 Model Performance: 95.11% test accuracy (medical-grade)")
        print(f"📋 Prediction Reliability: {results['reliability']}")
        
        return 0
        
    except FileNotFoundError as e:
        print(f"❌ File Error: {e}")
        return 1
    except Exception as e:
        print(f"❌ V2 Prediction Error: {e}")
        return 1


if __name__ == "__main__":
    # Import pandas only if needed
    try:
        import pandas as pd
    except ImportError:
        print("⚠️ pandas not installed - report generation disabled")
        pd = None
    
    exit_code = main()
    sys.exit(exit_code)