#!/usr/bin/env python3
"""
Configuration file for EfficientNet-B3 Dual Branch System
Contains all hyperparameters, paths, and model settings
"""

import os
from datetime import datetime

# Data Configuration
DATA_CONFIG = {
    'IMAGE_SIZE': (224, 224, 3),
    'BATCH_SIZE': 32,
    'VALIDATION_SPLIT': 0.2,
    'TEST_SPLIT': 0.1,
    'NUM_CLASSES': 4,
    'CLASS_NAMES': ['glioma', 'meningioma', 'pituitary', 'no_tumor'],
    'CLASS_NAMES_DISPLAY': ['Glioma', 'Meningioma', 'Pituitary', 'No Tumor']
}

# Model Architecture Configuration
MODEL_CONFIG = {
    # Custom CNN Branch Configuration
    'CUSTOM_CNN': {
        'FILTERS': [32, 64, 128, 256],
        'KERNEL_SIZE': (3, 3),
        'ACTIVATION': 'relu',
        'PADDING': 'same',
        'POOL_SIZE': (2, 2),
        'DROPOUT_RATES': [0.25, 0.25, 0.30, 0.30],
        'DENSE_UNITS': [256, 128],
        'DENSE_DROPOUT': [0.5, 0.4],
        'USE_BATCH_NORM': True
    },
    
    # EfficientNet-B3 Branch Configuration
    'EFFICIENTNET_B3': {
        'WEIGHTS': 'imagenet',
        'INCLUDE_TOP': False,
        'POOLING': 'avg',
        'FREEZE_LAYERS': True,
        'UNFREEZE_FROM_LAYER': -30,  # More layers για B3 (larger network)
        'FEATURE_DIM': 384,  # Smaller reduction για B3 (was 512)
        'DROPOUT_RATE': 0.4  # Lower dropout για better model
    },
    
    # Feature Fusion Configuration
    'FUSION': {
        'STRATEGY': 'attention',  # CHANGED: Use attention fusion για better performance
        # Options: 'concatenate', 'add', 'attention', 'gated', 'dense', 'weighted_average'
        'FUSION_DROPOUT': 0.4,  # Lower dropout
        'FUSION_UNITS': [256, 128],
        'FUSION_BATCH_NORM': True
    },
    
    # Final Classification Head
    'CLASSIFIER': {
        'HIDDEN_UNITS': [128],
        'DROPOUT_RATE': 0.3,  # Lower dropout για better performance
        'ACTIVATION': 'relu',
        'FINAL_ACTIVATION': 'softmax',
        'USE_BATCH_NORM': True
    }
}

# Training Configuration - Optimized για EfficientNet-B3
TRAINING_CONFIG = {
    # Phase 1: Frozen EfficientNet training
    'PHASE1': {
        'EPOCHS': 30,  # More epochs για better convergence
        'LEARNING_RATE': 0.005,  # Moderate LR για EfficientNet
        'OPTIMIZER': 'adam',
        'LOSS': 'categorical_crossentropy',
        'METRICS': ['accuracy', 'precision', 'recall'],
        'FREEZE_EFFICIENTNET': True
    },
    
    # Phase 2: EfficientNet fine-tuning
    'PHASE2': {
        'EPOCHS': 30,
        'LEARNING_RATE': 0.0005,  # Lower για fine-tuning
        'OPTIMIZER': 'adam',
        'LOSS': 'categorical_crossentropy',
        'METRICS': ['accuracy', 'precision', 'recall'],
        'FREEZE_EFFICIENTNET': False
    },
    
    # Phase 3: Full model optimization
    'PHASE3': {
        'EPOCHS': 25,
        'LEARNING_RATE': 0.00005,  # Very low για final polish
        'OPTIMIZER': 'adam',
        'LOSS': 'categorical_crossentropy',
        'METRICS': ['accuracy', 'precision', 'recall'],
        'FREEZE_EFFICIENTNET': False
    }
}

# Callbacks Configuration
CALLBACKS_CONFIG = {
    'EARLY_STOPPING': {
        'MONITOR': 'val_accuracy',
        'PATIENCE': 12,  # More patience για EfficientNet
        'RESTORE_BEST_WEIGHTS': True,
        'MODE': 'max',
        'VERBOSE': 1
    },
    
    'REDUCE_LR': {
        'MONITOR': 'val_loss',
        'FACTOR': 0.7,  # Less aggressive LR reduction
        'PATIENCE': 6,
        'MIN_LR': 1e-8,
        'MODE': 'min',
        'VERBOSE': 1
    },
    
    'MODEL_CHECKPOINT': {
        'MONITOR': 'val_accuracy',
        'SAVE_BEST_ONLY': True,
        'SAVE_WEIGHTS_ONLY': False,
        'MODE': 'max',
        'VERBOSE': 1
    }
}

# Data Augmentation Configuration
AUGMENTATION_CONFIG = {
    'ENABLED': True,
    'ROTATION_RANGE': 12,  # Less aggressive για EfficientNet
    'WIDTH_SHIFT_RANGE': 0.08,
    'HEIGHT_SHIFT_RANGE': 0.08,
    'ZOOM_RANGE': 0.08,
    'HORIZONTAL_FLIP': True,
    'VERTICAL_FLIP': False,
    'FILL_MODE': 'nearest',
    'BRIGHTNESS_RANGE': [0.92, 1.08],  # Tighter range
    'VALIDATION_SPLIT': 0.2
}

# Paths Configuration
def get_paths_config(base_data_path):
    """Generate paths configuration based on base data path"""
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    experiment_dir = f"efnet_dual_system_experiment_{timestamp}"
    
    return {
        'BASE_DATA_PATH': base_data_path,
        'EXPERIMENT_DIR': experiment_dir,
        'MODEL_SAVE_PATH': os.path.join(experiment_dir, 'models'),
        'LOGS_PATH': os.path.join(experiment_dir, 'logs'),
        'RESULTS_PATH': os.path.join(experiment_dir, 'results'),
        'PLOTS_PATH': os.path.join(experiment_dir, 'plots'),
        
        # Model file names
        'PHASE1_MODEL': 'efnet_dual_phase1.h5',
        'PHASE2_MODEL': 'efnet_dual_phase2.h5',
        'PHASE3_MODEL': 'efnet_dual_phase3.h5',
        'BEST_MODEL': 'best_efnet_dual_system_model.h5',
        
        # Results file names
        'TRAINING_HISTORY': 'efnet_training_history.json',
        'EVALUATION_RESULTS': 'efnet_evaluation_results.json',
        'CONFUSION_MATRIX': 'efnet_confusion_matrix.png',
        'TRAINING_PLOTS': 'efnet_training_plots.png',
        'PERFORMANCE_COMPARISON': 'efnet_performance_comparison.json'
    }

# Evaluation Configuration
EVALUATION_CONFIG = {
    'METRICS': [
        'accuracy', 'precision', 'recall', 'f1_score',
        'confusion_matrix', 'classification_report'
    ],
    'SAVE_PREDICTIONS': True,
    'SAVE_PROBABILITIES': True,
    'PLOT_CONFUSION_MATRIX': True,
    'PLOT_ROC_CURVES': True,
    'PLOT_PRECISION_RECALL_CURVES': True
}

# Comparison Models
COMPARISON_MODELS = {
    'BINARY_MODEL': {
        'NAME': 'Binary Classification',
        'ACCURACY': 0.9871,
        'PATH': None
    },
    'MULTICLASS_4': {
        'NAME': '4-Class Multiclass',
        'ACCURACY': 0.9872,
        'PATH': None
    },
    'MULTICLASS_3': {
        'NAME': '3-Class Multiclass', 
        'ACCURACY': 0.9781,
        'PATH': None
    },
    'ENSEMBLE_1VS1': {
        'NAME': '1-vs-1 Ensemble',
        'ACCURACY': 0.9642,
        'PATH': None
    },
    'DUAL_RESNET50': {
        'NAME': 'Dual ResNet-50',
        'ACCURACY': 0.8545,  # Previous result
        'PATH': None
    }
}

# Logging Configuration
LOGGING_CONFIG = {
    'LEVEL': 'INFO',
    'FORMAT': '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    'SAVE_TO_FILE': True,
    'LOG_FILE': 'efnet_dual_system_training.log'
}

# Hardware Configuration
HARDWARE_CONFIG = {
    'GPU_MEMORY_GROWTH': True,
    'MIXED_PRECISION': False,
    'PARALLEL_DATA_LOADING': True,
    'NUM_WORKERS': 4
}

# Reproducibility Configuration
REPRODUCIBILITY_CONFIG = {
    'RANDOM_SEED': 42,
    'SET_TENSORFLOW_DETERMINISTIC': True,
    'SET_NUMPY_SEED': True,
    'SET_PYTHON_HASH_SEED': True
}

# Validation Configuration
VALIDATION_CONFIG = {
    'CROSS_VALIDATION': False,
    'CV_FOLDS': 5,
    'STRATIFIED': True,
    'SHUFFLE': True
}

# Advanced Training Techniques
ADVANCED_CONFIG = {
    'LABEL_SMOOTHING': 0.05,  # Light label smoothing για EfficientNet
    'MIXUP': False,
    'CUTMIX': False,
    'PROGRESSIVE_RESIZING': False,
    'GRADIENT_CLIPPING': False,
    'COSINE_ANNEALING': False
}

# Export main configurations
def get_config():
    """Return all configuration dictionaries"""
    return {
        'data': DATA_CONFIG,
        'model': MODEL_CONFIG,
        'training': TRAINING_CONFIG,
        'callbacks': CALLBACKS_CONFIG,
        'augmentation': AUGMENTATION_CONFIG,
        'evaluation': EVALUATION_CONFIG,
        'comparison': COMPARISON_MODELS,
        'logging': LOGGING_CONFIG,
        'hardware': HARDWARE_CONFIG,
        'reproducibility': REPRODUCIBILITY_CONFIG,
        'validation': VALIDATION_CONFIG,
        'advanced': ADVANCED_CONFIG
    }

# Default configuration
DEFAULT_CONFIG = get_config()

if __name__ == "__main__":
    print("EfficientNet-B3 Dual System Configuration Loaded")
    print(f"Model: Custom CNN + EfficientNet-B3 Dual Branch")
    print(f"Image Size: {DATA_CONFIG['IMAGE_SIZE']}")
    print(f"Batch Size: {DATA_CONFIG['BATCH_SIZE']}")
    print(f"Number of Classes: {DATA_CONFIG['NUM_CLASSES']}")
    print(f"Training Phases: 3 (Frozen EfficientNet -> Fine-tuning -> Full Optimization)")
    print("Optimizations:")
    print(f"  - Moderate learning rates optimized για EfficientNet")
    print(f"  - Extended unfreezing (-30 layers)")
    print(f"  - Reduced dropout rates")
    print(f"  - Label smoothing enabled")