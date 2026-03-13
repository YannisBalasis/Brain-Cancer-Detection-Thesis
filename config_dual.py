#!/usr/bin/env python3
"""
Configuration file for Dual Branch System (Custom CNN + ResNet-50)
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
    
    # ResNet-50 Branch Configuration
    'RESNET50': {
        'WEIGHTS': 'imagenet',
        'INCLUDE_TOP': False,
        'POOLING': 'avg',
        'FREEZE_LAYERS': True,
        'UNFREEZE_FROM_LAYER': -20,  # Unfreeze last 10 layers during fine-tuning
        'FEATURE_DIM': 512,  # Dimension reduction from 2048 to this
        'DROPOUT_RATE': 0.5
    },
    
    # Feature Fusion Configuration
    'FUSION': {
        'STRATEGY': 'concatenate',  # Options: 'concatenate', 'add', 'attention', 'dense'
        'FUSION_DROPOUT': 0.5,
        'FUSION_UNITS': [256, 128],  # Dense layers after fusion
        'FUSION_BATCH_NORM': True
    },
    
    # Final Classification Head
    'CLASSIFIER': {
        'HIDDEN_UNITS': [128],
        'DROPOUT_RATE': 0.4,
        'ACTIVATION': 'relu',
        'FINAL_ACTIVATION': 'softmax',
        'USE_BATCH_NORM': True
    }
}

# Training Configuration
TRAINING_CONFIG = {
    # Phase 1: Frozen ResNet training
    'PHASE1': {
        'EPOCHS': 25,
        'LEARNING_RATE': 0.01,
        'OPTIMIZER': 'adam',
        'LOSS': 'categorical_crossentropy',
        'METRICS': ['accuracy', 'precision', 'recall'],
        'FREEZE_RESNET': True
    },
    
    # Phase 2: ResNet fine-tuning
    'PHASE2': {
        'EPOCHS': 25,
        'LEARNING_RATE': 0.001,  # Lower LR for fine-tuning
        'OPTIMIZER': 'adam',
        'LOSS': 'categorical_crossentropy',
        'METRICS': ['accuracy', 'precision', 'recall'],
        'FREEZE_RESNET': False
    },
    
    # Phase 3: Full model optimization (optional)
    'PHASE3': {
        'EPOCHS': 20,
        'LEARNING_RATE': 0.0001,  # Very low LR for final optimization
        'OPTIMIZER': 'adam',
        'LOSS': 'categorical_crossentropy',
        'METRICS': ['accuracy', 'precision', 'recall'],
        'FREEZE_RESNET': False
    }
}

# Callbacks Configuration
CALLBACKS_CONFIG = {
    'EARLY_STOPPING': {
        'MONITOR': 'val_accuracy',
        'PATIENCE': 15,
        'RESTORE_BEST_WEIGHTS': True,
        'MODE': 'max',
        'VERBOSE': 1
    },
    
    'REDUCE_LR': {
        'MONITOR': 'val_loss',
        'FACTOR': 0.5,
        'PATIENCE': 8,
        'MIN_LR': 1e-7,
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
    'ROTATION_RANGE': 15,
    'WIDTH_SHIFT_RANGE': 0.1,
    'HEIGHT_SHIFT_RANGE': 0.1,
    'ZOOM_RANGE': 0.1,
    'HORIZONTAL_FLIP': True,
    'VERTICAL_FLIP': False,  # Medical images typically don't flip vertically
    'FILL_MODE': 'nearest',
    'BRIGHTNESS_RANGE': [0.9, 1.1],
    'VALIDATION_SPLIT': 0.2
}

# Paths Configuration
def get_paths_config(base_data_path):
    """Generate paths configuration based on base data path"""
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    experiment_dir = f"dual_system_experiment_{timestamp}"
    
    return {
        'BASE_DATA_PATH': base_data_path,
        'EXPERIMENT_DIR': experiment_dir,
        'MODEL_SAVE_PATH': os.path.join(experiment_dir, 'models'),
        'LOGS_PATH': os.path.join(experiment_dir, 'logs'),
        'RESULTS_PATH': os.path.join(experiment_dir, 'results'),
        'PLOTS_PATH': os.path.join(experiment_dir, 'plots'),
        
        # Model file names
        'PHASE1_MODEL': 'dual_system_phase1.h5',
        'PHASE2_MODEL': 'dual_system_phase2.h5',
        'PHASE3_MODEL': 'dual_system_phase3.h5',
        'BEST_MODEL': 'best_dual_system_model.h5',
        
        # Results file names
        'TRAINING_HISTORY': 'training_history.json',
        'EVALUATION_RESULTS': 'evaluation_results.json',
        'CONFUSION_MATRIX': 'confusion_matrix.png',
        'TRAINING_PLOTS': 'training_plots.png',
        'PERFORMANCE_COMPARISON': 'performance_comparison.json'
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

# Comparison Models (for performance comparison)
COMPARISON_MODELS = {
    'BINARY_MODEL': {
        'NAME': 'Binary Classification',
        'ACCURACY': 0.9871,  # Update with actual performance
        'PATH': None  # Will be updated during comparison
    },
    'MULTICLASS_4': {
        'NAME': '4-Class Multiclass',
        'ACCURACY': 0.9872,  # Current champion
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
    }
}

# Logging Configuration
LOGGING_CONFIG = {
    'LEVEL': 'INFO',
    'FORMAT': '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    'SAVE_TO_FILE': True,
    'LOG_FILE': 'dual_system_training.log'
}

# Hardware Configuration
HARDWARE_CONFIG = {
    'GPU_MEMORY_GROWTH': True,
    'MIXED_PRECISION': False,  # Enable if using compatible GPU
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
    'CROSS_VALIDATION': False,  # Enable for k-fold CV
    'CV_FOLDS': 5,
    'STRATIFIED': True,
    'SHUFFLE': True
}

# Advanced Training Techniques (Optional)
ADVANCED_CONFIG = {
    'LABEL_SMOOTHING': 0.0,  # Set to 0.1 for label smoothing
    'MIXUP': False,  # Enable mixup data augmentation
    'CUTMIX': False,  # Enable cutmix data augmentation
    'PROGRESSIVE_RESIZING': False,  # Start with smaller images
    'GRADIENT_CLIPPING': False,  # Enable gradient clipping
    'COSINE_ANNEALING': False  # Cosine annealing learning rate schedule
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

# Default configuration for easy import
DEFAULT_CONFIG = get_config()

if __name__ == "__main__":
    print("Dual System Configuration Loaded Successfully")
    print(f"Model: Custom CNN + ResNet-50 Dual Branch")
    print(f"Image Size: {DATA_CONFIG['IMAGE_SIZE']}")
    print(f"Batch Size: {DATA_CONFIG['BATCH_SIZE']}")
    print(f"Number of Classes: {DATA_CONFIG['NUM_CLASSES']}")
    print(f"Training Phases: 3 (Frozen ResNet -> Fine-tuning -> Full Optimization)")