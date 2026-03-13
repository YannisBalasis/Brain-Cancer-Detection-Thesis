#!/usr/bin/env python3
"""
Dual Branch System Architecture
Combines Custom CNN (medical-optimized) with ResNet-50 (pre-trained) for brain tumor classification
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models
from tensorflow.keras.applications import ResNet50
import numpy as np
from config_dual import MODEL_CONFIG, DATA_CONFIG
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DualBranchSystem:
    """
    Dual Branch Architecture combining Custom CNN and ResNet-50
    """
    
    def __init__(self, config=None):
        """
        Initialize Dual Branch System
        
        Args:
            config: Configuration dictionary, defaults to MODEL_CONFIG
        """
        self.config = config if config else MODEL_CONFIG
        self.input_shape = DATA_CONFIG['IMAGE_SIZE']
        self.num_classes = DATA_CONFIG['NUM_CLASSES']
        
        # Model components
        self.custom_cnn_branch = None
        self.resnet_branch = None
        self.fusion_layer = None
        self.classifier = None
        self.model = None
        
        logger.info("Dual Branch System initialized")
        logger.info(f"Input shape: {self.input_shape}")
        logger.info(f"Number of classes: {self.num_classes}")
    
    def create_custom_cnn_branch(self, inputs):
        """
        Create the custom CNN branch (proven 4-class architecture)
        
        Args:
            inputs: Input tensor
            
        Returns:
            Custom CNN branch output tensor
        """
        x = inputs
        custom_config = self.config['CUSTOM_CNN']
        filters = custom_config['FILTERS']
        dropout_rates = custom_config['DROPOUT_RATES']
        
        # Block 1: Basic feature detection
        x = layers.Conv2D(filters[0], custom_config['KERNEL_SIZE'], 
                         activation=custom_config['ACTIVATION'], 
                         padding=custom_config['PADDING'],
                         name='custom_conv1_1')(x)
        if custom_config['USE_BATCH_NORM']:
            x = layers.BatchNormalization(name='custom_bn1_1')(x)
            
        x = layers.Conv2D(filters[0], custom_config['KERNEL_SIZE'], 
                         activation=custom_config['ACTIVATION'], 
                         padding=custom_config['PADDING'],
                         name='custom_conv1_2')(x)
        if custom_config['USE_BATCH_NORM']:
            x = layers.BatchNormalization(name='custom_bn1_2')(x)
            
        x = layers.MaxPooling2D(custom_config['POOL_SIZE'], name='custom_pool1')(x)
        x = layers.Dropout(dropout_rates[0], name='custom_dropout1')(x)
        
        # Block 2: Medical patterns
        x = layers.Conv2D(filters[1], custom_config['KERNEL_SIZE'], 
                         activation=custom_config['ACTIVATION'], 
                         padding=custom_config['PADDING'],
                         name='custom_conv2_1')(x)
        if custom_config['USE_BATCH_NORM']:
            x = layers.BatchNormalization(name='custom_bn2_1')(x)
            
        x = layers.Conv2D(filters[1], custom_config['KERNEL_SIZE'], 
                         activation=custom_config['ACTIVATION'], 
                         padding=custom_config['PADDING'],
                         name='custom_conv2_2')(x)
        if custom_config['USE_BATCH_NORM']:
            x = layers.BatchNormalization(name='custom_bn2_2')(x)
            
        x = layers.MaxPooling2D(custom_config['POOL_SIZE'], name='custom_pool2')(x)
        x = layers.Dropout(dropout_rates[1], name='custom_dropout2')(x)
        
        # Block 3: Tumor features
        x = layers.Conv2D(filters[2], custom_config['KERNEL_SIZE'], 
                         activation=custom_config['ACTIVATION'], 
                         padding=custom_config['PADDING'],
                         name='custom_conv3_1')(x)
        if custom_config['USE_BATCH_NORM']:
            x = layers.BatchNormalization(name='custom_bn3_1')(x)
            
        x = layers.Conv2D(filters[2], custom_config['KERNEL_SIZE'], 
                         activation=custom_config['ACTIVATION'], 
                         padding=custom_config['PADDING'],
                         name='custom_conv3_2')(x)
        if custom_config['USE_BATCH_NORM']:
            x = layers.BatchNormalization(name='custom_bn3_2')(x)
            
        x = layers.MaxPooling2D(custom_config['POOL_SIZE'], name='custom_pool3')(x)
        x = layers.Dropout(dropout_rates[2], name='custom_dropout3')(x)
        
        # Block 4: Abstract features
        x = layers.Conv2D(filters[3], custom_config['KERNEL_SIZE'], 
                         activation=custom_config['ACTIVATION'], 
                         padding=custom_config['PADDING'],
                         name='custom_conv4_1')(x)
        if custom_config['USE_BATCH_NORM']:
            x = layers.BatchNormalization(name='custom_bn4_1')(x)
            
        x = layers.Conv2D(filters[3], custom_config['KERNEL_SIZE'], 
                         activation=custom_config['ACTIVATION'], 
                         padding=custom_config['PADDING'],
                         name='custom_conv4_2')(x)
        if custom_config['USE_BATCH_NORM']:
            x = layers.BatchNormalization(name='custom_bn4_2')(x)
            
        x = layers.MaxPooling2D(custom_config['POOL_SIZE'], name='custom_pool4')(x)
        x = layers.Dropout(dropout_rates[3], name='custom_dropout4')(x)
        
        # Global Average Pooling
        x = layers.GlobalAveragePooling2D(name='custom_gap')(x)
        
        # Dense layers
        dense_units = custom_config['DENSE_UNITS']
        dense_dropout = custom_config['DENSE_DROPOUT']
        
        x = layers.Dense(dense_units[0], activation=custom_config['ACTIVATION'], 
                        name='custom_dense1')(x)
        if custom_config['USE_BATCH_NORM']:
            x = layers.BatchNormalization(name='custom_bn_dense1')(x)
        x = layers.Dropout(dense_dropout[0], name='custom_dropout_dense1')(x)
        
        x = layers.Dense(dense_units[1], activation=custom_config['ACTIVATION'], 
                        name='custom_dense2')(x)
        if custom_config['USE_BATCH_NORM']:
            x = layers.BatchNormalization(name='custom_bn_dense2')(x)
        x = layers.Dropout(dense_dropout[1], name='custom_dropout_dense2')(x)
        
        return x
    
    def create_resnet_branch(self, inputs):
        """
        Create the ResNet-50 branch
        
        Args:
            inputs: Input tensor
            
        Returns:
            ResNet-50 branch output tensor
        """
        resnet_config = self.config['RESNET50']
        
        # Load pre-trained ResNet-50
        base_resnet = ResNet50(
            weights=resnet_config['WEIGHTS'],
            include_top=resnet_config['INCLUDE_TOP'],
            input_shape=self.input_shape,
            pooling=resnet_config['POOLING']
        )
        
        # Initially freeze ResNet layers if specified
        if resnet_config['FREEZE_LAYERS']:
            for layer in base_resnet.layers:
                layer.trainable = False
                
        # Get ResNet features
        x = base_resnet(inputs)
        
        # Dimension reduction to match custom CNN output
        x = layers.Dense(resnet_config['FEATURE_DIM'], 
                        activation='relu',
                        name='resnet_dense_reduction')(x)
        x = layers.BatchNormalization(name='resnet_bn_reduction')(x)
        x = layers.Dropout(resnet_config['DROPOUT_RATE'], 
                          name='resnet_dropout_reduction')(x)
        
        # Additional processing layer
        x = layers.Dense(128, activation='relu', name='resnet_dense_final')(x)
        x = layers.BatchNormalization(name='resnet_bn_final')(x)
        
        # Store base ResNet for later fine-tuning
        self.base_resnet = base_resnet
        
        return x
    
    def create_fusion_layer(self, custom_features, resnet_features):
        """
        Create fusion layer to combine features from both branches
        
        Args:
            custom_features: Features from custom CNN branch
            resnet_features: Features from ResNet-50 branch
            
        Returns:
            Fused features tensor
        """
        fusion_config = self.config['FUSION']
        strategy = fusion_config['STRATEGY']
        
        if strategy == 'concatenate':
            # Simple concatenation
            fused = layers.Concatenate(name='fusion_concatenate')([custom_features, resnet_features])
            
        elif strategy == 'add':
            # Element-wise addition
            fused = layers.Add(name='fusion_add')([custom_features, resnet_features])
            
        elif strategy == 'attention':
            # Attention-based fusion
            attention_custom = layers.Dense(128, activation='sigmoid', 
                                          name='attention_custom')(custom_features)
            attention_resnet = layers.Dense(128, activation='sigmoid', 
                                          name='attention_resnet')(resnet_features)
            
            weighted_custom = layers.Multiply(name='weighted_custom')([custom_features, attention_custom])
            weighted_resnet = layers.Multiply(name='weighted_resnet')([resnet_features, attention_resnet])
            
            fused = layers.Add(name='fusion_attention')([weighted_custom, weighted_resnet])
            
        elif strategy == 'dense':
            # Dense fusion layer
            concatenated = layers.Concatenate(name='fusion_concat_temp')([custom_features, resnet_features])
            fused = layers.Dense(128, activation='relu', name='fusion_dense')(concatenated)
            
        else:
            raise ValueError(f"Unknown fusion strategy: {strategy}")
        
        # Post-fusion processing
        if fusion_config['FUSION_BATCH_NORM']:
            fused = layers.BatchNormalization(name='fusion_bn')(fused)
            
        fused = layers.Dropout(fusion_config['FUSION_DROPOUT'], name='fusion_dropout')(fused)
        
        # Additional fusion layers
        for i, units in enumerate(fusion_config['FUSION_UNITS']):
            fused = layers.Dense(units, activation='relu', 
                               name=f'fusion_dense_{i+1}')(fused)
            if fusion_config['FUSION_BATCH_NORM']:
                fused = layers.BatchNormalization(name=f'fusion_bn_{i+1}')(fused)
            fused = layers.Dropout(fusion_config['FUSION_DROPOUT'], 
                                 name=f'fusion_dropout_{i+1}')(fused)
        
        return fused
    
    def create_classifier(self, fused_features):
        """
        Create final classification head
        
        Args:
            fused_features: Fused features from both branches
            
        Returns:
            Classification output tensor
        """
        classifier_config = self.config['CLASSIFIER']
        x = fused_features
        
        # Hidden layers
        for i, units in enumerate(classifier_config['HIDDEN_UNITS']):
            x = layers.Dense(units, activation=classifier_config['ACTIVATION'],
                           name=f'classifier_hidden_{i+1}')(x)
            if classifier_config['USE_BATCH_NORM']:
                x = layers.BatchNormalization(name=f'classifier_bn_{i+1}')(x)
            x = layers.Dropout(classifier_config['DROPOUT_RATE'],
                             name=f'classifier_dropout_{i+1}')(x)
        
        # Final classification layer
        outputs = layers.Dense(self.num_classes, 
                             activation=classifier_config['FINAL_ACTIVATION'],
                             name='classifier_output')(x)
        
        return outputs
    
    def build_model(self):
        """
        Build the complete dual branch model
        
        Returns:
            Complete Keras model
        """
        logger.info("Building Dual Branch System model...")
        
        # Input layer
        inputs = layers.Input(shape=self.input_shape, name='input_layer')
        
        # Create both branches
        logger.info("Creating Custom CNN branch...")
        custom_features = self.create_custom_cnn_branch(inputs)
        
        logger.info("Creating ResNet-50 branch...")
        resnet_features = self.create_resnet_branch(inputs)
        
        # Fusion layer
        logger.info("Creating fusion layer...")
        fused_features = self.create_fusion_layer(custom_features, resnet_features)
        
        # Classification head
        logger.info("Creating classifier...")
        outputs = self.create_classifier(fused_features)
        
        # Create model
        self.model = models.Model(inputs=inputs, outputs=outputs, name='dual_branch_system')
        
        logger.info("Dual Branch System model built successfully")
        logger.info(f"Total parameters: {self.model.count_params():,}")
        
        return self.model
    
    def unfreeze_resnet_layers(self, from_layer=-10):
        """
        Unfreeze ResNet layers for fine-tuning
        
        Args:
            from_layer: Number of layers from the end to unfreeze (negative index)
        """
        if hasattr(self, 'base_resnet'):
            logger.info(f"Unfreezing ResNet layers from layer {from_layer}")
            
            # Unfreeze specified layers
            for layer in self.base_resnet.layers[from_layer:]:
                layer.trainable = True
                
            trainable_params = sum([layer.count_params() for layer in self.base_resnet.layers if layer.trainable])
            logger.info(f"ResNet trainable parameters: {trainable_params:,}")
        else:
            logger.warning("Base ResNet not found. Cannot unfreeze layers.")
    
    def freeze_resnet_layers(self):
        """
        Freeze all ResNet layers
        """
        if hasattr(self, 'base_resnet'):
            logger.info("Freezing all ResNet layers")
            
            for layer in self.base_resnet.layers:
                layer.trainable = False
                
            logger.info("All ResNet layers frozen")
        else:
            logger.warning("Base ResNet not found. Cannot freeze layers.")
    
    def get_model_summary(self):
        """
        Get detailed model summary
        
        Returns:
            Model summary information
        """
        if self.model is None:
            logger.error("Model not built yet. Call build_model() first.")
            return None
        
        summary_info = {
            'total_params': self.model.count_params(),
            'trainable_params': sum([layer.count_params() for layer in self.model.layers if layer.trainable]),
            'non_trainable_params': sum([layer.count_params() for layer in self.model.layers if not layer.trainable]),
            'input_shape': self.input_shape,
            'output_classes': self.num_classes,
            'fusion_strategy': self.config['FUSION']['STRATEGY']
        }
        
        if hasattr(self, 'base_resnet'):
            resnet_trainable = sum([layer.count_params() for layer in self.base_resnet.layers if layer.trainable])
            summary_info['resnet_trainable_params'] = resnet_trainable
        
        return summary_info
    
    def compile_model(self, phase='phase1'):
        """
        Compile the model with appropriate settings for training phase
        
        Args:
            phase: Training phase ('phase1', 'phase2', 'phase3')
        """
        from config_dual import TRAINING_CONFIG
        
        if self.model is None:
            logger.error("Model not built yet. Call build_model() first.")
            return
        
        phase_config = TRAINING_CONFIG[phase.upper()]
        
        # Configure optimizer
        if phase_config['OPTIMIZER'] == 'adam':
            optimizer = keras.optimizers.Adam(learning_rate=phase_config['LEARNING_RATE'])
        else:
            optimizer = phase_config['OPTIMIZER']
        
        # Compile model
        self.model.compile(
            optimizer=optimizer,
            loss=phase_config['LOSS'],
            metrics=phase_config['METRICS']
        )
        
        logger.info(f"Model compiled for {phase} with LR: {phase_config['LEARNING_RATE']}")

def create_dual_system_model():
    """
    Convenience function to create and build dual system model
    
    Returns:
        Built and ready dual system model
    """
    dual_system = DualBranchSystem()
    model = dual_system.build_model()
    return dual_system, model

if __name__ == "__main__":
    # Test model creation
    print("Testing Dual Branch System...")
    
    dual_system, model = create_dual_system_model()
    
    # Print model summary
    model.summary()
    
    # Get detailed summary
    summary_info = dual_system.get_model_summary()
    print("\nDetailed Model Information:")
    for key, value in summary_info.items():
        print(f"{key}: {value:,}" if isinstance(value, int) else f"{key}: {value}")
    
    print("\nDual Branch System created successfully!")