#!/usr/bin/env python3
"""
EfficientNet-B3 Dual Branch System Architecture
Combines Custom CNN (medical-optimized) with EfficientNet-B3 for brain tumor classification
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models
from tensorflow.keras.applications import EfficientNetB3
import numpy as np
from efnet_dual_system_config import MODEL_CONFIG, DATA_CONFIG
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class EfficientNetDualBranchSystem:
    """
    Dual Branch Architecture combining Custom CNN and EfficientNet-B3
    """
    
    def __init__(self, config=None):
        """
        Initialize EfficientNet Dual Branch System
        
        Args:
            config: Configuration dictionary, defaults to MODEL_CONFIG
        """
        self.config = config if config else MODEL_CONFIG
        self.input_shape = DATA_CONFIG['IMAGE_SIZE']
        self.num_classes = DATA_CONFIG['NUM_CLASSES']
        
        # Model components
        self.custom_cnn_branch = None
        self.efficientnet_branch = None
        self.fusion_layer = None
        self.classifier = None
        self.model = None
        
        logger.info("EfficientNet-B3 Dual Branch System initialized")
        logger.info(f"Input shape: {self.input_shape}")
        logger.info(f"Number of classes: {self.num_classes}")
    
    def create_custom_cnn_branch(self, inputs):
        """
        Create the custom CNN branch (medical-optimized architecture)
        
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
    
    def create_efficientnet_branch(self, inputs):
        """
        Create the EfficientNet-B3 branch
        
        Args:
            inputs: Input tensor
            
        Returns:
            EfficientNet-B3 branch output tensor
        """
        efnet_config = self.config['EFFICIENTNET_B3']
        
        # Load pre-trained EfficientNet-B3
        base_efficientnet = EfficientNetB3(
            weights=efnet_config['WEIGHTS'],
            include_top=efnet_config['INCLUDE_TOP'],
            input_shape=self.input_shape,
            pooling=efnet_config['POOLING']
        )
        
        # Initially freeze EfficientNet layers if specified
        if efnet_config['FREEZE_LAYERS']:
            for layer in base_efficientnet.layers:
                layer.trainable = False
                
        # Get EfficientNet features
        x = base_efficientnet(inputs)
        
        # Dimension reduction to match custom CNN output size
        x = layers.Dense(efnet_config['FEATURE_DIM'], 
                        activation='relu',
                        name='efnet_dense_reduction')(x)
        x = layers.BatchNormalization(name='efnet_bn_reduction')(x)
        x = layers.Dropout(efnet_config['DROPOUT_RATE'], 
                          name='efnet_dropout_reduction')(x)
        
        # Additional processing layer
        x = layers.Dense(128, activation='relu', name='efnet_dense_final')(x)
        x = layers.BatchNormalization(name='efnet_bn_final')(x)
        
        # Store base EfficientNet για later fine-tuning
        self.base_efficientnet = base_efficientnet
        
        return x
    
    def create_fusion_layer(self, custom_features, efnet_features):
        """
        Create fusion layer to combine features from both branches
        Support for multiple fusion strategies
        
        Args:
            custom_features: Features από custom CNN branch
            efnet_features: Features από EfficientNet-B3 branch
            
        Returns:
            Fused features tensor
        """
        fusion_config = self.config['FUSION']
        strategy = fusion_config['STRATEGY']
        
        if strategy == 'concatenate':
            # Simple concatenation (original approach)
            fused = layers.Concatenate(name='fusion_concatenate')([custom_features, efnet_features])
            
        elif strategy == 'add':
            # Element-wise addition (same dimension required)
            fused = layers.Add(name='fusion_add')([custom_features, efnet_features])
            
        elif strategy == 'attention':
            # Attention-based fusion (RECOMMENDED for better performance)
            logger.info("Using attention-based fusion strategy")
            
            # Learn attention weights για each branch
            attention_custom = layers.Dense(128, activation='sigmoid', 
                                          name='attention_custom')(custom_features)
            attention_efnet = layers.Dense(128, activation='sigmoid', 
                                          name='attention_efnet')(efnet_features)
            
            # Apply attention weights
            weighted_custom = layers.Multiply(name='weighted_custom')([custom_features, attention_custom])
            weighted_efnet = layers.Multiply(name='weighted_efnet')([efnet_features, attention_efnet])
            
            # Combine weighted features
            fused = layers.Add(name='fusion_attention_add')([weighted_custom, weighted_efnet])
            
        elif strategy == 'gated':
            # Gated fusion - learn dynamic weighting
            logger.info("Using gated fusion strategy")
            
            # Concatenate για gate computation
            concat_for_gate = layers.Concatenate(name='gate_concat')([custom_features, efnet_features])
            
            # Learn gate values (0-1) για each feature dimension
            gate_weights = layers.Dense(128, activation='sigmoid', 
                                      name='gate_weights')(concat_for_gate)
            
            # Apply gates: gate*custom + (1-gate)*efnet
            complement_gate = layers.Lambda(lambda x: 1.0 - x, name='complement_gate')(gate_weights)
            
            gated_custom = layers.Multiply(name='gated_custom')([custom_features, gate_weights])
            gated_efnet = layers.Multiply(name='gated_efnet')([efnet_features, complement_gate])
            
            fused = layers.Add(name='fusion_gated_add')([gated_custom, gated_efnet])
            
        elif strategy == 'dense':
            # Dense fusion layer (original implementation)
            concatenated = layers.Concatenate(name='fusion_concat_temp')([custom_features, efnet_features])
            fused = layers.Dense(128, activation='relu', name='fusion_dense')(concatenated)
            
        elif strategy == 'weighted_average':
            # Learnable weighted average
            logger.info("Using weighted average fusion strategy")
            
            # Learn scalar weights για each branch
            weight_custom = layers.Dense(1, activation='sigmoid', name='weight_custom', use_bias=False)(
                layers.GlobalAveragePooling1D()(layers.Reshape((128, 1))(custom_features))
            )
            weight_efnet = layers.Dense(1, activation='sigmoid', name='weight_efnet', use_bias=False)(
                layers.GlobalAveragePooling1D()(layers.Reshape((128, 1))(efnet_features))
            )
            
            # Normalize weights
            total_weight = layers.Add()([weight_custom, weight_efnet])
            norm_weight_custom = layers.Lambda(lambda x: x[0] / x[1])([weight_custom, total_weight])
            norm_weight_efnet = layers.Lambda(lambda x: x[0] / x[1])([weight_efnet, total_weight])
            
            # Apply weights
            weighted_custom = layers.Multiply()([custom_features, norm_weight_custom])
            weighted_efnet = layers.Multiply()([efnet_features, norm_weight_efnet])
            
            fused = layers.Add(name='fusion_weighted_add')([weighted_custom, weighted_efnet])
            
        else:
            raise ValueError(f"Unknown fusion strategy: {strategy}")
        
        # Post-fusion processing (adapt based on fusion output dimension)
        if strategy in ['concatenate', 'dense']:
            # For concatenation/dense, we have 256-dim output
            if fusion_config['FUSION_BATCH_NORM']:
                fused = layers.BatchNormalization(name='fusion_bn')(fused)
            fused = layers.Dropout(fusion_config['FUSION_DROPOUT'], name='fusion_dropout')(fused)
            
            # Additional fusion layers για concatenation
            for i, units in enumerate(fusion_config['FUSION_UNITS']):
                fused = layers.Dense(units, activation='relu', 
                                   name=f'fusion_dense_{i+1}')(fused)
                if fusion_config['FUSION_BATCH_NORM']:
                    fused = layers.BatchNormalization(name=f'fusion_bn_{i+1}')(fused)
                fused = layers.Dropout(fusion_config['FUSION_DROPOUT'], 
                                     name=f'fusion_dropout_{i+1}')(fused)
        else:
            # For attention/add/gated/weighted - we have 128-dim output
            if fusion_config['FUSION_BATCH_NORM']:
                fused = layers.BatchNormalization(name='fusion_bn')(fused)
            fused = layers.Dropout(fusion_config['FUSION_DROPOUT'], name='fusion_dropout')(fused)
            
            # Optional additional processing for non-concatenate strategies
            if len(fusion_config['FUSION_UNITS']) > 0:
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
            fused_features: Fused features από both branches
            
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
        Build the complete EfficientNet dual branch model
        
        Returns:
            Complete Keras model
        """
        logger.info("Building EfficientNet-B3 Dual Branch System model...")
        
        # Input layer
        inputs = layers.Input(shape=self.input_shape, name='input_layer')
        
        # Create both branches
        logger.info("Creating Custom CNN branch...")
        custom_features = self.create_custom_cnn_branch(inputs)
        
        logger.info("Creating EfficientNet-B3 branch...")
        efnet_features = self.create_efficientnet_branch(inputs)
        
        # Fusion layer
        logger.info("Creating fusion layer...")
        fused_features = self.create_fusion_layer(custom_features, efnet_features)
        
        # Classification head
        logger.info("Creating classifier...")
        outputs = self.create_classifier(fused_features)
        
        # Create model
        self.model = models.Model(inputs=inputs, outputs=outputs, name='efnet_dual_branch_system')
        
        logger.info("EfficientNet-B3 Dual Branch System model built successfully")
        logger.info(f"Total parameters: {self.model.count_params():,}")
        
        return self.model
    
    def unfreeze_efficientnet_layers(self, from_layer=-30):
        """
        Unfreeze EfficientNet layers για fine-tuning
        
        Args:
            from_layer: Number of layers από end to unfreeze (negative index)
        """
        if hasattr(self, 'base_efficientnet'):
            logger.info(f"Unfreezing EfficientNet layers από layer {from_layer}")
            
            # Unfreeze specified layers
            for layer in self.base_efficientnet.layers[from_layer:]:
                layer.trainable = True
                
            trainable_params = sum([layer.count_params() for layer in self.base_efficientnet.layers if layer.trainable])
            logger.info(f"EfficientNet trainable parameters: {trainable_params:,}")
        else:
            logger.warning("Base EfficientNet not found. Cannot unfreeze layers.")
    
    def freeze_efficientnet_layers(self):
        """
        Freeze all EfficientNet layers
        """
        if hasattr(self, 'base_efficientnet'):
            logger.info("Freezing all EfficientNet layers")
            
            for layer in self.base_efficientnet.layers:
                layer.trainable = False
                
            logger.info("All EfficientNet layers frozen")
        else:
            logger.warning("Base EfficientNet not found. Cannot freeze layers.")
    
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
            'fusion_strategy': self.config['FUSION']['STRATEGY'],
            'backbone': 'EfficientNet-B3'
        }
        
        if hasattr(self, 'base_efficientnet'):
            efnet_trainable = sum([layer.count_params() for layer in self.base_efficientnet.layers if layer.trainable])
            summary_info['efficientnet_trainable_params'] = efnet_trainable
        
        return summary_info
    
    def compile_model(self, phase='phase1'):
        """
        Compile the model with appropriate settings για training phase
        
        Args:
            phase: Training phase ('phase1', 'phase2', 'phase3')
        """
        from efnet_dual_system_config import TRAINING_CONFIG
        
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
        
        logger.info(f"Model compiled για {phase} with LR: {phase_config['LEARNING_RATE']}")

def create_efnet_dual_system_model():
    """
    Convenience function to create and build EfficientNet dual system model
    
    Returns:
        Built and ready EfficientNet dual system model
    """
    efnet_dual_system = EfficientNetDualBranchSystem()
    model = efnet_dual_system.build_model()
    return efnet_dual_system, model

if __name__ == "__main__":
    # Test model creation
    print("Testing EfficientNet-B3 Dual Branch System...")
    
    efnet_dual_system, model = create_efnet_dual_system_model()
    
    # Print model summary
    model.summary()
    
    # Get detailed summary
    summary_info = efnet_dual_system.get_model_summary()
    print("\nDetailed Model Information:")
    for key, value in summary_info.items():
        print(f"{key}: {value:,}" if isinstance(value, int) else f"{key}: {value}")
    
    print("\nEfficientNet-B3 Dual Branch System created successfully!")