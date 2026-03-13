#!/usr/bin/env python3
"""
Multi-Dual System Architecture: Hybrid 3-Class + 4-Class Brain Tumor Classification
================================================================================

This module implements a novel hybrid dual-branch system that combines:
- Branch 1: 3-class classification (Glioma, Meningioma, Pituitary) 
- Branch 2: 4-class classification (Glioma, Meningioma, No Tumor, Pituitary)

The system provides complementary information and clinical workflow alignment.

Author: Yannis Balasis
Date: March 2026
"""

import tensorflow as tf
from tensorflow.keras import layers, Model, Input
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import SparseCategoricalCrossentropy
from tensorflow.keras.metrics import SparseCategoricalAccuracy
import numpy as np
from typing import Tuple, Dict, List, Optional


class MultiDualSystemArchitecture:
    """
    Hybrid Dual Branch System for Brain Tumor Classification
    
    Features:
    - Shared CNN backbone for feature extraction
    - Branch 1: 3-class tumor type classification (masked loss for no_tumor samples)
    - Branch 2: 4-class full classification (including No Tumor)
    - Multiple fusion strategies
    - Clinical workflow alignment
    """
    
    def __init__(
        self,
        input_shape: Tuple[int, int, int] = (224, 224, 3),
        backbone_type: str = "efficient",
        fusion_strategy: str = "weighted_ensemble",
        dropout_rate: float = 0.3,
        seed: int = 789
    ):
        self.input_shape = input_shape
        self.backbone_type = backbone_type
        self.fusion_strategy = fusion_strategy
        self.dropout_rate = dropout_rate
        self.seed = seed
        
        tf.random.set_seed(seed)
        np.random.seed(seed)
        
        self.class_mapping_3 = {
            0: "Glioma",
            1: "Meningioma",
            2: "Pituitary"
        }
        
        self.class_mapping_4 = {
            0: "Glioma",
            1: "Meningioma",
            2: "No Tumor",
            3: "Pituitary"
        }
        
        self.model = self._build_hybrid_model()

    # ------------------------------------------------------------------
    # Masked loss and accuracy for Branch 1
    # (no_tumor samples have label=-1 and must be ignored)
    # ------------------------------------------------------------------

    def _masked_sparse_crossentropy(self, y_true, y_pred):
        """
        Sparse categorical cross-entropy that ignores samples with label=-1.
        Used for Branch 1 (3-class) so that no_tumor samples do not
        contribute to the loss.
        """
        mask = tf.cast(tf.not_equal(y_true, -1), dtype=tf.float32)
        # Replace -1 with 0 just to avoid out-of-range errors during computation;
        # those positions are zeroed out by the mask anyway.
        y_true_clipped = tf.maximum(y_true, 0)
        loss = tf.keras.losses.sparse_categorical_crossentropy(y_true_clipped, y_pred)
        loss = loss * mask
        return tf.reduce_sum(loss) / (tf.reduce_sum(mask) + 1e-8)

    def _masked_sparse_accuracy(self, y_true, y_pred):
        """
        Sparse categorical accuracy that ignores samples with label=-1.
        Used for Branch 1 (3-class) metric tracking.
        """
        mask = tf.cast(tf.not_equal(y_true, -1), dtype=tf.float32)
        y_true_clipped = tf.maximum(y_true, 0)
        pred_classes = tf.cast(tf.argmax(y_pred, axis=1), dtype=y_true_clipped.dtype)
        correct = tf.cast(tf.equal(pred_classes, y_true_clipped), dtype=tf.float32)
        correct = correct * mask
        return tf.reduce_sum(correct) / (tf.reduce_sum(mask) + 1e-8)

    # ------------------------------------------------------------------
    # Backbone
    # ------------------------------------------------------------------

    def _create_efficient_backbone(self, input_tensor):
        """Create efficient CNN backbone optimized for medical imaging."""
        # Block 1
        x = layers.Conv2D(32, (3, 3), activation='relu', padding='same', name='conv1_1')(input_tensor)
        x = layers.BatchNormalization(name='bn1_1')(x)
        x = layers.Conv2D(32, (3, 3), activation='relu', padding='same', name='conv1_2')(x)
        x = layers.BatchNormalization(name='bn1_2')(x)
        x = layers.MaxPooling2D((2, 2), name='pool1')(x)
        x = layers.Dropout(self.dropout_rate * 0.5, name='dropout1')(x)

        # Block 2
        x = layers.Conv2D(64, (3, 3), activation='relu', padding='same', name='conv2_1')(x)
        x = layers.BatchNormalization(name='bn2_1')(x)
        x = layers.Conv2D(64, (3, 3), activation='relu', padding='same', name='conv2_2')(x)
        x = layers.BatchNormalization(name='bn2_2')(x)
        x = layers.MaxPooling2D((2, 2), name='pool2')(x)
        x = layers.Dropout(self.dropout_rate * 0.7, name='dropout2')(x)

        # Block 3
        x = layers.Conv2D(128, (3, 3), activation='relu', padding='same', name='conv3_1')(x)
        x = layers.BatchNormalization(name='bn3_1')(x)
        x = layers.Conv2D(128, (3, 3), activation='relu', padding='same', name='conv3_2')(x)
        x = layers.BatchNormalization(name='bn3_2')(x)
        x = layers.MaxPooling2D((2, 2), name='pool3')(x)
        x = layers.Dropout(self.dropout_rate, name='dropout3')(x)

        # Block 4
        x = layers.Conv2D(256, (3, 3), activation='relu', padding='same', name='conv4_1')(x)
        x = layers.BatchNormalization(name='bn4_1')(x)
        x = layers.Conv2D(256, (3, 3), activation='relu', padding='same', name='conv4_2')(x)
        x = layers.BatchNormalization(name='bn4_2')(x)
        x = layers.MaxPooling2D((2, 2), name='pool4')(x)
        x = layers.Dropout(self.dropout_rate, name='dropout4')(x)

        # Global pooling
        x = layers.GlobalAveragePooling2D(name='global_pool')(x)
        x = layers.Dense(512, activation='relu', name='feature_dense1')(x)
        x = layers.BatchNormalization(name='feature_bn')(x)
        x = layers.Dropout(self.dropout_rate, name='feature_dropout')(x)

        return x

    # ------------------------------------------------------------------
    # Fusion
    # ------------------------------------------------------------------

    def _create_weighted_ensemble_fusion(self, branch1_features, branch2_features,
                                          branch1_output, branch2_output):
        """Weighted ensemble fusion strategy."""
        combined_features = layers.Concatenate(name='fusion_concat')([branch1_features, branch2_features])
        fusion_weights = layers.Dense(128, activation='relu', name='fusion_weights_dense')(combined_features)
        fusion_weights = layers.BatchNormalization(name='fusion_weights_bn')(fusion_weights)
        fusion_weights = layers.Dropout(self.dropout_rate * 0.5, name='fusion_weights_dropout')(fusion_weights)
        fusion_output = layers.Dense(4, activation='softmax', name='fusion_output')(fusion_weights)
        return fusion_output

    # ------------------------------------------------------------------
    # Model build
    # ------------------------------------------------------------------

    def _build_hybrid_model(self) -> Model:
        """Build the complete hybrid dual-branch model."""
        inputs = Input(shape=self.input_shape, name='input_image')

        # Shared backbone
        backbone_features = self._create_efficient_backbone(inputs)

        # Branch 1: 3-class (tumor types only)
        branch1_features = layers.Dense(256, activation='relu', name='branch1_dense1')(backbone_features)
        branch1_features = layers.BatchNormalization(name='branch1_bn1')(branch1_features)
        branch1_features = layers.Dropout(self.dropout_rate, name='branch1_dropout1')(branch1_features)
        branch1_features = layers.Dense(128, activation='relu', name='branch1_dense2')(branch1_features)
        branch1_features = layers.BatchNormalization(name='branch1_bn2')(branch1_features)
        branch1_features = layers.Dropout(self.dropout_rate * 0.5, name='branch1_dropout2')(branch1_features)
        branch1_output = layers.Dense(3, activation='softmax', name='branch1_3class')(branch1_features)

        # Branch 2: 4-class (including No Tumor)
        branch2_features = layers.Dense(256, activation='relu', name='branch2_dense1')(backbone_features)
        branch2_features = layers.BatchNormalization(name='branch2_bn1')(branch2_features)
        branch2_features = layers.Dropout(self.dropout_rate, name='branch2_dropout1')(branch2_features)
        branch2_features = layers.Dense(128, activation='relu', name='branch2_dense2')(branch2_features)
        branch2_features = layers.BatchNormalization(name='branch2_bn2')(branch2_features)
        branch2_features = layers.Dropout(self.dropout_rate * 0.5, name='branch2_dropout2')(branch2_features)
        branch2_output = layers.Dense(4, activation='softmax', name='branch2_4class')(branch2_features)

        # Fusion
        fusion_output = self._create_weighted_ensemble_fusion(
            branch1_features, branch2_features, branch1_output, branch2_output
        )

        model = Model(
            inputs=inputs,
            outputs={
                'branch1_3class': branch1_output,
                'branch2_4class': branch2_output,
                'fusion_output': fusion_output
            },
            name='MultiDualSystem'
        )

        return model

    # ------------------------------------------------------------------
    # Compile
    # ------------------------------------------------------------------

    def compile_model(
        self,
        learning_rate: float = 0.001,
        branch1_weight: float = 0.3,
        branch2_weight: float = 0.4,
        fusion_weight: float = 0.3
    ):
        """
        Compile the model with multi-task loss.

        Branch 1 uses masked loss/accuracy so that no_tumor samples
        (label=-1) are excluded from gradient updates and metric tracking.
        """
        self.model.compile(
            optimizer=Adam(learning_rate=learning_rate),
            loss={
                'branch1_3class': self._masked_sparse_crossentropy,
                'branch2_4class': SparseCategoricalCrossentropy(name='branch2_loss'),
                'fusion_output': SparseCategoricalCrossentropy(name='fusion_loss')
            },
            loss_weights={
                'branch1_3class': branch1_weight,
                'branch2_4class': branch2_weight,
                'fusion_output': fusion_weight
            },
            metrics={
                'branch1_3class': [self._masked_sparse_accuracy],
                'branch2_4class': [SparseCategoricalAccuracy(name='branch2_accuracy')],
                'fusion_output': [SparseCategoricalAccuracy(name='fusion_accuracy')]
            }
        )

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------

    def get_model_summary(self) -> str:
        summary_lines = []
        summary_lines.append("Multi-Dual System Architecture Summary")
        summary_lines.append("=" * 50)
        summary_lines.append(f"Backbone Type: {self.backbone_type}")
        summary_lines.append(f"Fusion Strategy: {self.fusion_strategy}")
        summary_lines.append(f"Input Shape: {self.input_shape}")
        summary_lines.append(f"Dropout Rate: {self.dropout_rate}")
        summary_lines.append(f"Random Seed: {self.seed}")
        summary_lines.append("")
        summary_lines.append("Outputs:")
        summary_lines.append(f"  - Branch 1 (3-class): {list(self.class_mapping_3.values())}")
        summary_lines.append(f"  - Branch 2 (4-class): {list(self.class_mapping_4.values())}")
        summary_lines.append(f"  - Fusion Output: Primary classification")
        summary_lines.append("")

        total_params = self.model.count_params()
        try:
            trainable_params = sum([w.shape.num_elements() for w in self.model.trainable_weights])
        except Exception:
            trainable_params = total_params

        summary_lines.append(f"Total Parameters: {total_params:,}")
        summary_lines.append(f"Trainable Parameters: {trainable_params:,}")
        summary_lines.append("")

        return "\n".join(summary_lines)


def create_multi_dual_system(
    backbone_type: str = "efficient",
    fusion_strategy: str = "weighted_ensemble",
    dropout_rate: float = 0.3
) -> MultiDualSystemArchitecture:
    """Factory function to create Multi-Dual System."""
    print(f"Creating Multi-Dual System...")
    print(f"   Backbone: {backbone_type}")
    print(f"   Fusion: {fusion_strategy}")
    print(f"   Dropout: {dropout_rate}")

    system = MultiDualSystemArchitecture(
        backbone_type=backbone_type,
        fusion_strategy=fusion_strategy,
        dropout_rate=dropout_rate
    )

    print("Multi-Dual System created successfully!")
    print(system.get_model_summary())

    return system


if __name__ == "__main__":
    print("Multi-Dual System Architecture Demo")

    system = create_multi_dual_system("efficient", "weighted_ensemble", 0.3)
    system.compile_model()

    print(f"Model input shape: {system.model.input.shape}")
    print(f"Model outputs: {list(system.model.output.keys())}")
    print(f"Total parameters: {system.model.count_params():,}")