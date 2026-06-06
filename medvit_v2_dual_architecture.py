"""
MedViT V2 Dual System — Architecture
======================================
Two-branch fusion model:
    Branch A: MedViT V2 (pre-trained)  → spatial feature extractor
    Branch B: 4-Class CNN (pre-trained) → deep feature extractor
    Fusion  : Concat → BatchNorm → Dense layers → 4-class softmax

Both branches share the same input tensor.
Feature extraction layers can be frozen during phase-1 training, then
unfrozen for fine-tuning in phase-2.
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

from medvit_v2_architecture import (
    build_medvit_v2, LayerNorm2D, KANLayer, LFFNLayer, DiNALayer,
    LFPBlock, EMHSALayer, MHCALayer, GFPBlock, StemLayer, PatchEmbedding,
    MEDVIT_V2_CONFIGS,
)

# ══════════════════════════════════════════════════════════════════════
CUSTOM_OBJECTS = {
    'LayerNorm2D':    LayerNorm2D,
    'KANLayer':       KANLayer,
    'LFFNLayer':      LFFNLayer,
    'DiNALayer':      DiNALayer,
    'LFPBlock':       LFPBlock,
    'EMHSALayer':     EMHSALayer,
    'MHCALayer':      MHCALayer,
    'GFPBlock':       GFPBlock,
    'StemLayer':      StemLayer,
    'PatchEmbedding': PatchEmbedding,
}

CNN_FEATURE_LAYER = 'dense2'        # name inside the 4-class CNN Sequential model
MEDVIT_FEATURE_LAYER = 'head_ln'    # LayerNorm before GAP inside MedViT V2


# ══════════════════════════════════════════════════════════════════════
# BRANCH BUILDERS
# ══════════════════════════════════════════════════════════════════════

def _find_layer_by_substr(model: keras.Model, substr: str):
    """Return the first layer whose name contains substr (case-insensitive)."""
    for layer in model.layers:
        if substr.lower() in layer.name.lower():
            return layer
    raise ValueError(
        f'No layer containing "{substr}" found in {model.name}. '
        f'Available layers: {[l.name for l in model.layers]}')


def build_medvit_branch(medvit_model: keras.Model,
                        inputs: tf.Tensor,
                        name_prefix: str = 'medvit') -> tf.Tensor:
    """
    Extract features from a pre-trained MedViT V2 model.
    Targets the layer named 'head_ln' (the LayerNorm before GAP) and
    applies GAP here so we get a 1-D feature vector.

    Falls back to searching for any 4-D spatial output if 'head_ln' is absent.
    """
    try:
        feat_layer = medvit_model.get_layer(MEDVIT_FEATURE_LAYER)
    except ValueError:
        feat_layer = None
        for layer in reversed(medvit_model.layers):
            out = layer.output
            if hasattr(out, 'shape') and len(out.shape) == 4:
                feat_layer = layer
                break
        if feat_layer is None:
            raise RuntimeError('Cannot find a spatial feature layer in MedViT V2.')

    # Sub-model: input → target spatial features
    feat_extractor = keras.Model(
        inputs=medvit_model.inputs,
        outputs=feat_layer.output,
        name=f'{name_prefix}_extractor',
    )

    feats = feat_extractor(inputs)
    out_rank = len(feat_extractor.output_shape)
    if out_rank == 4:       # (B, H, W, C) — spatial
        feats = layers.GlobalAveragePooling2D(name=f'{name_prefix}_gap')(feats)
    elif out_rank == 3:     # (B, N, C) — sequence
        feats = layers.GlobalAveragePooling1D(name=f'{name_prefix}_gap')(feats)
    # out_rank == 2: already (B, C), use as-is
    return feats, feat_extractor


def build_cnn_branch(cnn_model: keras.Model,
                     inputs: tf.Tensor,
                     name_prefix: str = 'cnn') -> tf.Tensor:
    """
    Extract features from the pre-trained 4-Class CNN.
    Looks for the 'dense2' layer (Dense-128 with BN+Dropout).
    Falls back to the GlobalAveragePooling layer if not found.
    """
    try:
        feat_layer = _find_layer_by_substr(cnn_model, CNN_FEATURE_LAYER)
    except ValueError:
        feat_layer = _find_layer_by_substr(cnn_model, 'global_avg_pool')

    feat_extractor = keras.Model(
        inputs=cnn_model.inputs,
        outputs=feat_layer.output,
        name=f'{name_prefix}_extractor',
    )

    feats = feat_extractor(inputs)                         # (B, F)
    return feats, feat_extractor


# ══════════════════════════════════════════════════════════════════════
# FUSION HEAD
# ══════════════════════════════════════════════════════════════════════

def build_fusion_head(medvit_feats: tf.Tensor,
                      cnn_feats: tf.Tensor,
                      num_classes: int = 4,
                      dropout_rate: float = 0.4) -> tf.Tensor:
    """
    Concatenate MedViT V2 and CNN features and classify via 2-layer MLP.
    """
    x = layers.Concatenate(name='fusion_concat')([medvit_feats, cnn_feats])
    x = layers.BatchNormalization(name='fusion_bn0')(x)

    x = layers.Dense(512, activation='relu', name='fusion_dense1')(x)
    x = layers.BatchNormalization(name='fusion_bn1')(x)
    x = layers.Dropout(dropout_rate, name='fusion_drop1')(x)

    x = layers.Dense(256, activation='relu', name='fusion_dense2')(x)
    x = layers.BatchNormalization(name='fusion_bn2')(x)
    x = layers.Dropout(dropout_rate * 0.75, name='fusion_drop2')(x)

    out = layers.Dense(num_classes, activation='softmax',
                       name='fusion_output')(x)
    return out


# ══════════════════════════════════════════════════════════════════════
# TOP-LEVEL BUILDER
# ══════════════════════════════════════════════════════════════════════

def build_dual_system(medvit_model: keras.Model,
                      cnn_model: keras.Model,
                      input_shape: tuple = (224, 224, 3),
                      num_classes: int = 4,
                      dropout_rate: float = 0.4) -> keras.Model:
    """
    Build the MedViT V2 + 4-Class CNN dual-branch fusion model.

    Returns the compiled Keras Model with both extractors embedded.
    Branch weights start frozen; call set_branches_trainable(True) to unfreeze.
    """
    inputs = keras.Input(shape=input_shape, name='dual_input')

    medvit_feats, medvit_extractor = build_medvit_branch(
        medvit_model, inputs, name_prefix='medvit')
    cnn_feats, cnn_extractor = build_cnn_branch(
        cnn_model, inputs, name_prefix='cnn')

    output = build_fusion_head(medvit_feats, cnn_feats,
                               num_classes=num_classes,
                               dropout_rate=dropout_rate)

    model = keras.Model(inputs=inputs, outputs=output,
                        name='MedViT_V2_Dual_System')

    # Freeze branch extractors by default (phase-1 training)
    medvit_extractor.trainable = False
    cnn_extractor.trainable    = False

    return model, medvit_extractor, cnn_extractor


def set_branches_trainable(model: keras.Model, trainable: bool,
                           medvit_extractor: keras.Model = None,
                           cnn_extractor: keras.Model = None):
    """Toggle branch trainability for phase-1 / phase-2 training."""
    if medvit_extractor is not None:
        medvit_extractor.trainable = trainable
    if cnn_extractor is not None:
        cnn_extractor.trainable = trainable

    # Re-assign to propagate the flag through the functional API
    for layer in model.layers:
        if hasattr(layer, 'trainable'):
            n = layer.name.lower()
            if 'medvit_extractor' in n or 'cnn_extractor' in n:
                layer.trainable = trainable


# ══════════════════════════════════════════════════════════════════════
# QUICK TEST
# ══════════════════════════════════════════════════════════════════════

if __name__ == '__main__':
    print('Building MedViT V2 Dual System (architecture test) ...')

    # Build a fresh MedViT V2 Tiny (no pre-trained weights needed for shape test)
    medvit = build_medvit_v2(
        input_shape=(224, 224, 3), num_classes=4, variant='tiny')

    # Build a minimal 4-class CNN placeholder
    cnn_input = keras.Input(shape=(224, 224, 3), name='input_1')
    x = layers.Conv2D(32, 3, activation='relu', padding='same', name='conv1_1')(cnn_input)
    x = layers.Conv2D(32, 3, activation='relu', padding='same', name='conv1_2')(x)
    x = layers.MaxPooling2D(name='pool1')(x)
    x = layers.GlobalAveragePooling2D(name='global_avg_pool')(x)
    x = layers.Dense(256, activation='relu', name='dense1')(x)
    x = layers.BatchNormalization(name='bn_dense1')(x)
    x = layers.Dropout(0.5, name='dropout_dense1')(x)
    x = layers.Dense(128, activation='relu', name='dense2')(x)
    x = layers.BatchNormalization(name='bn_dense2')(x)
    x = layers.Dense(4, activation='softmax', name='multiclass_output')(x)
    cnn = keras.Model(inputs=cnn_input, outputs=x, name='4class_cnn')

    dual, med_ext, cnn_ext = build_dual_system(medvit, cnn)
    dual.summary(line_length=100)
    print(f'Total params: {dual.count_params():,}')
    print('Dual System architecture test passed.')
