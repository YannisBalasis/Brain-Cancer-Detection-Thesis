"""
MedViT V2 Architecture — TensorFlow/Keras Implementation
=========================================================
Based on: "MedViT V2: Medical Image Classification with KAN-Integrated
Transformers and Dilated Neighborhood Attention" (arXiv:2502.13693v2)

Key components:
    - Stem: 4-layer conv stem
    - LFP Block: DiNA (Dilated Neighborhood Attention) + LFFN
    - GFP Block: E-MHSA + MHCA + KAN → Concat
    - Hierarchical Hybrid Strategy: (LFP×N + GFP×1)×L per stage
    - KAN with RSWAF (Reflectional Switch Activation Functions)

Variants: Tiny (~12M), Small (~29M), Base (~72M), Large (~162M)
"""

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


# ══════════════════════════════════════════════════════════════════════
# UTILITY LAYERS
# ══════════════════════════════════════════════════════════════════════

class LayerNorm2D(layers.Layer):
    """LayerNorm applied along channel axis for (B, H, W, C) tensors."""
    def __init__(self, eps=1e-6, **kwargs):
        super().__init__(**kwargs)
        self.eps = eps

    def build(self, input_shape):
        c = input_shape[-1]
        self.gamma = self.add_weight(shape=(c,), initializer='ones',  name='gamma')
        self.beta  = self.add_weight(shape=(c,), initializer='zeros', name='beta')

    def call(self, x):
        mean = tf.reduce_mean(x, axis=-1, keepdims=True)
        var  = tf.math.reduce_variance(x, axis=-1, keepdims=True)
        x    = (x - mean) / tf.sqrt(var + self.eps)
        return self.gamma * x + self.beta


# ══════════════════════════════════════════════════════════════════════
# KAN WITH RSWAF (Section 3.3)
# ══════════════════════════════════════════════════════════════════════

class KANLayer(layers.Layer):
    """
    Kolmogorov-Arnold Network layer using RSWAF basis functions.
    φ_RSWAF(r) = 1 − (tanh(r / h))²
    RSWAF(x)   = Σᵢ wᵢ · φ_RSWAF((x − cᵢ) / h)
    Total:  φ(x) = w_b · SiLU(x) + w_s · RSWAF(x)
    """
    def __init__(self, in_dim, out_dim, num_centers=8, **kwargs):
        super().__init__(**kwargs)
        self.in_dim      = in_dim
        self.out_dim     = out_dim
        self.num_centers = num_centers

    def build(self, input_shape):
        k, d = self.num_centers, self.in_dim
        self.centers = self.add_weight(
            shape=(k,), name='centers',
            initializer=tf.initializers.RandomUniform(-1.0, 1.0), trainable=True)
        self.h = self.add_weight(
            shape=(), name='bandwidth',
            initializer=tf.initializers.constant(1.0), trainable=True)
        # Spline weights: (in_dim, num_centers, out_dim)
        self.W_spline = self.add_weight(
            shape=(d, k, self.out_dim), name='W_spline',
            initializer='glorot_uniform', trainable=True)
        # Base weights: (in_dim, out_dim)
        self.W_base = self.add_weight(
            shape=(d, self.out_dim), name='W_base',
            initializer='glorot_uniform', trainable=True)
        self.bias = self.add_weight(
            shape=(self.out_dim,), name='bias',
            initializer='zeros', trainable=True)

    def call(self, x):
        # x: (..., in_dim)
        # Base: SiLU = x * sigmoid(x)
        base_out = tf.matmul(x * tf.sigmoid(x), self.W_base)   # (..., out_dim)

        # RSWAF spline
        # x_exp: (..., in_dim, 1)  centers: (1,...,1, num_centers)
        x_exp = tf.expand_dims(x, -1)
        c_shape = [1] * (len(x.shape) - 1) + [1, self.num_centers]
        c = tf.reshape(self.centers, c_shape)
        diff  = (x_exp - c) / (tf.abs(self.h) + 1e-8)          # (..., in_dim, k)
        phi   = 1.0 - tf.tanh(diff) ** 2                        # RSWAF
        spline_out = tf.einsum('...ik,iko->...o', phi, self.W_spline)  # (..., out_dim)

        return base_out + spline_out + self.bias


# ══════════════════════════════════════════════════════════════════════
# LFP BLOCK: DiNA + LFFN (Section 3.2)
# ══════════════════════════════════════════════════════════════════════

class LFFNLayer(layers.Layer):
    """
    Local Feed-Forward Network.
    Seq2Img → Conv1×1 → DWConv3×3 → Conv1×1 → Img2Seq
    Expansion ratio = 3 (from paper Section 3.5).
    """
    def __init__(self, dim, expansion=3, **kwargs):
        super().__init__(**kwargs)
        hidden = dim * expansion
        self.conv1  = layers.Conv2D(hidden, 1, padding='same', use_bias=True)
        self.dw     = layers.DepthwiseConv2D(3, padding='same', use_bias=True)
        self.conv2  = layers.Conv2D(dim, 1, padding='same', use_bias=True)
        self.bn1    = layers.BatchNormalization()
        self.bn2    = layers.BatchNormalization()
        self.bn3    = layers.BatchNormalization()

    def call(self, x, training=False):
        x = tf.nn.relu(self.bn1(self.conv1(x), training=training))
        x = tf.nn.relu(self.bn2(self.dw(x),    training=training))
        x = self.bn3(self.conv2(x),             training=training)
        return x


class DiNALayer(layers.Layer):
    """
    Dilated Neighborhood Attention (Section 3.2).
    For each token i, attends to k nearest neighbours satisfying
    i mod δ = j mod δ (i.e., dilated neighbourhood of size kernel_size² at dilation δ).
    Implemented via tf.image.extract_patches with rates=dilation.
    """
    def __init__(self, dim, num_heads=4, kernel_size=7, dilation=1,
                 qkv_bias=True, **kwargs):
        super().__init__(**kwargs)
        assert dim % num_heads == 0, "dim must be divisible by num_heads"
        self.dim         = dim
        self.num_heads   = num_heads
        self.head_dim    = dim // num_heads
        self.kernel_size = kernel_size
        self.dilation    = dilation
        self.scale       = self.head_dim ** -0.5
        self.n_neighbors = kernel_size ** 2

        self.qkv  = layers.Dense(dim * 3, use_bias=qkv_bias)
        self.proj = layers.Dense(dim)
        self.ln   = LayerNorm2D()

    def call(self, x, training=False):
        B  = tf.shape(x)[0]
        H  = tf.shape(x)[1]
        W  = tf.shape(x)[2]
        C  = self.dim

        # ── QKV projection ──────────────────────────────────────────
        qkv = self.qkv(x)                                # (B,H,W,3C)
        qkv = tf.reshape(qkv, [B, H*W, 3, self.num_heads, self.head_dim])
        qkv = tf.transpose(qkv, [2, 0, 1, 3, 4])        # (3,B,HW,heads,hd)
        q, k, v = qkv[0], qkv[1], qkv[2]                # each (B,HW,heads,hd)
        q = q * self.scale

        # ── Extract dilated patches for K and V ─────────────────────
        k_sp = tf.reshape(k, [B, H, W, self.num_heads * self.head_dim])
        v_sp = tf.reshape(v, [B, H, W, self.num_heads * self.head_dim])

        pad = (self.kernel_size // 2) * self.dilation
        k_sp = tf.pad(k_sp, [[0,0],[pad,pad],[pad,pad],[0,0]])
        v_sp = tf.pad(v_sp, [[0,0],[pad,pad],[pad,pad],[0,0]])

        k_patches = tf.image.extract_patches(
            k_sp,
            sizes=[1, self.kernel_size, self.kernel_size, 1],
            strides=[1, 1, 1, 1],
            rates=[1, self.dilation, self.dilation, 1],
            padding='VALID')                              # (B,H,W,k²·C)

        v_patches = tf.image.extract_patches(
            v_sp,
            sizes=[1, self.kernel_size, self.kernel_size, 1],
            strides=[1, 1, 1, 1],
            rates=[1, self.dilation, self.dilation, 1],
            padding='VALID')                              # (B,H,W,k²·C)

        # ── Reshape for attention ────────────────────────────────────
        k_patches = tf.reshape(k_patches,
            [B, H*W, self.n_neighbors, self.num_heads, self.head_dim])
        v_patches = tf.reshape(v_patches,
            [B, H*W, self.n_neighbors, self.num_heads, self.head_dim])

        # q: (B, HW, heads, 1, hd)
        q = tf.expand_dims(q, 2)                         # (B,HW,1,heads,hd)
        q = tf.transpose(q, [0, 1, 3, 2, 4])            # (B,HW,heads,1,hd)

        # k: (B, HW, heads, hd, k²)
        k_attn = tf.transpose(k_patches, [0, 1, 3, 4, 2])

        # Attention scores: (B, HW, heads, 1, k²)
        attn = tf.matmul(q, k_attn)
        attn = tf.nn.softmax(attn, axis=-1)

        # v: (B, HW, heads, k², hd)
        v_attn = tf.transpose(v_patches, [0, 1, 3, 2, 4])

        # out: (B, HW, heads, 1, hd) → squeeze → (B, HW, heads, hd)
        out = tf.matmul(attn, v_attn)
        out = tf.squeeze(out, axis=-2)                   # (B,HW,heads,hd)
        out = tf.reshape(out, [B, H*W, C])
        out = tf.reshape(out, [B, H, W, C])

        return self.proj(out)


class LFPBlock(layers.Layer):
    """
    Local Feature Perception Block (Section 3.2).
    z̃ⁿ = DiNA(LN(z^{n-1})) + z^{n-1}
    zⁿ  = LFFN(LN(z̃ⁿ))     + z̃ⁿ
    """
    def __init__(self, dim, num_heads=4, kernel_size=7, dilation=1, **kwargs):
        super().__init__(**kwargs)
        self.ln1  = LayerNorm2D()
        self.ln2  = LayerNorm2D()
        self.dina = DiNALayer(dim, num_heads=num_heads,
                              kernel_size=kernel_size, dilation=dilation)
        self.lffn = LFFNLayer(dim, expansion=3)

    def call(self, x, training=False):
        x = self.dina(self.ln1(x), training=training) + x
        x = self.lffn(self.ln2(x), training=training) + x
        return x


# ══════════════════════════════════════════════════════════════════════
# GFP BLOCK: E-MHSA + MHCA + KAN (Section 3.3 & eq. 11)
# ══════════════════════════════════════════════════════════════════════

class EMHSALayer(layers.Layer):
    """
    Efficient Multi-Head Self-Attention with spatial reduction for K/V.
    K and V are downsampled by spatial_reduction factor via AvgPool.
    """
    def __init__(self, dim, num_heads=4, spatial_reduction=8,
                 qkv_bias=True, **kwargs):
        super().__init__(**kwargs)
        self.dim              = dim
        self.num_heads        = num_heads
        self.head_dim         = dim // num_heads
        self.spatial_reduction = spatial_reduction
        self.scale            = self.head_dim ** -0.5

        self.q   = layers.Dense(dim, use_bias=qkv_bias)
        self.kv  = layers.Dense(dim * 2, use_bias=qkv_bias)
        self.proj = layers.Dense(dim)
        self.bn   = layers.BatchNormalization()

        if spatial_reduction > 1:
            self.sr = layers.AveragePooling2D(
                pool_size=spatial_reduction,
                strides=spatial_reduction, padding='same')
        else:
            self.sr = None

    def call(self, x, training=False):
        B  = tf.shape(x)[0]
        H  = tf.shape(x)[1]
        W  = tf.shape(x)[2]
        C  = self.dim

        x_seq = tf.reshape(x, [B, H*W, C])

        # ── Q ───────────────────────────────────────────────────────
        q = self.q(x_seq)                               # (B, HW, C)
        q = tf.reshape(q, [B, H*W, self.num_heads, self.head_dim])
        q = tf.transpose(q, [0, 2, 1, 3]) * self.scale # (B,heads,HW,hd)

        # ── Spatially-reduced K, V ──────────────────────────────────
        if self.sr is not None:
            x_r   = self.sr(x)
            x_r   = tf.nn.relu(self.bn(x_r, training=training))
            Hr    = tf.shape(x_r)[1]
            Wr    = tf.shape(x_r)[2]
            x_r_s = tf.reshape(x_r, [B, Hr*Wr, C])
        else:
            x_r_s = x_seq

        kv     = self.kv(x_r_s)                        # (B, Hr*Wr, 2C)
        n_kv   = tf.shape(kv)[1]
        kv     = tf.reshape(kv, [B, n_kv, 2, self.num_heads, self.head_dim])
        kv     = tf.transpose(kv, [2, 0, 3, 1, 4])    # (2,B,heads,n_kv,hd)
        k, v   = kv[0], kv[1]

        # ── Attention ────────────────────────────────────────────────
        attn = tf.matmul(q, k, transpose_b=True)       # (B,heads,HW,n_kv)
        attn = tf.nn.softmax(attn, axis=-1)

        out  = tf.matmul(attn, v)                       # (B,heads,HW,hd)
        out  = tf.transpose(out, [0, 2, 1, 3])         # (B,HW,heads,hd)
        out  = tf.reshape(out, [B, H*W, C])
        out  = tf.reshape(out, [B, H, W, C])

        return self.proj(out)


class MHCALayer(layers.Layer):
    """
    Multi-Head Convolutional Attention.
    Seq2Img → Conv1×1 → DWConv3×3 → Conv1×1 → Img2Seq
    Channel shrink ratio r = 0.75 (paper Section 3.5).
    """
    def __init__(self, dim, shrink_ratio=0.75, **kwargs):
        super().__init__(**kwargs)
        inner = max(1, int(dim * shrink_ratio))
        self.conv1 = layers.Conv2D(inner, 1, padding='same')
        self.dw    = layers.DepthwiseConv2D(3, padding='same')
        self.conv2 = layers.Conv2D(dim, 1, padding='same')
        self.bn1   = layers.BatchNormalization()
        self.bn2   = layers.BatchNormalization()
        self.bn3   = layers.BatchNormalization()

    def call(self, x, training=False):
        x = tf.nn.relu(self.bn1(self.conv1(x), training=training))
        x = tf.nn.relu(self.bn2(self.dw(x),    training=training))
        x = self.bn3(self.conv2(x),             training=training)
        return x


class GFPBlock(layers.Layer):
    """
    Global Feature Perception Block (Section 3.3, eq. 11).
    z̃^{n+1} = E-MHSA(LN(zⁿ)) + zⁿ
    ẑ^{n+1}  = MHCA(z̃^{n+1})
    z̄^{n+1}  = KAN(ẑ^{n+1})
    z^{n+1}  = Concat(z̃^{n+1}, z̄^{n+1}) → Linear(2C → C)
    """
    def __init__(self, dim, num_heads=4, spatial_reduction=8,
                 kan_expansion=2, **kwargs):
        super().__init__(**kwargs)
        self.dim = dim

        self.ln    = LayerNorm2D()
        self.emhsa = EMHSALayer(dim, num_heads=num_heads,
                                spatial_reduction=spatial_reduction)
        self.mhca  = MHCALayer(dim, shrink_ratio=0.75)
        self.kan   = KANLayer(dim, dim * kan_expansion)
        self.kan_proj    = layers.Dense(dim)      # dim*expansion → dim
        self.concat_proj = layers.Dense(dim)      # 2·dim → dim

    def call(self, x, training=False):
        B = tf.shape(x)[0]
        H = tf.shape(x)[1]
        W = tf.shape(x)[2]
        C = self.dim

        # Step 1: E-MHSA + residual
        z_tilde = self.emhsa(self.ln(x), training=training) + x  # (B,H,W,C)

        # Step 2: MHCA
        z_hat = self.mhca(z_tilde, training=training)             # (B,H,W,C)

        # Step 3: KAN on flattened tokens
        z_hat_flat = tf.reshape(z_hat, [B, H*W, C])
        z_bar      = self.kan(z_hat_flat)                          # (B,HW,C*exp)
        z_bar      = self.kan_proj(z_bar)                          # (B,HW,C)
        z_bar      = tf.reshape(z_bar, [B, H, W, C])

        # Step 4: Concat + projection
        z_tilde_f = tf.reshape(z_tilde, [B, H*W, C])
        z_bar_f   = tf.reshape(z_bar,   [B, H*W, C])
        concat    = tf.concat([z_tilde_f, z_bar_f], axis=-1)      # (B,HW,2C)
        out       = self.concat_proj(concat)                        # (B,HW,C)
        out       = tf.reshape(out, [B, H, W, C])

        return out


# ══════════════════════════════════════════════════════════════════════
# STEM & PATCH EMBEDDING
# ══════════════════════════════════════════════════════════════════════

class StemLayer(layers.Layer):
    """
    4-layer conv stem (Table 3):
    Conv3×3(64,s=2) → Conv3×3(32,s=1) → Conv3×3(64,s=1) → Conv3×3(64,s=2)
    Output: (B, H/4, W/4, 64)
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.conv1 = layers.Conv2D(64, 3, strides=2, padding='same')
        self.conv2 = layers.Conv2D(32, 3, strides=1, padding='same')
        self.conv3 = layers.Conv2D(64, 3, strides=1, padding='same')
        self.conv4 = layers.Conv2D(64, 3, strides=2, padding='same')
        self.bn1   = layers.BatchNormalization()
        self.bn2   = layers.BatchNormalization()
        self.bn3   = layers.BatchNormalization()
        self.bn4   = layers.BatchNormalization()

    def call(self, x, training=False):
        x = tf.nn.relu(self.bn1(self.conv1(x), training=training))
        x = tf.nn.relu(self.bn2(self.conv2(x), training=training))
        x = tf.nn.relu(self.bn3(self.conv3(x), training=training))
        x = tf.nn.relu(self.bn4(self.conv4(x), training=training))
        return x


class PatchEmbedding(layers.Layer):
    """
    Patch embedding for each stage transition (Table 3).
    Stage 1 : Conv 1×1 (no downsampling, stride=1)
    Stages 2-4: AvgPool(s=2) + Conv 1×1
    """
    def __init__(self, out_channels, stride=1, **kwargs):
        super().__init__(**kwargs)
        self.stride = stride
        if stride > 1:
            self.pool = layers.AveragePooling2D(
                pool_size=stride, strides=stride, padding='same')
        self.proj = layers.Conv2D(out_channels, 1, padding='same')
        self.bn   = layers.BatchNormalization()

    def call(self, x, training=False):
        if self.stride > 1:
            x = self.pool(x)
        return tf.nn.relu(self.bn(self.proj(x), training=training))


# ══════════════════════════════════════════════════════════════════════
# MEDVIT V2 MODEL BUILDER
# ══════════════════════════════════════════════════════════════════════

# Variant configurations — matches Table 3
MEDVIT_V2_CONFIGS = {
    'tiny': {
        'channels':          [96, 128, 192, 384],
        'num_heads':         [3,  4,   6,   12],
        'spatial_reduction': [1,  8,   4,   2],
        'dilation':          [1,  8,   4,   1],
        'stage_blocks': [
            # (n_lfp, n_gfp, L)
            (2, 0, 1),   # Stage 1: LFP×2
            (1, 1, 1),   # Stage 2: (LFP×1 + GFP×1) × 1
            (2, 1, 2),   # Stage 3: (LFP×2 + GFP×1) × 2
            (0, 1, 1),   # Stage 4: GFP×1
        ],
    },
    'small': {
        'channels':          [128, 128, 256, 512],
        'num_heads':         [4,   4,   8,   16],
        'spatial_reduction': [1,   8,   4,   2],
        'dilation':          [1,   8,   4,   1],
        'stage_blocks': [
            (2, 0, 1),
            (1, 1, 1),
            (2, 1, 2),
            (0, 1, 2),
        ],
    },
    'base': {
        'channels':          [192, 192, 384, 768],
        'num_heads':         [6,   6,   12,  24],
        'spatial_reduction': [1,   8,   4,   2],
        'dilation':          [1,   8,   4,   1],
        'stage_blocks': [
            (2, 0, 1),
            (1, 1, 1),
            (2, 1, 2),
            (0, 1, 2),
        ],
    },
    'large': {
        'channels':          [256, 256, 512, 1024],
        'num_heads':         [8,   8,   16,  32],
        'spatial_reduction': [1,   8,   4,   2],
        'dilation':          [1,   8,   4,   1],
        'stage_blocks': [
            (2, 0, 1),
            (1, 1, 1),
            (2, 1, 2),
            (0, 1, 2),
        ],
    },
}


def build_medvit_v2(
    input_shape=(224, 224, 3),
    num_classes=4,
    variant='tiny',
    dropout_rate=0.1,
):
    """
    Build the MedViT V2 Keras functional model.

    Args:
        input_shape  : (H, W, C), default (224, 224, 3)
        num_classes  : number of output classes
        variant      : 'tiny' | 'small' | 'base' | 'large'
        dropout_rate : dropout before the classifier head

    Returns:
        keras.Model
    """
    cfg      = MEDVIT_V2_CONFIGS[variant]
    channels = cfg['channels']
    heads    = cfg['num_heads']
    sr_list  = cfg['spatial_reduction']
    dil_list = cfg['dilation']
    blocks   = cfg['stage_blocks']

    inputs = keras.Input(shape=input_shape, name='input')

    # ── Stem ─────────────────────────────────────────────────────
    x = StemLayer(name='stem')(inputs)                   # (B, H/4, W/4, 64)

    # ── Stage loop ───────────────────────────────────────────────
    for stage_idx, (n_lfp, n_gfp, L) in enumerate(blocks):
        C        = channels[stage_idx]
        n_heads  = heads[stage_idx]
        sr       = sr_list[stage_idx]
        dil      = dil_list[stage_idx]
        stride   = 1 if stage_idx == 0 else 2

        # Patch embedding
        x = PatchEmbedding(C, stride=stride,
                           name=f's{stage_idx+1}_patch_embed')(x)

        # Repeat (LFP×n_lfp + GFP×n_gfp) L times
        for l_idx in range(L):
            for i in range(n_lfp):
                x = LFPBlock(C, num_heads=n_heads,
                             kernel_size=7, dilation=dil,
                             name=f's{stage_idx+1}_l{l_idx}_lfp{i}')(x)
            for i in range(n_gfp):
                x = GFPBlock(C, num_heads=n_heads,
                             spatial_reduction=sr,
                             kan_expansion=2,
                             name=f's{stage_idx+1}_l{l_idx}_gfp{i}')(x)

    # ── Classifier head ──────────────────────────────────────────
    x = layers.GlobalAveragePooling2D(name='gap')(x)
    x = layers.LayerNormalization(epsilon=1e-6, name='head_ln')(x)
    if dropout_rate > 0:
        x = layers.Dropout(dropout_rate, name='head_dropout')(x)
    outputs = layers.Dense(num_classes, activation='softmax',
                           name='classifier')(x)

    model = keras.Model(inputs=inputs, outputs=outputs,
                        name=f'MedViTV2_{variant}')
    return model


def build_medvit_v2_feature_extractor(variant='tiny', input_shape=(224, 224, 3)):
    """
    Returns MedViT V2 model without the classifier head.
    Used as a feature extractor for the Dual System (Experiment 2).
    Output: (B, channels[-1]) after GAP.
    """
    full_model = build_medvit_v2(
        input_shape=input_shape,
        num_classes=4,
        variant=variant,
        dropout_rate=0.0,
    )
    # Remove the last Dense (classifier)
    feature_output = full_model.get_layer('head_ln').output
    feature_model  = keras.Model(
        inputs=full_model.input,
        outputs=feature_output,
        name=f'MedViTV2_{variant}_features'
    )
    return feature_model


# ══════════════════════════════════════════════════════════════════════
# QUICK SANITY CHECK
# ══════════════════════════════════════════════════════════════════════

if __name__ == '__main__':
    import os
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

    for variant in ['tiny', 'small']:
        print(f'\n{"="*60}')
        print(f'MedViT V2 — {variant.upper()}')
        print(f'{"="*60}')
        model = build_medvit_v2(
            input_shape=(224, 224, 3),
            num_classes=4,
            variant=variant,
        )
        model.summary(line_length=90)
        dummy = tf.random.uniform([2, 224, 224, 3])
        out   = model(dummy, training=False)
        print(f'Output shape: {out.shape}')
        total = model.count_params()
        print(f'Parameters: {total:,}  (~{total/1e6:.1f} M)')
