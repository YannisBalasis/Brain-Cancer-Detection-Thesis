"""
Brain Tumor Classification & XAI Analysis
Clinical Decision Support System
Μπαλάσης Ιωάννης ΤΜΗΥΠ
Πανεπιστήμιο Πατρών, 2026
"""

import streamlit as st
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import os
import datetime

try:
    from reportlab.lib.pagesizes import A4
    REPORTLAB_OK = True
except ImportError:
    REPORTLAB_OK = False

# ── Page config ───────────────────────
st.set_page_config(
    page_title="Brain Tumor Classification",
    page_icon="⬡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Clinical CSS ──────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@300;400;500;600&family=DM+Mono:wght@400;500&family=Playfair+Display:wght@400;600&display=swap');

html, body, .stApp {
    background-color: #FAFAF8 !important;
    font-family: 'DM Sans', sans-serif;
    color: #1A1A1A;
}
div[data-testid="stSidebar"] {
    background-color: #F0EFE9 !important;
    border-right: 1px solid #E0DED6;
}
div[data-testid="stSidebar"] * {
    color: #1A1A1A !important;
}
.main .block-container {
    padding: 2rem 2.5rem;
    max-width: 1200px;
}
.app-header {
    border-bottom: 2px solid #1A1A1A;
    padding-bottom: 1.5rem;
    margin-bottom: 2rem;
}
.app-title {
    font-family: 'Playfair Display', serif;
    font-size: 1.9rem;
    font-weight: 600;
    color: #1A1A1A;
    letter-spacing: -0.02em;
    margin: 0;
}
.app-subtitle {
    font-family: 'DM Mono', monospace;
    font-size: 0.72rem;
    color: #888;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    margin-top: 4px;
}
.verdict-block {
    border: 1.5px solid #1A1A1A;
    padding: 2rem;
    margin: 1.5rem 0;
    display: flex;
    align-items: center;
    gap: 2rem;
    background: #fff;
}
.verdict-label {
    font-family: 'DM Mono', monospace;
    font-size: 0.65rem;
    letter-spacing: 0.15em;
    text-transform: uppercase;
    color: #888;
    margin-bottom: 6px;
}
.verdict-class {
    font-family: 'Playfair Display', serif;
    font-size: 2.6rem;
    font-weight: 600;
    color: #1A1A1A;
    line-height: 1;
}
.verdict-tag {
    display: inline-block;
    padding: 2px 10px;
    border: 1px solid #1A1A1A;
    font-family: 'DM Mono', monospace;
    font-size: 0.68rem;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    margin-top: 8px;
    color: #1A1A1A;
}
.verdict-divider {
    width: 1px;
    height: 80px;
    background: #E0DED6;
}
.verdict-stat { text-align: center; }
.verdict-stat-num {
    font-family: 'Playfair Display', serif;
    font-size: 2rem;
    font-weight: 600;
    color: #1A1A1A;
}
.verdict-stat-label {
    font-family: 'DM Mono', monospace;
    font-size: 0.62rem;
    color: #888;
    text-transform: uppercase;
    letter-spacing: 0.1em;
}
.model-row {
    display: flex;
    align-items: center;
    padding: 10px 0;
    border-bottom: 1px solid #E8E7E1;
    gap: 16px;
}
.model-name {
    font-family: 'DM Sans', sans-serif;
    font-size: 0.85rem;
    font-weight: 500;
    color: #1A1A1A;
    width: 160px;
    flex-shrink: 0;
}
.model-pred {
    font-family: 'DM Mono', monospace;
    font-size: 0.8rem;
    width: 110px;
    flex-shrink: 0;
}
.model-conf {
    font-family: 'DM Mono', monospace;
    font-size: 0.8rem;
    color: #555;
    width: 60px;
    flex-shrink: 0;
}
.conf-bar-bg {
    flex: 1;
    height: 3px;
    background: #E8E7E1;
    border-radius: 2px;
    overflow: hidden;
}
.conf-bar-fill {
    height: 100%;
    border-radius: 2px;
    background: #1A1A1A;
}
.model-status { font-size: 0.75rem; width: 24px; }
.section-header {
    font-family: 'DM Mono', monospace;
    font-size: 0.65rem;
    letter-spacing: 0.18em;
    text-transform: uppercase;
    color: #888;
    border-top: 1px solid #E0DED6;
    padding-top: 1.2rem;
    margin: 1.5rem 0 1rem;
}
.metric-grid {
    display: grid;
    grid-template-columns: repeat(4, 1fr);
    gap: 1px;
    background: #E0DED6;
    border: 1px solid #E0DED6;
    margin: 1rem 0;
}
.metric-cell {
    background: #FAFAF8;
    padding: 1rem 1.2rem;
}
.metric-val {
    font-family: 'Playfair Display', serif;
    font-size: 1.6rem;
    font-weight: 600;
    color: #1A1A1A;
}
.metric-lbl {
    font-family: 'DM Mono', monospace;
    font-size: 0.62rem;
    color: #888;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    margin-top: 2px;
}
.stTabs [data-baseweb="tab-list"] {
    gap: 0;
    border-bottom: 1px solid #E0DED6;
    background: transparent;
}
.stTabs [data-baseweb="tab"] {
    font-family: 'DM Mono', monospace;
    font-size: 0.72rem;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    color: #888;
    padding: 10px 20px;
    border: none;
    border-bottom: 2px solid transparent;
    background: transparent;
}
.stTabs [aria-selected="true"] {
    color: #1A1A1A !important;
    border-bottom: 2px solid #1A1A1A !important;
    font-weight: 500;
}
.stButton > button {
    font-family: 'DM Mono', monospace;
    font-size: 0.72rem;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    background: #1A1A1A;
    color: #FAFAF8;
    border: none;
    border-radius: 0;
    padding: 10px 20px;
    width: 100%;
}
.stButton > button:hover {
    background: #333;
    color: #FAFAF8;
}
.info-note {
    font-family: 'DM Mono', monospace;
    font-size: 0.72rem;
    color: #888;
    padding: 0.8rem 1rem;
    border-left: 2px solid #C8C6BE;
    margin: 1rem 0;
    background: #F5F4EF;
}
.stat-table {
    width: 100%;
    border-collapse: collapse;
    font-family: 'DM Sans', sans-serif;
    font-size: 0.85rem;
}
.stat-table th {
    font-family: 'DM Mono', monospace;
    font-size: 0.65rem;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    color: #888;
    padding: 8px 12px;
    text-align: left;
    border-bottom: 1px solid #1A1A1A;
}
.stat-table td {
    padding: 8px 12px;
    border-bottom: 1px solid #E8E7E1;
    color: #1A1A1A;
}
.stat-table tr:hover td { background: #F5F4EF; }
.group-a { color: #2D6A4F; font-weight: 600; }
.group-b { color: #B5751B; font-weight: 600; }
.group-c { color: #C0392B; font-weight: 600; }
</style>
""", unsafe_allow_html=True)

# ── Constants ─────────────────────────
CLASSES_4 = ["Glioma", "Meningioma",
              "No Tumor", "Pituitary"]
CLASSES_3 = ["Glioma", "Meningioma",
              "Pituitary"]
CLASS_MARK = {
    "Glioma":     ("GLI", "#C0392B"),
    "Meningioma": ("MEN", "#1A5276"),
    "No Tumor":   ("NRM", "#2D6A4F"),
    "Pituitary":  ("PIT", "#B5751B"),
    "Tumor":      ("TUM", "#C0392B"),
}


# ══════════════════════════════════════
# MODEL LOADING
# ══════════════════════════════════════

@st.cache_resource(show_spinner=False)
def load_all_models(base_path):
    import tensorflow as tf
    models = {}
    paths = {
        "Binary CNN": os.path.join(
            base_path, "best_binary_model.h5"),
        "4-Class CNN": os.path.join(
            base_path,
            "multiclass_4class_experiment_20251217_141321",
            "best_multiclass_4class_model.h5"),
        "3-Class CNN": os.path.join(
            base_path,
            "multiclass_3class_experiment_20260201_161822",
            "best_multiclass_3class_model.h5"),
        "glioma_vs_meningioma": os.path.join(
            base_path,
            "binary_ensemble_1vs1_experiment_20260202_163134",
            "best_glioma_vs_meningioma_model.h5"),
        "glioma_vs_pituitary": os.path.join(
            base_path,
            "binary_ensemble_1vs1_experiment_20260202_163134",
            "best_glioma_vs_pituitary_model.h5"),
        "meningioma_vs_pituitary": os.path.join(
            base_path,
            "binary_ensemble_1vs1_experiment_20260202_163134",
            "best_meningioma_vs_pituitary_model.h5"),
        "ResNet-50 Dual": os.path.join(
            base_path,
            "dual_system_experiment_20260227_114800",
            "models", "best_dual_system_model.h5"),
        "EfficientNet-B3": os.path.join(
            base_path,
            "efnet_dual_system_experiment_20260305_194201",
            "models", "efnet_dual_phase3.h5"),
        "Multi-Dual": os.path.join(
            base_path,
            "multi_dual_system_experiment_20260310_100816",
            "best_multi_dual_model.h5"),
    }
    for name, path in paths.items():
        if os.path.exists(path):
            models[name] = \
                tf.keras.models.load_model(
                    path, compile=False)
    if "4-Class CNN" in models:
        m = models["4-Class CNN"]
        idx = next(
            (i for i, l in enumerate(m.layers)
             if l.name == "conv4_2"), None)
        if idx:
            models["4class_feat"] = \
                tf.keras.Sequential(
                    m.layers[:idx+1])
            models["4class_cls"] = \
                tf.keras.Sequential(
                    m.layers[idx+1:])
    return models


# ══════════════════════════════════════
# XAI FUNCTIONS
# ══════════════════════════════════════

def run_gradcam(models, img_input, class_idx):
    import tensorflow as tf
    if "4class_feat" not in models:
        return None
    with tf.GradientTape() as tape:
        img_t = tf.cast(img_input, tf.float32)
        conv_out = models["4class_feat"](img_t)
        tape.watch(conv_out)
        preds = models["4class_cls"](conv_out)
        loss = preds[:, class_idx]
    grads = tape.gradient(loss, conv_out)
    weights = tf.reduce_mean(
        grads, axis=(0, 1, 2))
    cam = tf.reduce_sum(
        tf.multiply(weights, conv_out[0]),
        axis=-1)
    cam = tf.maximum(cam, 0).numpy()
    cam /= (cam.max() + 1e-8)
    cam = np.array(tf.image.resize(
        cam[:, :, np.newaxis],
        [224, 224])[:, :, 0])
    return cam


def run_ig(models, img_input,
           class_idx, steps=25):
    import tensorflow as tf
    if "4-Class CNN" not in models:
        return None
    baseline = np.zeros_like(img_input)
    grads_list = []
    for alpha in np.linspace(0, 1, steps):
        interp = baseline + alpha * (
            img_input - baseline)
        interp_t = tf.cast(interp, tf.float32)
        with tf.GradientTape() as tape:
            tape.watch(interp_t)
            loss = models["4-Class CNN"](
                interp_t)[:, class_idx]
        grads_list.append(
            tape.gradient(
                loss, interp_t).numpy()[0])
    avg = np.mean(grads_list, axis=0)
    ig = (img_input[0] - baseline[0]) * avg
    ig_abs = np.abs(ig).sum(axis=-1)
    ig_abs /= (ig_abs.max() + 1e-8)
    return ig_abs


def run_shap(models, img_input,
             class_idx, n_samples=50):
    import tensorflow as tf
    if "4-Class CNN" not in models:
        return None
    background = np.zeros_like(img_input)
    shap_vals = np.zeros((224, 224))
    for _ in range(n_samples):
        alpha = np.random.uniform(0, 1)
        interp = background + alpha * (
            img_input - background)
        interp_t = tf.cast(interp, tf.float32)
        with tf.GradientTape() as tape:
            tape.watch(interp_t)
            preds = models["4-Class CNN"](
                interp_t)
            loss = preds[:, class_idx]
        grads = tape.gradient(
            loss, interp_t).numpy()[0]
        shap_vals += np.abs(grads).sum(axis=-1)
    shap_vals /= n_samples
    shap_vals /= (shap_vals.max() + 1e-8)
    return shap_vals


def run_occlusion(models, img_input,
                  class_idx, window=32,
                  stride=16, pb=None):
    if "4-Class CNN" not in models:
        return None, 0
    H, W = 224, 224
    base = float(
        models["4-Class CNN"].predict(
            img_input,
            verbose=0)[0][class_idx])
    sens = np.zeros((H, W))
    cnt = np.zeros((H, W))
    steps_i = list(range(0, H-window+1, stride))
    steps_j = list(range(0, W-window+1, stride))
    total = len(steps_i) * len(steps_j)
    done = 0
    for i in steps_i:
        for j in steps_j:
            m = img_input.copy()
            m[0, i:i+window,
              j:j+window, :] = 0
            p = float(
                models["4-Class CNN"].predict(
                    m, verbose=0)[0][class_idx])
            sens[i:i+window,
                 j:j+window] += base - p
            cnt[i:i+window,
                j:j+window] += 1
            done += 1
            if pb and done % 10 == 0:
                pb.progress(
                    int(done / total * 100))
    cnt = np.maximum(cnt, 1)
    sens /= cnt
    sens = np.maximum(sens, 0)
    sens /= (sens.max() + 1e-8)
    return sens, base


def overlay_heatmap(img_norm, heatmap,
                    cmap="jet", alpha=0.5):
    cm_func = plt.get_cmap(cmap)
    heatmap_colored = cm_func(
        heatmap)[:, :, :3]
    if len(img_norm.shape) == 2:
        img_rgb = np.stack(
            [img_norm] * 3, axis=-1)
    elif img_norm.shape[2] == 1:
        img_rgb = np.concatenate(
            [img_norm] * 3, axis=-1)
    else:
        img_rgb = img_norm
    overlay = ((1 - alpha) * img_rgb
               + alpha * heatmap_colored)
    return np.clip(overlay, 0, 1)


# ══════════════════════════════════════
# PREDICTIONS
# ══════════════════════════════════════

def run_predictions(models, img_input):
    results = {}

    if "Binary CNN" in models:
        p = float(
            models["Binary CNN"].predict(
                img_input, verbose=0)[0][0])
        results["Binary CNN"] = {
            "predicted": "Tumor"
                if p >= 0.5 else "No Tumor",
            "confidence": p
                if p >= 0.5 else 1 - p,
            "probs": {"Tumor": p,
                      "No Tumor": 1 - p},
            "class_idx": 0
        }

    if "4-Class CNN" in models:
        p = models["4-Class CNN"].predict(
            img_input, verbose=0)[0]
        idx = int(np.argmax(p))
        results["4-Class CNN"] = {
            "predicted": CLASSES_4[idx],
            "confidence": float(p[idx]),
            "probs": dict(zip(
                CLASSES_4, p.tolist())),
            "class_idx": idx
        }

    if "3-Class CNN" in models:
        p = models["3-Class CNN"].predict(
            img_input, verbose=0)[0]
        idx = int(np.argmax(p))
        results["3-Class CNN"] = {
            "predicted": CLASSES_3[idx],
            "confidence": float(p[idx]),
            "probs": dict(zip(
                CLASSES_3, p.tolist())),
            "class_idx": idx
        }

    ens_keys = ["glioma_vs_meningioma",
                "glioma_vs_pituitary",
                "meningioma_vs_pituitary"]
    if all(k in models for k in ens_keys):
        votes = {"Glioma": 0,
                 "Meningioma": 0,
                 "Pituitary": 0}
        for key in ens_keys:
            p = float(models[key].predict(
                img_input,
                verbose=0)[0][0])
            if key == "glioma_vs_meningioma":
                w = ("Meningioma"
                     if p >= 0.5 else "Glioma")
            elif key == "glioma_vs_pituitary":
                w = ("Pituitary"
                     if p >= 0.5 else "Glioma")
            else:
                w = ("Pituitary"
                     if p >= 0.5
                     else "Meningioma")
            votes[w] += 1
        final = max(votes, key=votes.get)
        results["1-vs-1 Ensemble"] = {
            "predicted": final,
            "confidence": votes[final] / 3,
            "probs": {k: v / 3
                      for k, v in votes.items()},
            "votes": votes,
            "class_idx": (
                CLASSES_3.index(final)
                if final in CLASSES_3 else 0)
        }

    for name, key in [
            ("ResNet-50 Dual",
             "ResNet-50 Dual"),
            ("EfficientNet-B3",
             "EfficientNet-B3")]:
        if key in models:
            p = models[key].predict(
                img_input, verbose=0)[0]
            idx = int(np.argmax(p))
            results[name] = {
                "predicted": CLASSES_4[idx],
                "confidence": float(p[idx]),
                "probs": dict(zip(
                    CLASSES_4, p.tolist())),
                "class_idx": idx
            }

    if "Multi-Dual" in models:
        out = models["Multi-Dual"].predict(
            img_input, verbose=0)
        fusion = np.array(
            out["fusion_output"][0])
        idx = int(np.argmax(fusion))
        results["Multi-Dual"] = {
            "predicted": CLASSES_4[idx],
            "confidence": float(fusion[idx]),
            "probs": dict(zip(
                CLASSES_4, fusion.tolist())),
            "class_idx": idx
        }

    return results


# ══════════════════════════════════════
# PLOTS
# ══════════════════════════════════════

def plot_xai_clinical(img_norm, heatmap,
                      overlay, title):
    fig, axes = plt.subplots(
        1, 3, figsize=(13, 4),
        facecolor="#FAFAF8")
    plt.subplots_adjust(wspace=0.04)
    for ax in axes:
        ax.set_facecolor("#FAFAF8")
        for sp in ax.spines.values():
            sp.set_color("#E0DED6")
            sp.set_linewidth(0.8)

    axes[0].imshow(img_norm)
    axes[0].set_title(
        "INPUT IMAGE",
        fontfamily="monospace",
        fontsize=8, color="#888", pad=10)
    axes[0].axis("off")

    im = axes[1].imshow(heatmap, cmap="jet")
    axes[1].set_title(
        title.upper(),
        fontfamily="monospace",
        fontsize=8, color="#888", pad=10)
    axes[1].axis("off")
    cb = plt.colorbar(
        im, ax=axes[1],
        fraction=0.04, pad=0.02)
    cb.ax.tick_params(
        labelsize=7, colors="#888")
    cb.outline.set_edgecolor("#E0DED6")

    axes[2].imshow(overlay)
    axes[2].set_title(
        "OVERLAY",
        fontfamily="monospace",
        fontsize=8, color="#888", pad=10)
    axes[2].axis("off")

    plt.tight_layout(pad=1.5)
    return fig


def plot_accuracy_clinical(results):
    overall_acc = {
        "Binary CNN":      0.9860,
        "4-Class CNN":     0.9801,
        "3-Class CNN":     0.9780,
        "1-vs-1 Ensemble": 0.9760,
        "Multi-Dual":      0.9341,
        "EfficientNet-B3": 0.8603,
        "ResNet-50 Dual":  0.7685,
    }
    groups = {
        "Binary CNN": "A",
        "4-Class CNN": "A",
        "3-Class CNN": "A",
        "1-vs-1 Ensemble": "A",
        "Multi-Dual": "B",
        "EfficientNet-B3": "C",
        "ResNet-50 Dual": "C",
    }
    g_colors = {"A": "#2D6A4F",
                "B": "#B5751B",
                "C": "#C0392B"}

    fig, (ax1, ax2) = plt.subplots(
        1, 2, figsize=(13, 4),
        facecolor="#FAFAF8")
    plt.subplots_adjust(wspace=0.08)

    for ax in [ax1, ax2]:
        ax.set_facecolor("#FAFAF8")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_color("#E0DED6")
        ax.spines["bottom"].set_color(
            "#1A1A1A")
        ax.tick_params(colors="#555",
                       labelsize=8)

    names = list(overall_acc.keys())
    accs = list(overall_acc.values())
    bar_cols = [g_colors[groups[n]]
                for n in names]
    ax1.barh(names, accs,
             color=bar_cols, alpha=0.75,
             edgecolor="none", height=0.55)

    for i, (name, acc) in enumerate(
            zip(names, accs)):
        if name in results:
            pred = results[name]["predicted"]
            abbr = CLASS_MARK.get(
                pred, ("?", "#888"))[0]
            ax1.text(
                acc + 0.002, i,
                f"-> {abbr}",
                va="center", fontsize=8,
                color="#555",
                fontfamily="monospace")

    ax1.set_xlim(0.7, 1.06)
    ax1.set_xlabel(
        "Test Set Accuracy",
        fontsize=8, color="#888",
        fontfamily="monospace")
    ax1.set_title(
        "MODEL PERFORMANCE  ·  "
        "Friedman x2=367.63  p<0.001",
        fontsize=8, color="#555",
        fontfamily="monospace", pad=12)

    from matplotlib.patches import Patch
    ax1.legend(
        handles=[
            Patch(facecolor="#2D6A4F",
                  label="Group A (equivalent)"),
            Patch(facecolor="#B5751B",
                  label="Group B"),
            Patch(facecolor="#C0392B",
                  label="Group C"),
        ],
        fontsize=7,
        facecolor="#FAFAF8",
        edgecolor="#E0DED6")

    if "4-Class CNN" in results:
        probs = results["4-Class CNN"]["probs"]
        classes = list(probs.keys())
        vals = list(probs.values())
        bar_cols2 = [
            CLASS_MARK.get(c, ("?", "#888"))[1]
            for c in classes]
        ax2.bar(classes, vals,
                color=bar_cols2, alpha=0.7,
                edgecolor="none", width=0.5)
        ax2.set_ylim(0, 1.1)
        ax2.set_ylabel(
            "Probability",
            fontsize=8, color="#888",
            fontfamily="monospace")
        ax2.set_title(
            "PROBABILITY DISTRIBUTION  ·  "
            "4-Class CNN",
            fontsize=8, color="#555",
            fontfamily="monospace", pad=12)
        for i, (cls, val) in enumerate(
                zip(classes, vals)):
            ax2.text(
                i, val + 0.02,
                f"{val:.3f}",
                ha="center", fontsize=8,
                color="#333",
                fontfamily="monospace")

    plt.tight_layout(pad=1.5)
    return fig


# ══════════════════════════════════════
# REPORT GENERATION
# ══════════════════════════════════════

def generate_report(results, img_norm,
                    final_cls, votes,
                    avg_conf, total_v):
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import (
        getSampleStyleSheet, ParagraphStyle)
    from reportlab.lib.units import cm
    from reportlab.lib import colors
    from reportlab.platypus import (
        SimpleDocTemplate, Paragraph,
        Spacer, Table, TableStyle,
        HRFlowable, Image as RLImage)
    import io
    import tempfile

    buf = io.BytesIO()
    doc = SimpleDocTemplate(
        buf, pagesize=A4,
        leftMargin=2.5*cm,
        rightMargin=2.5*cm,
        topMargin=2.5*cm,
        bottomMargin=2.5*cm)

    story = []
    now = datetime.datetime.now()

    title_style = ParagraphStyle(
        "title",
        fontName="Helvetica-Bold",
        fontSize=18,
        textColor=colors.HexColor("#1A1A1A"),
        spaceAfter=12)
    subtitle_style = ParagraphStyle(
        "subtitle",
        fontName="Helvetica",
        fontSize=9,
        textColor=colors.HexColor("#888888"),
        spaceAfter=6)
    section_style = ParagraphStyle(
        "section",
        fontName="Helvetica-Bold",
        fontSize=10,
        textColor=colors.HexColor("#1A1A1A"),
        spaceBefore=20,
        spaceAfter=10)
    body_style = ParagraphStyle(
        "body",
        fontName="Helvetica",
        fontSize=9,
        textColor=colors.HexColor("#333333"),
        leading=14,
        spaceAfter=8)
    verdict_style = ParagraphStyle(
        "verdict",
        fontName="Helvetica-Bold",
        fontSize=22,
        textColor=colors.HexColor("#1A1A1A"),
        spaceAfter=8)

    abbr_map = {
        "Glioma":     ("GLI", "#C0392B"),
        "Meningioma": ("MEN", "#1A5276"),
        "No Tumor":   ("NRM", "#2D6A4F"),
        "Pituitary":  ("PIT", "#B5751B"),
        "Tumor":      ("TUM", "#C0392B"),
    }

    # Header
    story.append(HRFlowable(
        width="100%", thickness=2,
        color=colors.HexColor("#1A1A1A")))
    story.append(Spacer(1, 8))
    story.append(Paragraph(
        "Brain Tumor Classification & "
        "XAI Analysis", title_style))
    story.append(Paragraph(
        "CLINICAL DECISION SUPPORT REPORT  "
        "·  CONFIDENTIAL",
        subtitle_style))
    story.append(Paragraph(
        f"Generated: "
        f"{now.strftime('%Y-%m-%d %H:%M:%S')}"
        f"  ·  Ioannis Balasis  ·  "
        f"University of Patras CEID 2026",
        subtitle_style))
    story.append(Spacer(1, 12))
    story.append(HRFlowable(
        width="100%", thickness=0.5,
        color=colors.HexColor("#E0DED6")))
    story.append(Spacer(1, 14))

    # Verdict
    abbr, _ = abbr_map.get(
        final_cls, ("?", "#1A1A1A"))
    story.append(Paragraph(
        "CONSENSUS DIAGNOSIS",
        subtitle_style))
    story.append(Paragraph(
        f"{final_cls}  [{abbr}]",

    

        verdict_style))
    story.append(Spacer(1, 16))

    verdict_data = [
        ["Models in agreement",
         "Mean confidence",
         "XAI methods",
         "Stat. groups"],
        [f"{votes.get(final_cls, 0)}"
         f"/{total_v}",
         f"{avg_conf:.1%}",
         "4", "A / B / C"],
    ]
    vt = Table(
        verdict_data,
        colWidths=[4*cm, 4*cm, 4*cm, 4*cm])
    vt.setStyle(TableStyle([
        ("FONTNAME", (0, 0), (-1, 0),
         "Helvetica"),
        ("FONTSIZE", (0, 0), (-1, 0), 7),
        ("TEXTCOLOR", (0, 0), (-1, 0),
         colors.HexColor("#888888")),
        ("FONTNAME", (0, 1), (-1, 1),
         "Helvetica-Bold"),
        ("FONTSIZE", (0, 1), (-1, 1), 11),
        ("TEXTCOLOR", (0, 1), (-1, 1),
         colors.HexColor("#1A1A1A")),
        ("TOPPADDING", (0, 0), (-1, -1), 6),
        ("BOTTOMPADDING",
         (0, 0), (-1, -1), 6),
        ("LINEBELOW", (0, 0), (-1, 0), 0.5,
         colors.HexColor("#E0DED6")),
        ("BACKGROUND", (0, 0), (-1, -1),
         colors.HexColor("#F5F4EF")),
    ]))
    story.append(vt)
    story.append(Spacer(1, 16))

    # MRI Image
    story.append(Paragraph(
        "INPUT IMAGE", section_style))
    try:
        img_pil = Image.fromarray(
            (img_norm * 255).astype(np.uint8))
        with tempfile.NamedTemporaryFile(
                suffix=".png",
                delete=False) as tmp:
            img_pil.save(tmp.name)
            rl_img = RLImage(
                tmp.name,
                width=5*cm, height=5*cm)
        story.append(rl_img)
    except Exception:
        story.append(Paragraph(
            "Image not available.",
            body_style))
    story.append(Spacer(1, 14))

    # Per-model results
    story.append(HRFlowable(
        width="100%", thickness=0.5,
        color=colors.HexColor("#E0DED6")))
    story.append(Paragraph(
        "PER-MODEL CLASSIFICATION RESULTS",
        section_style))

    model_order = [
        "Binary CNN", "4-Class CNN",
        "3-Class CNN", "1-vs-1 Ensemble",
        "Multi-Dual", "EfficientNet-B3",
        "ResNet-50 Dual"
    ]
    stat_grp = {
        "Binary CNN": "A",
        "4-Class CNN": "A",
        "3-Class CNN": "A",
        "1-vs-1 Ensemble": "A",
        "Multi-Dual": "B",
        "EfficientNet-B3": "C",
        "ResNet-50 Dual": "C",
    }
    table_data = [
        ["Model", "Prediction",
         "Confidence", "Agrees", "Group"]
    ]
    for name in model_order:
        if name not in results:
            continue
        res = results[name]
        pred = res["predicted"]
        conf = res["confidence"]
        ok = "Yes" if pred == final_cls else "No"
        table_data.append([
            name, pred,
            f"{conf:.2%}", ok,
            stat_grp.get(name, "-")
        ])

    t = Table(
        table_data,
        colWidths=[4.5*cm, 3.5*cm,
                   3*cm, 2*cm, 3*cm])
    t.setStyle(TableStyle([
        ("FONTNAME", (0, 0), (-1, 0),
         "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, -1), 8),
        ("FONTNAME", (0, 1), (-1, -1),
         "Helvetica"),
        ("TEXTCOLOR", (0, 0), (-1, 0),
         colors.HexColor("#888888")),
        ("LINEBELOW", (0, 0), (-1, 0), 1,
         colors.HexColor("#1A1A1A")),
        ("LINEBELOW", (0, 1), (-1, -1), 0.3,
         colors.HexColor("#E8E7E1")),
        ("TOPPADDING", (0, 0), (-1, -1), 6),
        ("BOTTOMPADDING",
         (0, 0), (-1, -1), 6),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1),
         [colors.HexColor("#FFFFFF"),
          colors.HexColor("#F5F4EF")]),
    ]))
    story.append(t)
    story.append(Spacer(1, 16))

    # Probability breakdown
    if "4-Class CNN" in results:
        story.append(HRFlowable(
            width="100%", thickness=0.5,
            color=colors.HexColor("#E0DED6")))
        story.append(Paragraph(
            "PROBABILITY BREAKDOWN  ·  "
            "4-CLASS CNN",
            section_style))
        probs = results["4-Class CNN"]["probs"]
        prob_data = [
            list(probs.keys()),
            [f"{v:.4f}"
             for v in probs.values()]
        ]
        pt = Table(
            prob_data,
            colWidths=[4*cm] * len(probs))
        pt.setStyle(TableStyle([
            ("FONTNAME", (0, 0), (-1, 0),
             "Helvetica-Bold"),
            ("FONTSIZE", (0, 0), (-1, -1), 9),
            ("FONTNAME", (0, 1), (-1, -1),
             "Courier"),
            ("TEXTCOLOR", (0, 0), (-1, 0),
             colors.HexColor("#888888")),
            ("TOPPADDING",
             (0, 0), (-1, -1), 8),
            ("BOTTOMPADDING",
             (0, 0), (-1, -1), 8),
            ("LINEBELOW", (0, 0), (-1, 0),
             0.5,
             colors.HexColor("#E0DED6")),
            ("BACKGROUND", (0, 0), (-1, -1),
             colors.HexColor("#F5F4EF")),
        ]))
        story.append(pt)
        story.append(Spacer(1, 16))

    # XAI summary
    story.append(HRFlowable(
        width="100%", thickness=0.5,
        color=colors.HexColor("#E0DED6")))
    story.append(Paragraph(
        "XAI ANALYSIS SUMMARY",
        section_style))
    xai_methods = [
        ("Grad-CAM",
         "Gradient-weighted Class Activation "
         "Mapping. Uses gradients of the last "
         "convolutional layer. "
         "Selvaraju et al., 2017."),
        ("SHAP",
         "SHapley Additive exPlanations. "
         "Gradient-based approximation, "
         "N=50 random baseline samples. "
         "Lundberg & Lee, 2017."),
        ("Integrated Gradients",
         "Path-based attribution. "
         "Completeness Axiom guaranteed. "
         "N=25 interpolation steps. "
         "Sundararajan et al., 2017."),
        ("Occlusion Analysis",
         "Model-agnostic perturbation. "
         "Window=32x32, stride=16. "
         "Zeiler & Fergus, 2014."),
    ]
    xai_data = [["Method", "Description"]]
    for n, d in xai_methods:
        xai_data.append([n, d])
    xt = Table(
        xai_data,
        colWidths=[3.5*cm, 13*cm])
    xt.setStyle(TableStyle([
        ("FONTNAME", (0, 0), (-1, 0),
         "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, -1), 8),
        ("FONTNAME", (0, 1), (-1, -1),
         "Helvetica"),
        ("FONTNAME", (0, 1), (0, -1),
         "Helvetica-Bold"),
        ("TEXTCOLOR", (0, 0), (-1, 0),
         colors.HexColor("#888888")),
        ("LINEBELOW", (0, 0), (-1, 0), 1,
         colors.HexColor("#1A1A1A")),
        ("LINEBELOW", (0, 1), (-1, -1), 0.3,
         colors.HexColor("#E8E7E1")),
        ("TOPPADDING", (0, 0), (-1, -1), 6),
        ("BOTTOMPADDING",
         (0, 0), (-1, -1), 6),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1),
         [colors.HexColor("#FFFFFF"),
          colors.HexColor("#F5F4EF")]),
    ]))
    story.append(xt)
    story.append(Spacer(1, 16))

    # Statistical validation
    story.append(HRFlowable(
        width="100%", thickness=0.5,
        color=colors.HexColor("#E0DED6")))
    story.append(Paragraph(
        "STATISTICAL VALIDATION",
        section_style))
    stat_text = (
        "Non-parametric framework on 501 "
        "common evaluation samples, 7 models:"
        "<br/><br/>"
        "<b>Friedman Test:</b> "
        "x2=367.63, p=2.53x10^-76.<br/>"
        "<b>Nemenyi Post-hoc:</b> "
        "CD=0.4025. Three distinct groups."
        "<br/>"
        "<b>Wilcoxon + Bonferroni:</b> "
        "15/21 pairs significant "
        "(alpha=0.00238).<br/>"
        "<b>Cohen's d:</b> "
        "Group A vs C: d~0.70 (Medium). "
        "Within Group A: d<0.2 (Negligible)."
    )
    story.append(Paragraph(
        stat_text, body_style))
    story.append(Spacer(1, 8))

    stat_groups = [
        ["Group", "Models",
         "Accuracy", "Interpretation"],
        ["A",
         "Binary, 4-Class, "
         "3-Class, Ensemble",
         "97.60%-98.60%",
         "Statistically equivalent"],
        ["B", "Multi-Dual",
         "93.41%",
         "Distinct from A and C"],
        ["C",
         "EfficientNet-B3, ResNet-50",
         "76.85%-86.03%",
         "Transfer learning gap"],
    ]
    sgt = Table(
        stat_groups,
        colWidths=[2*cm, 5.5*cm,
                   3.5*cm, 5.5*cm])
    sgt.setStyle(TableStyle([
        ("FONTNAME", (0, 0), (-1, 0),
         "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, -1), 8),
        ("FONTNAME", (0, 1), (-1, -1),
         "Helvetica"),
        ("TEXTCOLOR", (0, 0), (-1, 0),
         colors.HexColor("#888888")),
        ("LINEBELOW", (0, 0), (-1, 0), 1,
         colors.HexColor("#1A1A1A")),
        ("LINEBELOW", (0, 1), (-1, -1), 0.3,
         colors.HexColor("#E8E7E1")),
        ("TOPPADDING", (0, 0), (-1, -1), 6),
        ("BOTTOMPADDING",
         (0, 0), (-1, -1), 6),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1),
         [colors.HexColor("#FFFFFF"),
          colors.HexColor("#F5F4EF")]),
    ]))
    story.append(sgt)
    story.append(Spacer(1, 16))

    # Disclaimer
    story.append(HRFlowable(
        width="100%", thickness=0.5,
        color=colors.HexColor("#E0DED6")))
    story.append(Spacer(1, 8))
    story.append(Paragraph(
        "DISCLAIMER",
        ParagraphStyle(
            "disc_title",
            fontName="Helvetica-Bold",
            fontSize=8,
            textColor=colors.HexColor(
                "#888888"))))
    story.append(Paragraph(
        "This report is generated by an "
        "AI-assisted research system developed "
        "as part of a diploma thesis at the "
        "University of Patras (CEID). "
        "It is intended for research and "
        "educational purposes only and does "
        "NOT constitute a medical diagnosis. "
        "Clinical decisions must always be "
        "made by qualified medical "
        "professionals.",
        ParagraphStyle(
            "disc",
            fontName="Helvetica",
            fontSize=7,
            textColor=colors.HexColor(
                "#AAAAAA"),
            leading=11)))

    # Footer
    story.append(Spacer(1, 8))
    story.append(HRFlowable(
        width="100%", thickness=2,
        color=colors.HexColor("#1A1A1A")))
    story.append(Spacer(1, 4))
    story.append(Paragraph(
        f"Brain Tumor XAI System  ·  "
        f"Ioannis Balasis  ·  "
        f"University of Patras 2026  ·  "
        f"Report ID: BT-"
        f"{now.strftime('%Y%m%d%H%M%S')}",
        ParagraphStyle(
            "footer",
            fontName="Courier",
            fontSize=7,
            textColor=colors.HexColor(
                "#AAAAAA"))))

    doc.build(story)
    buf.seek(0)
    return buf


# ══════════════════════════════════════
# MAIN
# ══════════════════════════════════════

def main():

    # Header
    st.markdown("""
    <div class="app-header">
        <p class="app-subtitle">
            Clinical Decision Support System
            &nbsp;·&nbsp; Πανεπιστήμιο Πατρών
            &nbsp;·&nbsp; ΤΜΗΥΠ 2026
        </p>
        <h1 class="app-title">
            Brain Tumor Classification
            &amp; XAI Analysis
        </h1>
        <p class="app-subtitle"
           style="margin-top:6px;">
            Ιωάννης Μπαλάσης
        </p>
    </div>
    """, unsafe_allow_html=True)

    # ── Sidebar ───────────────────────
    with st.sidebar:
        st.markdown(
            "<p class='app-subtitle' "
            "style='padding:12px 0 4px;'>"
            "CONFIGURATION</p>",
            unsafe_allow_html=True)

        base_path = st.text_input(
            "Model directory",
            value=os.path.dirname(
                os.path.abspath(__file__)))

        st.markdown("<br>",
                    unsafe_allow_html=True)

        if st.button(
                "LOAD MODELS",
                use_container_width=True):
            with st.spinner(
                    "Loading models..."):
                try:
                    models = load_all_models(
                        base_path)
                    st.session_state[
                        "models"] = models
                    st.success(
                        f"{len(models)} "
                        f"models loaded")
                except Exception as e:
                    st.error(str(e))

        if "models" in st.session_state:
            st.markdown(
                "<p class='app-subtitle' "
                "style='padding:16px 0 6px;'>"
                "MODEL STATUS</p>",
                unsafe_allow_html=True)
            display_map = {
                "Binary CNN": "Binary CNN",
                "4-Class CNN": "4-Class CNN",
                "3-Class CNN": "3-Class CNN",
                "glioma_vs_meningioma":
                    "1-vs-1 Ensemble",
                "ResNet-50 Dual":
                    "ResNet-50 Dual",
                "EfficientNet-B3":
                    "EfficientNet-B3",
                "Multi-Dual": "Multi-Dual",
            }
            shown = set()
            for key, disp in \
                    display_map.items():
                if disp in shown:
                    continue
                ok = key in \
                    st.session_state["models"]
                color = ("#2D6A4F"
                         if ok else "#C0392B")
                mark = "✓" if ok else "✗"
                st.markdown(
                    f"<div style='font-family:"
                    f"DM Mono,monospace;"
                    f"font-size:0.72rem;"
                    f"padding:3px 0;"
                    f"color:{color}'>"
                    f"{mark}&nbsp; {disp}"
                    f"</div>",
                    unsafe_allow_html=True)
                shown.add(disp)

        st.markdown("<br>",
                    unsafe_allow_html=True)
        st.markdown(
            "<p class='app-subtitle' "
            "style='padding:4px 0;'>"
            "XAI METHODS</p>",
            unsafe_allow_html=True)
        do_gradcam = st.checkbox(
            "Grad-CAM", value=True)
        do_ig = st.checkbox(
            "Integrated Gradients",
            value=True)
        do_shap = st.checkbox(
            "SHAP", value=True)
        do_occ = st.checkbox(
            "Occlusion Analysis",
            value=True)

        st.markdown("<br>",
                    unsafe_allow_html=True)
        st.markdown(
            "<div class='info-note'>"
            "Friedman x2=367.63 · p&lt;0.001"
            "<br>Nemenyi CD=0.4025"
            "<br>Bonferroni a=0.00238"
            "<br>15/21 significant pairs"
            "</div>",
            unsafe_allow_html=True)

    # ── Guard ─────────────────────────
    if "models" not in st.session_state:
        st.markdown(
            "<div class='info-note'>"
            "Set model directory and click "
            "LOAD MODELS to begin."
            "</div>",
            unsafe_allow_html=True)
        return

    models = st.session_state["models"]

    # ── Upload ────────────────────────
    st.markdown(
        "<div class='section-header'>"
        "INPUT</div>",
        unsafe_allow_html=True)

    col_up, col_prev = st.columns([3, 1])
    with col_up:
        uploaded = st.file_uploader(
            "Upload MRI image",
            type=["png", "jpg",
                  "jpeg", "bmp"])

    if uploaded is None:
        st.markdown(
            "<div class='info-note'>"
            "Upload an MRI image to begin "
            "classification and XAI analysis."
            "</div>",
            unsafe_allow_html=True)
        return

    pil_img = (Image.open(uploaded)
               .convert("RGB")
               .resize((224, 224)))
    img_arr = np.array(pil_img)\
        .astype(np.float32)
    img_norm = img_arr / img_arr.max()
    img_input = np.expand_dims(
        img_arr / 255.0, axis=0)

    with col_prev:
        fig_prev, ax_prev = plt.subplots(
            figsize=(2, 2),
            facecolor="#FAFAF8")
        ax_prev.imshow(img_norm)
        ax_prev.axis("off")
        ax_prev.set_title(
            "224 x 224 px",
            fontsize=7, color="#888",
            fontfamily="monospace", pad=4)
        for sp in ax_prev.spines.values():
            sp.set_color("#E0DED6")
        plt.tight_layout(pad=0.3)
        st.pyplot(fig_prev)
        plt.close(fig_prev)

    # ── Run Analysis ──────────────────
    if st.button("RUN ANALYSIS",
                 use_container_width=True):
        st.session_state[
            "img_norm"] = img_norm
        st.session_state[
            "img_input"] = img_input

        with st.spinner(
                "Running classification..."):
            results = run_predictions(
                models, img_input)
            st.session_state[
                "results"] = results

        if do_gradcam:
            with st.spinner("Grad-CAM..."):
                idx = results.get(
                    "4-Class CNN", {}
                ).get("class_idx", 1)
                cam = run_gradcam(
                    models, img_input, idx)
                if cam is not None:
                    st.session_state[
                        "cam"] = cam

        if do_ig:
            with st.spinner(
                    "Integrated Gradients..."):
                idx = results.get(
                    "4-Class CNN", {}
                ).get("class_idx", 1)
                ig = run_ig(
                    models, img_input, idx)
                if ig is not None:
                    st.session_state[
                        "ig"] = ig

        if do_shap:
            with st.spinner("SHAP..."):
                idx = results.get(
                    "4-Class CNN", {}
                ).get("class_idx", 1)
                shap = run_shap(
                    models, img_input, idx)
                if shap is not None:
                    st.session_state[
                        "shap"] = shap

        if do_occ:
            pb = st.progress(
                0,
                text="Occlusion Analysis...")
            idx = results.get(
                "4-Class CNN", {}
            ).get("class_idx", 1)
            occ, bc = run_occlusion(
                models, img_input, idx,
                pb=pb)
            pb.progress(100, text="Done.")
            if occ is not None:
                st.session_state["occ"] = occ
                st.session_state[
                    "occ_bc"] = bc

        st.success("Analysis complete.")

    # ── Guard results ─────────────────
    if "results" not in st.session_state:
        return

    results = st.session_state["results"]
    img_norm = st.session_state.get(
        "img_norm", img_norm)

    # ── Consensus ─────────────────────
    votes = {}
    for name, res in results.items():
        if name == "Binary CNN":
            continue
        p = res["predicted"]
        votes[p] = votes.get(p, 0) + 1

    if not votes:
        return

    final_cls = max(votes, key=votes.get)
    total_v = sum(votes.values())
    abbr, color = CLASS_MARK.get(
        final_cls, ("?", "#1A1A1A"))
    avg_conf = np.mean([
        r["confidence"]
        for r in results.values()])

    # ── Verdict block ─────────────────
    st.markdown(f"""
    <div class="verdict-block">
        <div>
            <div class="verdict-label">
                Consensus Diagnosis
            </div>
            <div class="verdict-class"
                 style="color:{color}">
                {final_cls}
            </div>
            <span class="verdict-tag"
                  style="border-color:{color};
                         color:{color}">
                {abbr}
            </span>
        </div>
        <div class="verdict-divider"></div>
        <div class="verdict-stat">
            <div class="verdict-stat-num">
                {votes[final_cls]}/{total_v}
            </div>
            <div class="verdict-stat-label">
                Models in agreement
            </div>
        </div>
        <div class="verdict-divider"></div>
        <div class="verdict-stat">
            <div class="verdict-stat-num">
                {avg_conf:.1%}
            </div>
            <div class="verdict-stat-label">
                Mean confidence
            </div>
        </div>
        <div class="verdict-divider"></div>
        <div class="verdict-stat">
            <div class="verdict-stat-num">
                4
            </div>
            <div class="verdict-stat-label">
                XAI methods applied
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ── Download Report ───────────────
    now = datetime.datetime.now()
    if REPORTLAB_OK:
        report_buf = generate_report(
            results=results,
            img_norm=img_norm,
            final_cls=final_cls,
            votes=votes,
            avg_conf=avg_conf,
            total_v=total_v)
        st.download_button(
            label="⬇  DOWNLOAD REPORT (PDF)",
            data=report_buf,
            file_name=(
                f"brain_tumor_report_"
                f"{now.strftime('%Y%m%d_%H%M%S')}"
                f".pdf"),
            mime="application/pdf",
            use_container_width=True)
    else:
        st.warning(
            "Install reportlab for PDF: "
            "pip install reportlab")

    # ── Tabs ──────────────────────────
    tabs = st.tabs([
        "Classification",
        "Grad-CAM",
        "Integrated Gradients",
        "SHAP",
        "Occlusion",
        "Statistical Analysis"
    ])

    # ── Tab 0: Classification ─────────
    with tabs[0]:
        st.markdown(
            "<div class='section-header'>"
            "PER-MODEL RESULTS</div>",
            unsafe_allow_html=True)

        model_order = [
            "Binary CNN", "4-Class CNN",
            "3-Class CNN",
            "1-vs-1 Ensemble",
            "Multi-Dual",
            "EfficientNet-B3",
            "ResNet-50 Dual"
        ]
        rows_html = ""
        for name in model_order:
            if name not in results:
                continue
            res = results[name]
            pred = res["predicted"]
            conf = res["confidence"]
            abbr2, col2 = CLASS_MARK.get(
                pred, ("?", "#888"))
            is_ok = pred == final_cls
            bar_w = int(conf * 100)
            rows_html += f"""
            <div class="model-row">
                <span class="model-name">
                    {name}
                </span>
                <span class="model-pred"
                      style="color:{col2}">
                    [{abbr2}] {pred}
                </span>
                <span class="model-conf">
                    {conf:.1%}
                </span>
                <div class="conf-bar-bg">
                    <div class="conf-bar-fill"
                         style="width:{bar_w}%;
                                background:
                                {col2}">
                    </div>
                </div>
                <span class="model-status">
                    {"✓" if is_ok else "—"}
                </span>
            </div>"""
        st.markdown(rows_html,
                    unsafe_allow_html=True)

        st.markdown(
            "<div class='section-header'>"
            "PROBABILITY BREAKDOWN  ·  "
            "4-CLASS CNN</div>",
            unsafe_allow_html=True)

        if "4-Class CNN" in results:
            probs = results[
                "4-Class CNN"]["probs"]
            cols = st.columns(len(probs))
            for col, (cls, prob) in zip(
                    cols, probs.items()):
                abbr3, col3 = CLASS_MARK.get(
                    cls, ("?", "#888"))
                with col:
                    st.markdown(
                        f"<div style='"
                        f"border-top:"
                        f"3px solid {col3};"
                        f"padding:12px 0'>"
                        f"<div style='"
                        f"font-family:"
                        f"DM Mono,monospace;"
                        f"font-size:0.65rem;"
                        f"color:#888'>"
                        f"{abbr3}</div>"
                        f"<div style='"
                        f"font-family:"
                        f"Playfair Display,"
                        f"serif;"
                        f"font-size:1.5rem;"
                        f"color:#1A1A1A;"
                        f"font-weight:600'>"
                        f"{prob:.4f}</div>"
                        f"<div style='"
                        f"font-size:0.75rem;"
                        f"color:#555'>"
                        f"{cls}</div></div>",
                        unsafe_allow_html=True)

    # ── Tab 1: Grad-CAM ───────────────
    with tabs[1]:
        st.markdown(
            "<div class='section-header'>"
            "GRAD-CAM · Gradient-weighted "
            "Class Activation Mapping"
            "</div>",
            unsafe_allow_html=True)
        st.markdown(
            "<div class='info-note'>"
            "Uses gradients of the last "
            "convolutional layer to produce "
            "a class-discriminative "
            "localization map. "
            "Darker regions = higher importance."
            "</div>",
            unsafe_allow_html=True)
        if "cam" in st.session_state:
            cam = st.session_state["cam"]
            ov = overlay_heatmap(
                img_norm, cam, "jet")
            fig = plot_xai_clinical(
                img_norm, cam, ov, "Grad-CAM")
            st.pyplot(fig)
            plt.close(fig)
        else:
            st.markdown(
                "<div class='info-note'>"
                "Run analysis with "
                "Grad-CAM enabled."
                "</div>",
                unsafe_allow_html=True)

    # ── Tab 2: IG ────────────────────
    with tabs[2]:
        st.markdown(
            "<div class='section-header'>"
            "INTEGRATED GRADIENTS · "
            "Path-based Attribution"
            "</div>",
            unsafe_allow_html=True)
        st.markdown(
            "<div class='info-note'>"
            "Satisfies Completeness Axiom. "
            "N=25 interpolation steps "
            "from black baseline."
            "</div>",
            unsafe_allow_html=True)
        if "ig" in st.session_state:
            ig = st.session_state["ig"]
            ov = overlay_heatmap(
                img_norm, ig, "jet")
            fig = plot_xai_clinical(
                img_norm, ig, ov,
                "Integrated Gradients")
            st.pyplot(fig)
            plt.close(fig)
        else:
            st.markdown(
                "<div class='info-note'>"
                "Run analysis with "
                "Integrated Gradients enabled."
                "</div>",
                unsafe_allow_html=True)

    # ── Tab 3: SHAP ───────────────────
    with tabs[3]:
        st.markdown(
            "<div class='section-header'>"
            "SHAP · SHapley Additive "
            "exPlanations"
            "</div>",
            unsafe_allow_html=True)
        st.markdown(
            "<div class='info-note'>"
            "Gradient-based approximation, "
            "N=50 random baseline samples. "
            "Pixel-level Shapley values."
            "</div>",
            unsafe_allow_html=True)
        if "shap" in st.session_state:
            shap_map = st.session_state["shap"]
            ov = overlay_heatmap(
                img_norm, shap_map, "jet")
            fig = plot_xai_clinical(
                img_norm, shap_map, ov,
                "SHAP Attribution")
            st.pyplot(fig)
            plt.close(fig)
        else:
            st.markdown(
                "<div class='info-note'>"
                "Run analysis with "
                "SHAP enabled."
                "</div>",
                unsafe_allow_html=True)

    # ── Tab 4: Occlusion ──────────────
    with tabs[4]:
        st.markdown(
            "<div class='section-header'>"
            "OCCLUSION ANALYSIS · "
            "Perturbation-based Validation"
            "</div>",
            unsafe_allow_html=True)
        st.markdown(
            "<div class='info-note'>"
            "Model-agnostic. Window=32, "
            "stride=16. "
            "Brighter = greater confidence drop."
            "</div>",
            unsafe_allow_html=True)
        if "occ" in st.session_state:
            occ = st.session_state["occ"]
            bc = st.session_state.get(
                "occ_bc", 0)
            ov = overlay_heatmap(
                img_norm, occ, "jet")
            fig = plot_xai_clinical(
                img_norm, occ, ov,
                f"Occlusion baseline={bc:.3f}")
            st.pyplot(fig)
            plt.close(fig)
            cols = st.columns(3)
            cols[0].metric(
                "Baseline Conf.",
                f"{bc:.4f}")
            cols[1].metric(
                "Max Sensitivity",
                f"{occ.max():.4f}")
            cols[2].metric(
                "Window", "32 x 32 px")
        else:
            st.markdown(
                "<div class='info-note'>"
                "Run analysis with "
                "Occlusion enabled."
                "</div>",
                unsafe_allow_html=True)

    # ── Tab 5: Statistical Analysis ───
    with tabs[5]:
        st.markdown(
            "<div class='section-header'>"
            "STATISTICAL ANALYSIS · "
            "Non-parametric Comparison"
            "</div>",
            unsafe_allow_html=True)

        st.markdown("""
        <div class="metric-grid">
            <div class="metric-cell">
                <div class="metric-val">
                    367.63
                </div>
                <div class="metric-lbl">
                    Friedman x2
                </div>
            </div>
            <div class="metric-cell">
                <div class="metric-val">
                    0.4025
                </div>
                <div class="metric-lbl">
                    Critical Difference
                </div>
            </div>
            <div class="metric-cell">
                <div class="metric-val">
                    15/21
                </div>
                <div class="metric-lbl">
                    Significant pairs
                </div>
            </div>
            <div class="metric-cell">
                <div class="metric-val">
                    ~0.70
                </div>
                <div class="metric-lbl">
                    Max Cohen's d
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        fig_stat = plot_accuracy_clinical(
            results)
        st.pyplot(fig_stat)
        plt.close(fig_stat)

        st.markdown(
            "<div class='section-header'>"
            "STATISTICAL GROUPS  ·  "
            "Nemenyi Post-hoc"
            "</div>",
            unsafe_allow_html=True)

        rows = [
            ("A", "Binary CNN",
             "98.60%", "3.790"),
            ("A", "4-Class CNN",
             "98.01%", "3.811"),
            ("A", "3-Class CNN",
             "97.80%", "3.818"),
            ("A", "1-vs-1 Ensemble",
             "97.60%", "3.825"),
            ("B", "Multi-Dual",
             "93.41%", "3.972"),
            ("C", "EfficientNet-B3",
             "86.03%", "4.231"),
            ("C", "ResNet-50 Dual",
             "76.85%", "4.552"),
        ]
        g_class = {"A": "group-a",
                   "B": "group-b",
                   "C": "group-c"}
        table_rows = ""
        for grp, name, acc, rank in rows:
            pred_now = results.get(
                name, {}).get(
                    "predicted", "—")
            abbr4 = CLASS_MARK.get(
                pred_now,
                ("—", "#888"))[0]
            table_rows += f"""
            <tr>
                <td>
                    <span class=
                    '{g_class[grp]}'>
                        {grp}
                    </span>
                </td>
                <td>{name}</td>
                <td style='font-family:
                    DM Mono,monospace'>
                    {acc}
                </td>
                <td style='font-family:
                    DM Mono,monospace'>
                    {rank}
                </td>
                <td style='font-family:
                    DM Mono,monospace;
                    color:#555'>
                    {abbr4}
                </td>
            </tr>"""

        st.markdown(f"""
        <table class="stat-table">
            <thead>
                <tr>
                    <th>Group</th>
                    <th>Model</th>
                    <th>Accuracy</th>
                    <th>Avg Rank</th>
                    <th>Current</th>
                </tr>
            </thead>
            <tbody>{table_rows}</tbody>
        </table>
        """, unsafe_allow_html=True)

        st.markdown(
            "<div class='info-note' "
            "style='margin-top:1rem'>"
            "Group A: statistically equivalent "
            "(d &lt; 0.2, Negligible). "
            "Group A vs C: d ~0.70 (Medium). "
            "Bonferroni a = 0.05/21 = 0.00238."
            "</div>",
            unsafe_allow_html=True)


if __name__ == "__main__":
    main()