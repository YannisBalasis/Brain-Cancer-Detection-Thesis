# Brain Tumor Classification & XAI Analysis

> **Diploma Thesis** — Department of Computer Engineering & Informatics (CEID)  
> University of Patras, 2026  
> **Author:** Ioannis Balasis

---

## Overview

Deep Learning system for brain tumor classification from MRI images, combining **7 neural network architectures** with **4 XAI (Explainable AI) methods** and rigorous **statistical validation**.

The system classifies MRI brain scans into 4 categories:
- 🔴 **Glioma**
- 🔵 **Meningioma**
- 🟢 **No Tumor**
- 🟡 **Pituitary Adenoma**

---

## Dataset

**Kaggle Brain Tumor MRI Dataset**  
- 7,023 MRI images
- 4 classes: Glioma, Meningioma, No Tumor, Pituitary
- Input size: 224 × 224 × 3

---

## Architectures (7 Models)

| Model | Type | Parameters | Test Accuracy |
|-------|------|-----------|---------------|
| Binary CNN | Custom | 1.44M | 98.60% |
| 4-Class CNN | Custom | 1.28M | 98.01% |
| 3-Class CNN | Custom | 1.28M | 97.80% |
| 1-vs-1 Ensemble | Custom ×3 | ~3.8M | 97.60% |
| Multi-Dual System | Custom | 1.68M | 93.41% |
| EfficientNet-B3 Dual | Transfer Learning | 12.8M | 86.03% |
| ResNet-50 Dual | Transfer Learning | 26.1M | 76.85% |

---

## XAI Methods

| Method | Type | Description |
|--------|------|-------------|
| **Grad-CAM** | Gradient-based | Class activation maps from last conv layer |
| **SHAP** | Shapley values | Pixel-level attribution, N=50 samples |
| **Integrated Gradients** | Path-based | Completeness Axiom guaranteed, N=25 steps |
| **Occlusion Analysis** | Perturbation | Window=32, stride=16, model-agnostic |

---

## Statistical Validation

Non-parametric framework applied to **501 common evaluation samples**:

| Test | Result |
|------|--------|
| Friedman Test | χ²=367.63, p=2.53×10⁻⁷⁶ |
| Nemenyi CD | CD=0.4025 |
| Wilcoxon + Bonferroni | 15/21 pairs significant (α=0.00238) |
| Max Cohen's d | d≈0.70 (Medium effect) |

**Statistical Groups:**
- **Group A** (equivalent): Binary, 4-Class, 3-Class, Ensemble → 97.60%–98.60%
- **Group B**: Multi-Dual → 93.41%
- **Group C**: EfficientNet-B3, ResNet-50 → 76.85%–86.03%

---

## Key Finding: The Meningioma Paradox

> All models correctly localize Meningioma tumors at the pixel level (confirmed by all 4 XAI methods), yet show lower classification accuracy compared to other classes. This indicates a **classification boundary problem** in feature space, not a localization failure.

---

## Clinical Application

A **Streamlit-based** Clinical Decision Support System that integrates all 7 models and 4 XAI methods with PDF report generation.

### Run the app

```bash
# Install dependencies
pip install streamlit tensorflow pillow matplotlib reportlab

# Run
streamlit run brain_tumor_clinical.py
```

### Features
- Upload MRI image → instant classification by all 7 models
- Consensus diagnosis with confidence scores
- Interactive XAI visualizations (Grad-CAM, SHAP, IG, Occlusion)
- Statistical analysis dashboard
- Downloadable PDF clinical report

---

## Project Structure

```
Brain Cancer Detection/
│
├── best_binary_model.h5
│
├── multiclass_4class_experiment_20251217_141321/
│   └── best_multiclass_4class_model.h5
│
├── multiclass_3class_experiment_20260201_161822/
│   └── best_multiclass_3class_model.h5
│
├── binary_ensemble_1vs1_experiment_20260202_163134/
│   ├── best_glioma_vs_meningioma_model.h5
│   ├── best_glioma_vs_pituitary_model.h5
│   └── best_meningioma_vs_pituitary_model.h5
│
├── dual_system_experiment_20260227_114800/
│   └── models/best_dual_system_model.h5
│
├── efnet_dual_system_experiment_20260305_194201/
│   └── models/efnet_dual_phase3.h5
│
├── multi_dual_system_experiment_20260310_100816/
│   └── best_multi_dual_model.h5
│
├── statistical_analysis_results/
│   ├── statistical_results.json
│   ├── accuracy_comparison.png
│   ├── cd_diagram.png
│   ├── cohens_d_heatmap.png
│   ├── nemenyi_heatmap.png
│   └── wilcoxon_heatmap.png
│
├── xai_results/
│   ├── gradcam/
│   ├── shap/
│   ├── integrated_gradients/
│   └── occlusion/
│
├── thesis_figures/
│   ├── case_study_overview.png
│   ├── case_study_gradcam.png
│   └── case_study_occlusion.png
│
└── brain_tumor_clinical.py   ← Streamlit app
```

---

## Requirements

```
tensorflow>=2.10
streamlit>=1.30
numpy
pillow
matplotlib
scikit-learn
reportlab
scipy
```

Install:
```bash
pip install -r requirements.txt
```

---

## Hardware

Trained on **Apple M3 Pro** (no discrete GPU):
- Binary/4-Class/3-Class CNN: ~45–50 min
- 1-vs-1 Ensemble: ~2.5 hours
- ResNet-50 Dual: ~3 hours (3 phases)
- EfficientNet-B3 Dual: ~2 hours (3 phases)
- Multi-Dual System: ~1.5 hours

---

## References

- Selvaraju et al. (2017) — Grad-CAM
- Lundberg & Lee (2017) — SHAP
- Sundararajan et al. (2017) — Integrated Gradients
- He et al. (2016) — ResNet-50
- Tan & Le (2019) — EfficientNet
- Demšar (2006) — Statistical comparison of classifiers
- Tjoa & Guan (2021) — XAI in healthcare

---

## License

This project is part of a diploma thesis at the University of Patras.  
For academic and educational use only.  
**Not intended for clinical diagnosis.**

---

<div align="center">
  <sub>
    Ioannis Balasis · University of Patras CEID · 2026
  </sub>
</div>
