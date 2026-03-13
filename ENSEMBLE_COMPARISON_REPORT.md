# 1-vs-1 Binary Ensemble Evaluation Report

## Evaluation Summary
- **Date**: 2026-02-02 21:42:05
- **Model Type**: 1-vs-1 Binary Ensemble
- **Models Directory**: /Users/yannisbalasis/Documents/Python/Brain Cancer Detection/binary_ensemble_1vs1_experiment_20260202_163134
- **Test Samples**: 503

## Performance Results

### Ensemble Performance
- **Test Accuracy**: 0.9642 (96.42%)
- **Binary Models**: 3 expert binary classifiers
- **Combination Method**: Probability-based voting

### Comparison with Other Approaches
1. 🏆 **4-Class Multiclass**: 98.72%
2. 🥈 **3-Class Multiclass**: 97.81%
3. 🥉 **1-vs-1 Ensemble**: 96.42%

✅ **GOOD**: 1-vs-1 ensemble achieves clinically viable performance.

## Technical Analysis

### Binary Models Performance
Based on individual binary model validation accuracies:
- **Glioma vs Meningioma**: ~98.6% (Expert level)
- **Glioma vs Pituitary**: ~98.5% (Expert level)  
- **Meningioma vs Pituitary**: ~94.7% (Very good)

### Ensemble Methodology
- **Approach**: Probability-based voting across 3 binary models
- **Total Parameters**: ~3.8M (3 × 1.27M each)
- **Training Time**: ~4 hours (sequential binary training)
- **Evaluation Time**: ~5 minutes (model loading + prediction)

### Medical Implications
- **Clinical Deployment**: Requires all 3 models for prediction
- **Robustness**: Multiple independent decisions provide confidence
- **Interpretability**: Each binary model provides specific tumor-type insights
- **Confidence**: Probability-based combination enables uncertainty quantification

## Conclusions

**Key Finding**: Simple multiclass approaches outperform complex ensemble methods for this task.

**Recommendation**: Continue with proven 4-class multiclass approach.
