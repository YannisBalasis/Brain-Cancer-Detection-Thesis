import numpy as np
import tensorflow as tf
from sklearn.metrics import (
    confusion_matrix,
    roc_auc_score)

X_test = np.load(
    './multiclass_4class_experiment_20251217_141321/X_test.npy')
y_test = np.load(
    './multiclass_4class_experiment_20251217_141321/y_test.npy')

y_true = np.argmax(y_test, axis=1)
classes = ['Glioma', 'Meningioma',
           'No Tumor', 'Pituitary']

model = tf.keras.models.load_model(
    './multiclass_4class_experiment_20251217_141321/'
    'best_multiclass_4class_model.h5',
    compile=False)

# ── Διόρθωση normalization ────────────
print(f"X_test range: {X_test.min():.1f} - {X_test.max():.1f}")

# Normalize στο [0,1]
X_test_norm = X_test / 255.0

y_pred_prob = model.predict(
    X_test_norm, verbose=1)
y_pred = np.argmax(y_pred_prob, axis=1)

print(f"\nPrediction distribution:")
for i, cls in enumerate(classes):
    print(f"  {cls}: {np.sum(y_pred==i)}")

cm = confusion_matrix(y_true, y_pred)
print(f"\nConfusion Matrix:")
print(cm)

for i, cls in enumerate(classes):
    TP = cm[i, i]
    FP = cm[:, i].sum() - TP
    FN = cm[i, :].sum() - TP
    TN = cm.sum() - TP - FP - FN

    sensitivity = TP / (TP + FN) \
        if (TP + FN) > 0 else 0
    specificity = TN / (TN + FP) \
        if (TN + FP) > 0 else 0
    ppv = TP / (TP + FP) \
        if (TP + FP) > 0 else 0
    npv = TN / (TN + FN) \
        if (TN + FN) > 0 else 0
    f1 = 2*TP / (2*TP + FP + FN) \
        if (2*TP + FP + FN) > 0 else 0

    print(f"\n{cls}:")
    print(f"  Sensitivity: {sensitivity:.4f}")
    print(f"  Specificity: {specificity:.4f}")
    print(f"  PPV:         {ppv:.4f}")
    print(f"  NPV:         {npv:.4f}")
    print(f"  F1-Score:    {f1:.4f}")
    print(f"  TP:{TP} FP:{FP} "
          f"FN:{FN} TN:{TN}")

auc = roc_auc_score(
    y_test, y_pred_prob,
    multi_class='ovr',
    average='macro')
print(f"\nMacro AUC: {auc:.4f}")