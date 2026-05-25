import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.calibration import (
    calibration_curve)
from sklearn.metrics import (
    brier_score_loss)

# ── Load ──────────────────────────────
X_test = np.load(
    './multiclass_4class_experiment_20251217_141321/X_test.npy')
y_test = np.load(
    './multiclass_4class_experiment_20251217_141321/y_test.npy')

X_test_norm = X_test / 255.0
y_true_onehot = y_test
y_true = np.argmax(y_test, axis=1)
classes = ['Glioma', 'Meningioma',
           'No Tumor', 'Pituitary']
colors = ['#E74C3C', '#3498DB',
          '#2ECC71', '#F39C12']

model = tf.keras.models.load_model(
    './multiclass_4class_experiment_20251217_141321/'
    'best_multiclass_4class_model.h5',
    compile=False)

y_pred_prob = model.predict(
    X_test_norm, verbose=0)

# ── Reliability Diagram ───────────────
fig, axes = plt.subplots(
    2, 2, figsize=(12, 10))
fig.patch.set_facecolor('white')
fig.suptitle(
    'Calibration Analysis — 4-Class CNN\n'
    'Reliability Diagrams (One-vs-Rest)',
    fontsize=13, fontweight='bold',
    color='#2C3E50')

for i, (cls, color) in enumerate(
        zip(classes, colors)):
    ax = axes[i//2, i%2]
    ax.set_facecolor('#FAFAFA')

    # Binary: class i vs rest
    y_bin = (y_true == i).astype(int)
    prob_i = y_pred_prob[:, i]

    # Calibration curve
    fraction_pos, mean_pred = \
        calibration_curve(
            y_bin, prob_i,
            n_bins=10,
            strategy='uniform')

    # Brier Score
    brier = brier_score_loss(
        y_bin, prob_i)

    # Perfect calibration line
    ax.plot([0, 1], [0, 1],
            'k--', lw=1.5,
            alpha=0.6,
            label='Τέλεια βαθμονόμηση')

    # Calibration curve
    ax.plot(mean_pred, fraction_pos,
            color=color, lw=2.5,
            marker='o', markersize=6,
            label=f'Μοντέλο '
                  f'(Brier={brier:.4f})')

    # Fill between
    ax.fill_between(
        mean_pred, fraction_pos,
        mean_pred,
        alpha=0.15, color=color)

    ax.set_title(
        f'{cls}',
        fontsize=12, fontweight='bold',
        color='#2C3E50')
    ax.set_xlabel(
        'Μέση Προβλεπόμενη Πιθανότητα',
        fontsize=10, color='#555')
    ax.set_ylabel(
        'Κλάσμα Θετικών',
        fontsize=10, color='#555')
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3,
            linestyle='--')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # Brier score annotation
    ax.text(
        0.05, 0.92,
        f'Brier Score: {brier:.4f}',
        transform=ax.transAxes,
        fontsize=10,
        color=color,
        fontweight='bold')

    print(f'{cls}: Brier Score = '
          f'{brier:.4f}')

plt.tight_layout(pad=2.0)
plt.savefig(
    './thesis_figures/'
    'calibration_reliability.png',
    dpi=300, bbox_inches='tight',
    facecolor='white')
plt.close()
print('\nSaved: calibration_reliability.png')

# ── Confidence Histogram ──────────────
fig2, axes2 = plt.subplots(
    2, 2, figsize=(12, 8))
fig2.patch.set_facecolor('white')
fig2.suptitle(
    'Κατανομή Εμπιστοσύνης '
    'ανά Κλάση — 4-Class CNN',
    fontsize=13, fontweight='bold',
    color='#2C3E50')

for i, (cls, color) in enumerate(
        zip(classes, colors)):
    ax = axes2[i//2, i%2]
    ax.set_facecolor('#FAFAFA')

    # Confidence για σωστές/λάθος
    mask_correct = (y_true == i) & \
        (np.argmax(y_pred_prob, axis=1) == i)
    mask_wrong = (y_true == i) & \
        (np.argmax(y_pred_prob, axis=1) != i)

    conf_correct = y_pred_prob[
        mask_correct, i]
    conf_wrong = y_pred_prob[
        mask_wrong, i]

    ax.hist(conf_correct,
            bins=20, alpha=0.7,
            color=color,
            label=f'Σωστές '
                  f'(n={len(conf_correct)})',
            edgecolor='white')

    if len(conf_wrong) > 0:
        ax.hist(conf_wrong,
                bins=20, alpha=0.7,
                color='#E0E0E0',
                label=f'Λάθος '
                      f'(n={len(conf_wrong)})',
                edgecolor='white')

    mean_conf = np.mean(conf_correct) \
        if len(conf_correct) > 0 else 0

    ax.axvline(x=mean_conf,
               color='black',
               linestyle='--', lw=1.5,
               label=f'Μέση={mean_conf:.3f}')

    ax.set_title(
        f'{cls}',
        fontsize=12, fontweight='bold',
        color='#2C3E50')
    ax.set_xlabel(
        'Εμπιστοσύνη',
        fontsize=10, color='#555')
    ax.set_ylabel(
        'Πλήθος',
        fontsize=10, color='#555')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3,
            linestyle='--')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

plt.tight_layout(pad=2.0)
plt.savefig(
    './thesis_figures/'
    'calibration_confidence.png',
    dpi=300, bbox_inches='tight',
    facecolor='white')
plt.close()
print('Saved: calibration_confidence.png')