import pandas as pd
import matplotlib.pyplot as plt

# ── Paths ─────────────────────────────
BASE = './binary_ensemble_1vs1_experiment'\
       '_20260202_163134/'

logs = {
    'Glioma vs Meningioma': 
        BASE + 'glioma_vs_meningioma'
              '_training_log.csv',
    'Glioma vs Pituitary':  
        BASE + 'glioma_vs_pituitary'
              '_training_log.csv',
    'Meningioma vs Pituitary': 
        BASE + 'meningioma_vs_pituitary'
              '_training_log.csv'
}

fig, axes = plt.subplots(
    1, 3, figsize=(18, 5))
fig.patch.set_facecolor('white')
fig.suptitle(
    'Learning Curves — 1-vs-1 Ensemble '
    '(3 Sub-Models)',
    fontsize=14, fontweight='bold',
    color='#2C3E50', y=1.02
)

colors = ['#E74C3C', '#3498DB', '#27AE60']

for ax, (name, path), color in \
        zip(axes, logs.items(), colors):
    df = pd.read_csv(path)
    epochs = df['epoch'] + 1
    best   = df['val_accuracy'].idxmax()

    ax.plot(epochs, df['accuracy'],
            color='#2980B9', lw=2,
            label='Train Acc',
            marker='o', markersize=2)
    ax.plot(epochs, df['val_accuracy'],
            color='#E74C3C', lw=2,
            label='Val Acc',
            marker='s', markersize=2,
            linestyle='--')
    ax.axvline(x=best + 1,
               color='#27AE60',
               linestyle=':', lw=1.8,
               label=f'Best ({best+1})')
    ax.set_title(name, fontsize=10,
                 fontweight='bold',
                 color='#2C3E50')
    ax.set_xlabel('Epoch', fontsize=9)
    ax.set_ylabel('Accuracy', fontsize=9)
    ax.set_ylim(0, 1.05)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3,
            linestyle='--')
    ax.set_facecolor('#FAFAFA')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    print(f"{name}: best val_acc = "
          f"{df['val_accuracy'][best]:.4f} "
          f"at epoch {best+1}")

plt.tight_layout(pad=2.0)
plt.savefig(
    './thesis_figures/'
    'learning_curves_ensemble.png',
    dpi=300,
    bbox_inches='tight',
    facecolor='white'
)
plt.close()
print("Saved!")