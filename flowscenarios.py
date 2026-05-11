import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

fig, axes = plt.subplots(1, 3, figsize=(14, 6))
fig.patch.set_facecolor('white')

# ── Χρώματα κλάσεων ──────────────────
COLORS = {
    'tumor':      '#E74C3C',
    'no_tumor':   '#2ECC71',
    'glioma':     '#E74C3C',
    'meningioma': '#3498DB',
    'pituitary':  '#F39C12',
}

# ── Τίτλοι και κλάσεις ───────────────
scenarios = [
    {
        'title':    'Binary CNN',
        'subtitle': '(Δυαδική Ταξινόμηση)',
        'classes':  [
            ('Tumor\n(Όγκος)',    '#E74C3C'),
            ('No Tumor\n(Υγιής)', '#2ECC71'),
        ],
        'color': '#E8F8F5'
    },
    {
        'title':    '3-Class CNN',
        'subtitle': '(Τριαδική Ταξινόμηση)',
        'classes':  [
            ('Glioma\n(Γλοίωμα)',         '#E74C3C'),
            ('Meningioma\n(Μηνιγγίωμα)', '#3498DB'),
            ('Pituitary\n(Υπόφυση)',      '#F39C12'),
        ],
        'color': '#EBF5FB'
    },
    {
        'title':    '4-Class CNN',
        'subtitle': '(Τετραδική Ταξινόμηση)',
        'classes':  [
            ('Glioma\n(Γλοίωμα)',         '#E74C3C'),
            ('Meningioma\n(Μηνιγγίωμα)', '#3498DB'),
            ('Pituitary\n(Υπόφυση)',      '#F39C12'),
            ('No Tumor\n(Υγιής)',         '#2ECC71'),
        ],
        'color': '#FEF9E7'
    }
]

for ax, scenario in zip(axes, scenarios):
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.axis('off')
    ax.set_facecolor(scenario['color'])

    # ── Τίτλος ───────────────────────
    ax.text(5, 9.2, scenario['title'],
            ha='center', va='center',
            fontsize=14, fontweight='bold',
            color='#2C3E50')
    ax.text(5, 8.6, scenario['subtitle'],
            ha='center', va='center',
            fontsize=10, color='#7F8C8D',
            style='italic')

    # ── Input box ────────────────────
    input_box = mpatches.FancyBboxPatch(
        (2.5, 7.3), 5, 0.9,
        boxstyle="round,pad=0.15",
        facecolor='#2C3E50',
        edgecolor='#1A252F',
        linewidth=1.5
    )
    ax.add_patch(input_box)
    ax.text(5, 7.75, 'MRI Image Input',
            ha='center', va='center',
            fontsize=10, fontweight='bold',
            color='white')

    # ── Arrow down ───────────────────
    ax.annotate('',
        xy=(5, 6.9), xytext=(5, 7.3),
        arrowprops=dict(
            arrowstyle='->', 
            color='#2C3E50', lw=2
        )
    )

    # ── CNN box ──────────────────────
    cnn_colors = {
        'Binary CNN':  '#8E44AD',
        '3-Class CNN': '#8E44AD',
        '4-Class CNN': '#8E44AD'
    }
    cnn_box = mpatches.FancyBboxPatch(
        (2.5, 6.0), 5, 0.85,
        boxstyle="round,pad=0.15",
        facecolor='#8E44AD',
        edgecolor='#6C3483',
        linewidth=1.5
    )
    ax.add_patch(cnn_box)
    ax.text(5, 6.42, scenario['title'],
            ha='center', va='center',
            fontsize=10, fontweight='bold',
            color='white')

    # ── Arrows to classes ────────────
    n = len(scenario['classes'])
    
    # Υπολόγισε x positions
    if n == 2:
        xs = [2.8, 7.2]
    elif n == 3:
        xs = [1.5, 5.0, 8.5]
    else:  # n == 4
        xs = [1.0, 3.6, 6.4, 9.0]

    y_arrow_start = 6.0
    y_arrow_end   = 4.8
    y_box         = 3.2

    for i, ((cls_name, cls_color), x) \
            in enumerate(zip(scenario['classes'], xs)):

        # Βέλος
        ax.annotate('',
            xy=(x, y_arrow_end),
            xytext=(5, y_arrow_start),
            arrowprops=dict(
                arrowstyle='->',
                color=cls_color,
                lw=1.8,
                connectionstyle='arc3,rad=0'
            )
        )

        # Class box
        box_w = 2.2 if n <= 3 else 1.8
        box_h = 1.4

        class_box = mpatches.FancyBboxPatch(
            (x - box_w/2, y_box - box_h/2),
            box_w, box_h,
            boxstyle="round,pad=0.15",
            facecolor=cls_color,
            edgecolor='#2C3E50',
            linewidth=1.5,
            alpha=0.9
        )
        ax.add_patch(class_box)

        ax.text(x, y_box,
                cls_name,
                ha='center', va='center',
                fontsize=8 if n == 4 else 9,
                fontweight='bold',
                color='white',
                linespacing=1.4)

    # ── Border ───────────────────────
    for spine in ['top', 'bottom', 
                  'left', 'right']:
        ax.spines[spine].set_visible(True)
        ax.spines[spine].set_linewidth(2)
        ax.spines[spine].set_color('#BDC3C7')

# ── Τίτλος figure ────────────────────


# ── Labels κάτω ──────────────────────
labels = ['(α) Δυαδική', 
          '(β) Τριαδική', 
          '(γ) Τετραδική']
for ax, label in zip(axes, labels):
    ax.text(5, 0.4, label,
            ha='center', va='center',
            fontsize=10, color='#7F8C8D',
            fontweight='bold',
            transform=ax.transData)

plt.tight_layout(pad=1.5)
plt.savefig(
    './thesis_figures/classification_scenarios.png',
    dpi=300,
    bbox_inches='tight',
    facecolor='white'
)
plt.close()
print("Saved: classification_scenarios.png")