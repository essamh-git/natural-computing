# Diagram of the solution encodings for the two mini-project problems
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

BLUE, ORANGE = '#2a78d6', '#eb6834'
INK, MUTED, SURF = '#0b0b0b', '#52514e', '#fcfcfb'

A = [('Conv\nblocks', '2–5'), ('First\nfilters', '16/32/64'), ('Kernel\nsize', '3/5/7'),
     ('Dense\nunits', '64–512'), ('Dropout', '0–0.5'), ('Learning\nrate', '1e-4–3e-3'),
     ('Batch\nsize', '16/32/64'), ('Batch\nnorm', 'on/off'), ('Optimiser', 'Adam/SGD/\nRMSprop')]
B = [('Short MA', '5–50'), ('Long MA', '20–200'), ('RSI\nperiod', '7–30'), ('RSI buy', '10–40'),
     ('RSI sell', '60–90'), ('Stop-loss\n%', '1–10'), ('Take-profit\n%', '2–20'), ('Position\nsize %', '10–100')]

fig, ax = plt.subplots(figsize=(11, 4.0))
fig.patch.set_facecolor(SURF); ax.set_facecolor(SURF)
ax.set_xlim(0, 9.4); ax.set_ylim(0, 4.0); ax.axis('off')

def row(genes, y, col, title, sub):
    ax.text(0.05, y + 1.25, title, fontsize=11, color=INK, weight='bold', va='bottom')
    ax.text(0.05, y + 1.05, sub, fontsize=8.5, color=MUTED, va='bottom')
    for i, (name, rng) in enumerate(genes):
        x = 0.05 + i * 1.03
        ax.add_patch(FancyBboxPatch((x, y), 0.95, 0.95, boxstyle='round,pad=0,rounding_size=0.06',
                                    fc=col, ec=SURF, lw=2, alpha=0.9))
        ax.text(x + 0.475, y + 0.62, name, ha='center', va='center', fontsize=8, color='white',
                weight='bold')
        ax.text(x + 0.475, y + 0.2, rng, ha='center', va='center', fontsize=7, color='white')

row(A, 2.45, BLUE, 'Problem 1: CNN hyperparameters (9 integer genes)',
    'Each gene is an index into a list of allowed values  |  ≈ 6.2 × 10⁴ combinations')
row(B, 0.15, ORANGE, 'Problem 2: Trading strategy parameters (8 integer genes)',
    'Each gene is a value within its range  |  ≈ 5.1 × 10¹¹ valid combinations')
fig.savefig('fig9_encodings.png', dpi=150, bbox_inches='tight', facecolor=SURF)