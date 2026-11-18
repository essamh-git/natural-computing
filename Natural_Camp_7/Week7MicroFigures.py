# Figures for the Week 7 Part 2 page (reads week7_micro_results.npz made by week7_micro_experiments.py)
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

r = np.load('week7_micro_results.npz')
INK, MUTED, GRID, SURF = '#0b0b0b', '#52514e', '#e4e3df', '#fcfcfb'
COL = {'pixel_exact': '#9ec5f4', 'pixel_tol': '#2a78d6', 'patch_tol': '#eb6834', 'pixel_pair': '#1baf7a'}
LAB = {'pixel_exact': 'One pixel, exact', 'pixel_tol': 'One pixel, ±12', 'patch_tol': '3×3 block, ±12',
       'pixel_pair': 'Pixel pair (which is brighter)'}
CONDS = [('Exact', 'Exact copy'), ('Noisy', 'Noisy (sd 10)'), ('Brighter', 'Brighter (+40)')]
plt.rcParams.update({'axes.edgecolor': MUTED, 'axes.labelcolor': INK,
                     'xtick.color': MUTED, 'ytick.color': MUTED,
                     'axes.spines.top': False, 'axes.spines.right': False,
                     'figure.facecolor': SURF, 'axes.facecolor': SURF})
TX, TY = r['target']
model = r['model']

# Figure 1: the search space, the model and the micro-features
fig, axes = plt.subplots(1, 5, figsize=(14, 3.3), gridspec_kw={'width_ratios': [1, 1, 1, 1, 1.05]})
for ax, (key, title) in zip(axes[:3], CONDS):
    ax.imshow(r[f'img_{key}' if key == 'Exact' else f'img_{title}'], cmap='gray', vmin=0, vmax=255)
    ax.add_patch(Rectangle((TX - 0.5, TY - 0.5), 16, 16, fill=False, ec='#eb6834', lw=1.5))
    ax.set_title(f'Search space: {title.lower()}', fontsize=9, color=INK)
    ax.set_xticks([]); ax.set_yticks([])
axes[3].imshow(model, cmap='gray', vmin=0, vmax=255)
axes[3].set_title('Model (16×16 pixels)', fontsize=9, color=INK)
axes[3].set_xticks([]); axes[3].set_yticks([])
ax = axes[4]
ax.imshow(model, cmap='gray', vmin=0, vmax=255, alpha=0.45)
ax.add_patch(Rectangle((2.5, 11.5), 1, 1, fill=False, ec=COL['pixel_tol'], lw=2.2))
ax.text(4.2, 12.2, 'one pixel', color=COL['pixel_tol'], fontsize=8, va='center', weight='bold')
ax.add_patch(Rectangle((9.5, 1.5), 3, 3, fill=False, ec=COL['patch_tol'], lw=2.2))
ax.text(9.2, 5.9, '3×3 block', color=COL['patch_tol'], fontsize=8, weight='bold')
p1, p2 = (3, 4), (12, 9)
ax.plot(*zip(p1, p2), color=COL['pixel_pair'], lw=1.6, ls='--')
ax.scatter(*zip(p1, p2), s=40, color=COL['pixel_pair'], zorder=3)
ax.text(6.0, 9.6, 'pixel pair', color=COL['pixel_pair'], fontsize=8, weight='bold', rotation=-27)
ax.set_title('Micro-features of the model', fontsize=9, color=INK)
ax.set_xticks([]); ax.set_yticks([])
fig.tight_layout(); fig.savefig('fig6_micro_setup.png', dpi=150, bbox_inches='tight')

# Figure 2: how often each micro-feature found the face
fig, ax = plt.subplots(figsize=(9, 4))
w = 0.2
for j, k in enumerate(COL):
    vals = [100 * np.mean(r[f'{c}_{k}_first'] > 0) for c, _ in CONDS]
    bars = ax.bar(np.arange(3) + (j - 1.5) * w, vals, w * 0.92, color=COL[k], label=LAB[k])
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v + 1.5, f'{v:.0f}', ha='center', fontsize=7.5, color=INK)
ax.set_xticks(range(3)); ax.set_xticklabels([t for _, t in CONDS])
ax.set_ylabel('Runs that found the face (%)'); ax.set_ylim(0, 112)
ax.set_title('Which micro-feature finds the small image? (30 runs each, 1,500 iterations)',
             fontsize=10, color=INK)
ax.legend(fontsize=8, ncol=4, loc='upper center', bbox_to_anchor=(0.5, -0.1), frameon=False)
ax.grid(axis='y', color=GRID); ax.set_axisbelow(True)
fig.tight_layout(); fig.savefig('fig7_micro_success.png', dpi=150, bbox_inches='tight')

# Figure 3: size of the cluster on the face over time
fig, axes = plt.subplots(1, 3, figsize=(13, 3.8), sharey=True)
for ax, (c, title) in zip(axes, CONDS):
    for k in COL:
        S = r[f'{c}_{k}_size']
        ax.plot(np.arange(1, S.shape[1] + 1), S.mean(0), color=COL[k], lw=1.8, label=LAB[k])
    ax.set_title(title, fontsize=10, color=INK); ax.set_xlabel('Iteration'); ax.set_ylim(0, 100)
    ax.grid(color=GRID); ax.set_axisbelow(True)
axes[0].set_ylabel('Agents on the face (%, mean of 30 runs)')
axes[0].legend(fontsize=7.5, loc='lower right')
fig.suptitle('How quickly a cluster forms on the face', color=INK)
fig.tight_layout(); fig.savefig('fig8_micro_cluster.png', dpi=150, bbox_inches='tight')