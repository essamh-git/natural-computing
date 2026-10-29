# Figures for the Week 5 wiki page (reads week5_results.npz made by GA_vs_DFO.py)
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

r = np.load('week5_results.npz')
COL = {'GA': '#eb6834', 'DFO': '#2a78d6'}
INK, MUTED, GRID, SURF = '#0b0b0b', '#52514e', '#e4e3df', '#fcfcfb'
plt.rcParams.update({'axes.edgecolor': MUTED, 'axes.labelcolor': INK,
                     'xtick.color': MUTED, 'ytick.color': MUTED,
                     'axes.spines.top': False, 'axes.spines.right': False,
                     'figure.facecolor': SURF, 'axes.facecolor': SURF})
FLOOR = 1e-22
FUNCS = ['Sphere', 'Rastrigin']

# Figure 1: convergence, median of 30 runs with best-to-worst band
fig, axes = plt.subplots(1, 2, figsize=(11, 4.3))
for ax, fn in zip(axes, FUNCS):
    for alg in ('GA', 'DFO'):
        c = np.maximum(r[f'{alg}_{fn}'], FLOOR)
        it = np.arange(c.shape[1])
        med = np.median(c, axis=0)
        ax.fill_between(it, c.min(0), c.max(0), color=COL[alg], alpha=0.15, lw=0)
        ax.plot(it, med, color=COL[alg], lw=2, label=alg)
        ax.text(it[-1] + 15, med[-1], f'{alg}\n{med[-1]:.2g}', va='center', fontsize=8, color=INK)
    ax.set_yscale('log')
    ax.set_xlim(0, 1150)
    ax.set_xlabel('Iteration (100 evaluations each)')
    ax.set_title(fn, color=INK)
    ax.grid(color=GRID); ax.set_axisbelow(True)
    ax.legend(fontsize=8, loc='lower left')
axes[0].set_ylabel('Best cost so far (log scale)')
fig.suptitle('GA vs DFO, 30 dimensions, median of 30 runs (shaded = best to worst run)', color=INK)
fig.tight_layout(); fig.savefig('fig1_ga_vs_dfo_convergence.png', dpi=150, bbox_inches='tight')

# Figure 2: final cost of every run
fig, axes = plt.subplots(1, 2, figsize=(9, 4.2))
rng = np.random.default_rng(0)
for ax, fn in zip(axes, FUNCS):
    for k, alg in enumerate(('GA', 'DFO')):
        v = np.maximum(r[f'{alg}_{fn}'][:, -1], FLOOR)
        ax.scatter(k + rng.uniform(-0.12, 0.12, len(v)), v, s=26, color=COL[alg],
                   edgecolor=SURF, lw=0.8, zorder=3)
        ax.hlines(np.median(v), k - 0.25, k + 0.25, color=INK, lw=2, zorder=4)
        ax.text(k + 0.28, np.median(v), f'median\n{np.median(v):.2g}', va='center', fontsize=8,
                color=INK)
    ax.set_yscale('log')
    ax.set_xticks([0, 1]); ax.set_xticklabels(['GA', 'DFO'])
    ax.set_xlim(-0.5, 1.8)
    ax.set_title(fn, color=INK)
    ax.grid(axis='y', color=GRID); ax.set_axisbelow(True)
axes[0].set_ylabel('Final best cost (log scale)')
fig.suptitle('Final result of each of the 30 runs', color=INK)
fig.tight_layout(); fig.savefig('fig2_ga_vs_dfo_final.png', dpi=150, bbox_inches='tight')