# Figures for the Week 3 Part 3 wiki page (reads dfo_parameter_study.npz)
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

r = np.load('dfo_parameter_study.npz')
DIMS = [2, 10, 30, 50, 100]
DELTAS = [0.0, 0.0001, 0.001, 0.01, 0.1]
BLUE, ORANGE = '#2a78d6', '#eb6834'
INK, MUTED, GRID, SURF = '#0b0b0b', '#52514e', '#e4e3df', '#fcfcfb'
plt.rcParams.update({'axes.edgecolor': MUTED, 'axes.labelcolor': INK,
                     'xtick.color': MUTED, 'ytick.color': MUTED,
                     'axes.spines.top': False, 'axes.spines.right': False,
                     'figure.facecolor': SURF, 'axes.facecolor': SURF})
COL = {'Sphere': BLUE, 'Rastrigin': ORANGE}
FLOOR = 1e-22


def final(name, D, dl):
    return np.maximum(r[f'{name}_D{D}_delta{dl}_final'], FLOOR)


def strip(ax, xs_labels, getter, xlabel, title, legend_loc='upper left'):
    rng = np.random.default_rng(0)
    for k, name in enumerate(['Sphere', 'Rastrigin']):
        meds = []
        for j, key in enumerate(xs_labels):
            v = getter(name, key)
            x0 = j + (k - 0.5) * 0.36
            ax.scatter(x0 + rng.uniform(-0.08, 0.08, len(v)), v, s=14, color=COL[name],
                       alpha=0.55, lw=0, zorder=3)
            ax.hlines(np.median(v), x0 - 0.13, x0 + 0.13, color=COL[name], lw=2.2, zorder=4)
            meds.append((x0, np.median(v)))
        ax.plot(*zip(*meds), color=COL[name], lw=1.2, alpha=0.8, label=f'{name} (median)')
    ax.set_yscale('log')
    ax.set_xticks(range(len(xs_labels)))
    ax.set_xticklabels([str(x) for x in xs_labels])
    ax.set_xlabel(xlabel); ax.set_ylabel('Final best fitness (log scale)')
    ax.set_title(title, color=INK)
    ax.grid(axis='y', color=GRID); ax.set_axisbelow(True)
    ax.set_ylim(bottom=1e-25)
    ax.legend(fontsize=8, loc=legend_loc)
    ax.text(0.01, 0.02, 'Values below 10⁻²² (including exact zeros) are shown at 10⁻²²',
            transform=ax.transAxes, fontsize=7.5, color=MUTED)


# Figure 1: impact of dimensionality on final fitness
fig, ax = plt.subplots(figsize=(8, 4.6))
strip(ax, DIMS, lambda n, D: final(n, D, 0.001), 'Dimensionality (D)',
      'Impact of dimensionality (Δ = 0.001, 15 runs each)')
fig.tight_layout(); fig.savefig('fig4_dimensionality.png', dpi=150, bbox_inches='tight')

# Figure 2: convergence for each dimensionality (median of 15 runs)
fig, axes = plt.subplots(1, 2, figsize=(11, 4.3), sharey=True)
shades = ['#9ec5f4', '#6da7ec', '#2a78d6', '#1c5cab', '#0d366b']
for ax, name in zip(axes, ['Sphere', 'Rastrigin']):
    for D, c in zip(DIMS, shades):
        med = np.median(np.maximum(r[f'{name}_D{D}_delta0.001_curves'], FLOOR), axis=0)
        ax.plot(med, color=c, lw=1.8, label=f'D = {D}')
    ax.set_yscale('log')
    ax.set_xlabel('Iteration'); ax.set_title(name, color=INK)
    ax.grid(color=GRID); ax.set_axisbelow(True)
axes[0].set_ylabel('Best fitness (median, log scale)')
axes[1].legend(fontsize=8, loc='upper left', bbox_to_anchor=(1.02, 1))
fig.suptitle('Convergence slows as dimensionality grows', color=INK)
fig.tight_layout(); fig.savefig('fig5_dimensionality_convergence.png', dpi=150, bbox_inches='tight')

# Figure 3: impact of the disturbance threshold on final fitness
fig, ax = plt.subplots(figsize=(8, 4.6))
strip(ax, DELTAS, lambda n, dl: final(n, 30, dl), 'Disturbance threshold (Δ)',
      'Impact of the disturbance threshold (D = 30, 15 runs each)', legend_loc='lower right')
fig.tight_layout(); fig.savefig('fig6_disturbance.png', dpi=150, bbox_inches='tight')