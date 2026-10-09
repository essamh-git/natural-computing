# Figures for the Week 3 DFO wiki page (reads dfo_results.npz made by dfo_experiment.py)
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

r = np.load('dfo_results.npz')
BLUE, ORANGE = '#2a78d6', '#eb6834'
INK, MUTED, GRID, SURF = '#0b0b0b', '#52514e', '#e4e3df', '#fcfcfb'
plt.rcParams.update({'axes.edgecolor': MUTED, 'axes.labelcolor': INK,
                     'xtick.color': MUTED, 'ytick.color': MUTED,
                     'axes.spines.top': False, 'axes.spines.right': False,
                     'figure.facecolor': SURF, 'axes.facecolor': SURF})
COL = {'Sphere': BLUE, 'Rastrigin': ORANGE}
FLOOR = 1e-22   # PLOTTING FLOOR SO EXACT ZEROS CAN BE SHOWN ON A LOG SCALE

# Figure 1: convergence, median of 30 runs with min-max band
fig, ax = plt.subplots(figsize=(8, 4.5))
it = np.arange(r['Sphere_curves'].shape[1])
for name in ['Sphere', 'Rastrigin']:
    c = np.maximum(r[f'{name}_curves'], FLOOR)
    med = np.median(c, axis=0)
    ax.fill_between(it, c.min(0), c.max(0), color=COL[name], alpha=0.15, lw=0)
    ax.plot(it, med, color=COL[name], lw=2, label=f'{name} (median)')
    ax.text(it[-1] + 15, med[-1], f'{name}\n{med[-1]:.1e}', va='center', fontsize=8, color=INK)
ax.set_yscale('log')
ax.set_xlim(0, it[-1] + 140)
ax.set_xlabel('Iteration'); ax.set_ylabel('Best fitness (log scale)')
ax.set_title('DFO convergence over 30 runs (shaded = best to worst run)', color=INK)
ax.grid(color=GRID); ax.set_axisbelow(True)
ax.legend(fontsize=8, loc='lower left')
fig.tight_layout(); fig.savefig('fig1_dfo_convergence.png', dpi=150, bbox_inches='tight')

# Figure 2: final fitness of each of the 30 runs
fig, ax = plt.subplots(figsize=(6.5, 4.5))
rng = np.random.default_rng(0)
for k, name in enumerate(['Sphere', 'Rastrigin']):
    v = r[f'{name}_final']
    xs = k + rng.uniform(-0.12, 0.12, len(v))
    ax.scatter(xs, v, s=30, color=COL[name], edgecolor=SURF, lw=0.8, zorder=3)
    ax.hlines(np.median(v), k - 0.25, k + 0.25, color=INK, lw=2, zorder=4)
    ax.text(k + 0.28, np.median(v), f'median\n{np.median(v):.1e}', va='center', fontsize=8, color=INK)
ax.set_yscale('log')
ax.set_xticks([0, 1]); ax.set_xticklabels(['Sphere', 'Rastrigin'])
ax.set_xlim(-0.5, 1.75)
ax.set_ylabel('Final best fitness (log scale)')
ax.set_title('Final result of each of the 30 runs', color=INK)
ax.grid(axis='y', color=GRID); ax.set_axisbelow(True)
fig.tight_layout(); fig.savefig('fig2_dfo_final_runs.png', dpi=150, bbox_inches='tight')

# Figure 3: every Rastrigin run, showing the plateau at ~0.995 and the escape
c = np.maximum(r['Rastrigin_curves'], FLOOR)
fig, ax = plt.subplots(figsize=(8, 4.5))
for run in c:
    ax.plot(it, run, color=ORANGE, lw=0.8, alpha=0.45)
ax.hlines(0.995, 0, it[-1], color=INK, ls='--', lw=1)
ax.text(it[-1] + 15, 0.995, 'Local minimum\n≈ 0.995 (one\ndimension at ±1)', va='center',
        fontsize=8, color=INK)
ax.set_yscale('log'); ax.set_ylim(1e-14, 1e3)
ax.set_xlim(0, it[-1] + 170)
ax.set_xlabel('Iteration'); ax.set_ylabel('Best fitness (log scale)')
ax.set_title('Rastrigin: most runs stall at a local minimum, then escape', color=INK)
ax.grid(color=GRID); ax.set_axisbelow(True)
fig.tight_layout(); fig.savefig('fig3_dfo_rastrigin_runs.png', dpi=150, bbox_inches='tight')