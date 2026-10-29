# Figures for the card game page (run CardsExperiments.py first)
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import CardsGA as C

r = np.load('cards_results.npz')
INK, MUTED, GRID, SURF = '#0b0b0b', '#52514e', '#e4e3df', '#fcfcfb'
COL = {'Generational': '#2a78d6', 'Steady-state': '#eb6834', 'Random': '#52514e'}
LAB = {'Generational': 'Generational GA', 'Steady-state': 'Steady-state GA', 'Random': 'Random search'}
plt.rcParams.update({'axes.edgecolor': MUTED, 'axes.labelcolor': INK,
                     'xtick.color': MUTED, 'ytick.color': MUTED,
                     'axes.spines.top': False, 'axes.spines.right': False,
                     'figure.facecolor': SURF, 'axes.facecolor': SURF})

# Figure 1: why it is hard - the error after moving each card away from the perfect split
opt = list(r['optimum'])
moved = []
for k in range(C.L):
    g = list(opt); g[k] = 1 - g[k]
    moved.append(C.error(g))
fig, ax = plt.subplots(figsize=(7.5, 3.8))
bars = ax.bar([str(c) for c in C.CARDS], moved, color=['#eb6834' if m < 0.1 else '#2a78d6' for m in moved])
for b, m in zip(bars, moved):
    ax.text(b.get_x() + b.get_width() / 2, m * 1.15, f'{m:.2f}', ha='center', fontsize=8, color=INK)
ax.set_yscale('log'); ax.set_ylim(0.01, 30)
ax.set_xlabel('Card moved to the other pile'); ax.set_ylabel('Error (log scale)')
ax.set_title('The perfect split is isolated: moving almost any card makes it much worse',
             fontsize=10, color=INK)
ax.grid(axis='y', color=GRID); ax.set_axisbelow(True)
fig.tight_layout(); fig.savefig('fig8_cards_landscape.png', dpi=150, bbox_inches='tight')

# Figure 2: % of runs solved against the number of evaluations
fig, ax = plt.subplots(figsize=(7.5, 4))
for key in ('Generational', 'Steady-state', 'Random'):
    H = r[f'{key}_history']
    solved = 100 * np.mean(H == 0, axis=0)
    ax.plot(np.arange(1, H.shape[1] + 1), solved, color=COL[key], lw=2,
            ls='--' if key == 'Random' else '-', label=LAB[key])
ax.axvline(400, color=MUTED, lw=1, ls=':')
ax.text(410, 3, 'Table 1 budget\n(400 evaluations)', fontsize=8, color=MUTED)
ax.set_xlabel('Fitness evaluations'); ax.set_ylabel('Runs that found the perfect split (%)')
ax.set_title('50 runs each: neither GA beats random search on this problem', fontsize=10, color=INK)
ax.set_ylim(0, 100); ax.legend(fontsize=8, loc='lower right')
ax.grid(color=GRID); ax.set_axisbelow(True)
fig.tight_layout(); fig.savefig('fig9_cards_success.png', dpi=150, bbox_inches='tight')

# Figure 3: final error of every run at 400 evaluations
fig, ax = plt.subplots(figsize=(7.5, 4))
rng = np.random.default_rng(0)
for k, key in enumerate(('Generational', 'Steady-state', 'Random')):
    v = r[f'{key}_history'][:, 399]
    ax.scatter(k + rng.uniform(-0.15, 0.15, len(v)), v, s=22, color=COL[key], alpha=0.7, lw=0)
    ax.hlines(v.mean(), k - 0.28, k + 0.28, color=INK, lw=2)
    ax.text(k + 0.3, v.mean(), f'mean {v.mean():.4f}\nvariance {v.var(ddof=1):.6f}', va='center',
            fontsize=8, color=INK)
ax.set_xticks(range(3)); ax.set_xticklabels([LAB[k] for k in COL])
ax.set_xlim(-0.5, 2.9)
ax.set_ylabel('Best error after 400 evaluations')
ax.set_title('Final error of every run (0 = perfect split)', fontsize=10, color=INK)
ax.grid(axis='y', color=GRID); ax.set_axisbelow(True)
fig.tight_layout(); fig.savefig('fig10_cards_final.png', dpi=150, bbox_inches='tight')