# Figures for the Week 5 TSP page (reads week5_tsp_results.npz made by TSPExperiments.py)
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from TSPImproved import distance_matrix, route_length

r = np.load('week5_tsp_results.npz')
xy, ref, ref_len = r['xy'], r['ref_route'], float(r['ref_len'])
Dm = distance_matrix(xy)
INK, MUTED, GRID, SURF = '#0b0b0b', '#52514e', '#e4e3df', '#fcfcfb'
BLUE, ORANGE = '#2a78d6', '#eb6834'
plt.rcParams.update({'axes.edgecolor': MUTED, 'axes.labelcolor': INK,
                     'xtick.color': MUTED, 'ytick.color': MUTED,
                     'axes.spines.top': False, 'axes.spines.right': False,
                     'figure.facecolor': SURF, 'axes.facecolor': SURF})
KEYS = [('Tutorial', 'Tutorial'), ('Fixed_elitism', 'Fixed elitism'),
        ('plus_Inversion_mutation', '+ Inversion\nmutation'), ('plus_2opt_memetic', '+ 2-opt\n(memetic)')]


def draw_route(ax, route, title, colour):
    loop = np.append(route, route[0])
    ax.plot(xy[loop, 0], xy[loop, 1], color=colour, lw=1.6, zorder=2)
    ax.scatter(xy[:, 0], xy[:, 1], s=22, color=INK, zorder=3)
    ax.set_title(title, fontsize=9.5, color=INK)
    ax.set_xticks([]); ax.set_yticks([]); ax.set_aspect('equal')
    for s in ax.spines.values():
        s.set_visible(True); s.set_color(GRID)


# Figure 1: same cities, different answers
routes, prog = r['Tutorial_routes'], r['Tutorial_progress']
final = prog[:, -1]
picks = [int(np.argmin(final)), int(np.argsort(final)[len(final) // 2]), int(np.argmax(final))]
fig, axes = plt.subplots(1, 4, figsize=(12, 3.4))
draw_route(axes[0], ref, f'Best known (OR-Tools)\n{ref_len:.1f}', BLUE)
for ax, k, lab in zip(axes[1:], picks, ['best', 'median', 'worst']):
    draw_route(ax, routes[k], f'Tutorial GA, {lab} run\n{route_length(routes[k], Dm):.1f}', ORANGE)
fig.suptitle('Same 25 cities, different runs, different routes', color=INK)
fig.tight_layout(rect=(0, 0, 1, 0.92)); fig.savefig('fig3_tsp_routes.png', dpi=150, bbox_inches='tight')

# Figure 2: the tutorial's best route can get worse; the fixed version cannot
fig, axes = plt.subplots(1, 2, figsize=(11, 4), sharey=True)
for ax, (key, title, col) in zip(axes, [('Tutorial', 'Tutorial code', ORANGE),
                                        ('Fixed_elitism', 'Fixed elitism', BLUE)]):
    P = r[f'{key}_progress']
    for p in P:
        ax.plot(p, color=col, lw=0.7, alpha=0.25)
    k = int(np.argmax([np.sum(np.diff(p) > 1e-9) for p in P])) if key == 'Tutorial' else 0
    ax.plot(P[k], color=col, lw=1.8)
    ax.axhline(ref_len, color=INK, ls='--', lw=1)
    ax.text(500, ref_len - 12, f'Best known {ref_len:.1f}', ha='right', va='top', fontsize=8, color=INK)
    worse = np.mean([np.sum(np.diff(p) > 1e-9) for p in P])
    ax.set_title(f'{title}: best route got worse {worse:.1f} times per run (average)',
                 fontsize=9.5, color=INK)
    ax.set_xlabel('Generation'); ax.set_ylim(ref_len - 60, 1300)
    ax.grid(color=GRID); ax.set_axisbelow(True)
axes[0].set_ylabel('Shortest route in the population')
fig.tight_layout(); fig.savefig('fig4_tsp_elitism_bug.png', dpi=150, bbox_inches='tight')

# Figure 3: improvements, final distance and run time
fig, (ax, ax2) = plt.subplots(1, 2, figsize=(11, 4.2), gridspec_kw={'width_ratios': [2.2, 1]})
rng = np.random.default_rng(0)
for j, (key, lab) in enumerate(KEYS):
    v = r[f'{key}_progress'][:, -1]
    ax.scatter(j + rng.uniform(-0.13, 0.13, len(v)), v, s=16, color=BLUE if j else ORANGE,
               alpha=0.6, lw=0, zorder=3)
    ax.hlines(np.median(v), j - 0.25, j + 0.25, color=INK, lw=2, zorder=4)
    opt = np.sum(v <= ref_len + 1e-6)
    ax.annotate(f'{opt}/{len(v)}\noptimal', (j, ref_len), xytext=(0, -26), textcoords='offset points',
                ha='center', va='top', fontsize=8, color=INK)
ax.axhline(ref_len, color=INK, ls='--', lw=1)
ax.set_xticks(range(len(KEYS))); ax.set_xticklabels([k[1] for k in KEYS], fontsize=9)
ax.set_ylabel('Final route length'); ax.set_ylim(ref_len - 45, None)
ax.set_title('Final route length of every run (line = median, dashed = best known)',
             fontsize=9.5, color=INK)
ax.grid(axis='y', color=GRID); ax.set_axisbelow(True)
times = [r[f'{key}_time'].mean() for key, _ in KEYS]
bars = ax2.bar(range(len(KEYS)), times, color=[ORANGE] + [BLUE] * 3, width=0.6)
for b, t in zip(bars, times):
    ax2.text(b.get_x() + b.get_width() / 2, t + 0.3, f'{t:.1f} s', ha='center', fontsize=8.5, color=INK)
ax2.set_xticks(range(len(KEYS))); ax2.set_xticklabels([k[1] for k in KEYS], fontsize=8)
ax2.set_ylabel('Time per run (seconds)'); ax2.set_title('Run time', fontsize=9.5, color=INK)
ax2.grid(axis='y', color=GRID); ax2.set_axisbelow(True)
fig.tight_layout(); fig.savefig('fig5_tsp_improvements.png', dpi=150, bbox_inches='tight')