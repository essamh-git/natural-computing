"""
Week 3 Part 1: one DFO position update, done by hand and checked in code.
Update equation: x_id(t+1) = x_ind(t) + u * (x_sd(t) - x_id(t))
Fitness: Sphere (minimised). Topology: ring. The best fly (s) is not updated.
"""
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

X = np.array([[2, 3, 1], [1, 1, 1], [0, 0, 1], [-5, 0, 1]], dtype=float)
U = np.array([[0.1, 0.3, 0.01], [0.9, 0.4, 0.8], [np.nan] * 3, [0.5, 0.7, 0.6]])
N = len(X)

fitness = (X ** 2).sum(axis=1)
s = int(np.argmin(fitness))
print('Fitness at time t:', fitness, '-> best fly s = x%d' % s)

X_new = X.copy()           # SYNCHRONOUS: ALL UPDATES USE TIME-t POSITIONS
for i in range(N):
    if i == s:
        continue           # ELITISM: BEST FLY IS NOT UPDATED
    left, right = (i - 1) % N, (i + 1) % N
    bn = right if fitness[right] < fitness[left] else left
    for d in range(X.shape[1]):
        X_new[i, d] = X[bn, d] + U[i, d] * (X[s, d] - X[i, d])
        print(f'x{i} d{d}: {X[bn, d]:g} + {U[i, d]:g}({X[s, d]:g} - ({X[i, d]:g})) = {X_new[i, d]:g}')
    print(f'x{i}: best neighbour x{bn}, new position {X_new[i]}')

new_fitness = (X_new ** 2).sum(axis=1)
print('Fitness at time t+1:', new_fitness)

# FIGURE: MOVEMENT OF EACH FLY IN THE FIRST TWO DIMENSIONS
BLUE, ORANGE = '#2a78d6', '#eb6834'
INK, MUTED, GRID, SURF = '#0b0b0b', '#52514e', '#e4e3df', '#fcfcfb'
plt.rcParams.update({'axes.edgecolor': MUTED, 'axes.labelcolor': INK,
                     'xtick.color': MUTED, 'ytick.color': MUTED,
                     'axes.spines.top': False, 'axes.spines.right': False,
                     'figure.facecolor': SURF, 'axes.facecolor': SURF})
fig, ax = plt.subplots(figsize=(6.6, 4.8))
g = np.linspace(-6, 4, 200)
G0, G1 = np.meshgrid(g, g)
ax.contour(G0, G1, G0 ** 2 + G1 ** 2, levels=[1, 4, 9, 16, 25], colors=GRID, linewidths=1)
for i in range(N):
    if i == s:
        ax.scatter(*X[i, :2], marker='*', s=300, color=ORANGE, edgecolor=INK, zorder=4)
        ax.annotate(f'x{i} (best, not moved)', X[i, :2], xytext=(10, -24),
                    textcoords='offset points', fontsize=9, color=INK)
        continue
    ax.annotate('', xy=X_new[i, :2], xytext=X[i, :2],
                arrowprops=dict(arrowstyle='->', color=BLUE, lw=1.8))
    ax.scatter(*X[i, :2], s=50, color=SURF, edgecolor=BLUE, lw=1.5, zorder=3)
    ax.scatter(*X_new[i, :2], s=50, color=BLUE, zorder=3)
    ax.annotate(f'x{i}: {fitness[i]:g} → {new_fitness[i]:.2f}', X[i, :2], xytext=(8, 6),
                textcoords='offset points', fontsize=9, color=INK)
ax.scatter([], [], s=50, color=SURF, edgecolor=BLUE, lw=1.5, label='Position at t')
ax.scatter([], [], s=50, color=BLUE, label='Position at t+1')
ax.legend(fontsize=8, loc='upper left')
ax.set_xlabel('Dimension 0'); ax.set_ylabel('Dimension 1')
ax.set_title('One DFO update in dimensions 0 and 1 (labels = fitness before → after)', color=INK,
             fontsize=10)
ax.set_aspect('equal'); ax.set_xlim(-6, 4); ax.set_ylim(-1.5, 4)
fig.tight_layout()
fig.savefig('fig0_dfo_hand_update.png', dpi=150, bbox_inches='tight')