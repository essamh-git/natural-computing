# Figures for the Week 2 Part 3 wiki page (standalone: no other files needed)
import math as m
from random import random, seed
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# REPRODUCE THE SAME SOLUTIONS AS THE TASK CODE (SAME SEED, SAME ORDER OF DRAWS)
seed(42)
D, N = 30, 50
lower_bound, upper_bound = -5.12, 5.12
x = [lower_bound + random() * (upper_bound - lower_bound) for _ in range(D)]   # PART I SOLUTION
X = [[lower_bound + random() * (upper_bound - lower_bound) for j in range(D)] for i in range(N)]

def sphere(v):
    return sum(m.pow(a, 2) for a in v)

def rastrigin(v):
    return 10.0 * len(v) + sum(m.pow(a, 2) - 10.0 * m.cos(2 * m.pi * a) for a in v)

fitness = [sphere(s) for s in X]
best_index = int(np.argmin(fitness))
fs = np.array(fitness)
fr = np.array([rastrigin(s) for s in X])
b = int(np.argmin(fr))

BLUE, ORANGE = '#2a78d6', '#eb6834'
INK, MUTED, GRID, SURF = '#0b0b0b', '#52514e', '#e4e3df', '#fcfcfb'
plt.rcParams.update({'axes.edgecolor': MUTED, 'axes.labelcolor': INK,
                     'xtick.color': MUTED, 'ytick.color': MUTED,
                     'axes.spines.top': False, 'axes.spines.right': False,
                     'figure.facecolor': SURF, 'axes.facecolor': SURF})

# Figure 6: Rastrigin landscape in 2D
g = np.linspace(-5.12, 5.12, 500)
G1, G2 = np.meshgrid(g, g)
Z = 20 + G1**2 - 10*np.cos(2*np.pi*G1) + G2**2 - 10*np.cos(2*np.pi*G2)
fig, ax = plt.subplots(figsize=(5.6, 5))
cs = ax.contourf(G1, G2, Z, levels=30, cmap='Blues_r')
fig.colorbar(cs, ax=ax, label='Fitness')
ax.scatter([0], [0], marker='*', s=260, color=ORANGE, edgecolor=INK, label='Global optimum (0, 0)')
ax.set_xlabel('x1'); ax.set_ylabel('x2')
ax.set_title('Rastrigin function in 2D: many local minima', color=INK)
ax.legend(loc='upper right', fontsize=8); ax.set_aspect('equal')
fig.tight_layout(); fig.savefig('fig6_rastrigin_2d.png', dpi=150, bbox_inches='tight')

# Figure 7: Sphere vs Rastrigin fitness for the same 50 solutions
fig, ax = plt.subplots(figsize=(6.4, 4.6))
ax.scatter(fs, fr, s=40, color=BLUE, edgecolor=SURF, lw=1)
ax.scatter([fs[b]], [fr[b]], s=70, color=ORANGE, edgecolor=INK, lw=1, zorder=3)
ax.annotate(f'Solution {b+1}: best on both', xy=(fs[b], fr[b]),
            xytext=(fs[b] + 20, fr[b] + 5), fontsize=9, color=INK,
            arrowprops=dict(arrowstyle='->', color=MUTED))
k, c0 = np.polyfit(fs, fr, 1)
xx = np.array([fs.min(), fs.max()])
ax.plot(xx, k*xx + c0, color=MUTED, ls='--', lw=1.2)
ax.text(xx[1], k*xx[1] + c0 + 8, f'r = {np.corrcoef(fs, fr)[0,1]:.2f}', fontsize=9,
        color=MUTED, ha='right')
ax.set_xlabel('Sphere fitness'); ax.set_ylabel('Rastrigin fitness')
ax.set_title('Same 50 solutions, two fitness functions', color=INK)
ax.grid(color=GRID); ax.set_axisbelow(True)
fig.tight_layout(); fig.savefig('fig7_sphere_vs_rastrigin.png', dpi=150, bbox_inches='tight')