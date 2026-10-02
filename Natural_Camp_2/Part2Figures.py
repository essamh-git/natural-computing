# Figures for the Week 2 Part 2 wiki page (standalone: no other files needed)
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
lb, ub = lower_bound, upper_bound

BLUE, ORANGE = '#2a78d6', '#eb6834'
INK, MUTED, GRID, SURF = '#0b0b0b', '#52514e', '#e4e3df', '#fcfcfb'
plt.rcParams.update({'axes.edgecolor': MUTED, 'axes.labelcolor': INK,
                     'xtick.color': MUTED, 'ytick.color': MUTED,
                     'axes.spines.top': False, 'axes.spines.right': False,
                     'figure.facecolor': SURF, 'axes.facecolor': SURF})
expected = D * ub ** 2 / 3

# Figure 1: fitness of all 50 solutions, best highlighted
fig, ax = plt.subplots(figsize=(8, 4.2))
ax.bar(range(1, N + 1), fitness, width=0.7,
       color=[ORANGE if i == best_index else BLUE for i in range(N)])
ax.axhline(expected, color=MUTED, ls='--', lw=1.5)
ax.text(N + 0.8, expected, f'Expected for a\nrandom solution\n({expected:.1f})',
        va='center', fontsize=8, color=MUTED)
ax.annotate(f'Best: solution {best_index + 1}\nfitness {fitness[best_index]:.1f}',
            xy=(best_index + 1, fitness[best_index]),
            xytext=(best_index + 1, fitness[best_index] + 140),
            ha='center', fontsize=9, color=INK,
            arrowprops=dict(arrowstyle='->', color=MUTED))
ax.set_xlim(0, N + 1); ax.set_ylim(0, 480)
ax.set_xlabel('Solution number'); ax.set_ylabel('Fitness (lower is better)')
ax.set_title('Fitness of 50 random solutions (optimum = 0)', color=INK)
ax.grid(axis='y', color=GRID); ax.set_axisbelow(True)
fig.tight_layout(); fig.savefig('fig4_fitness_50.png', dpi=150, bbox_inches='tight')

# Figure 2: values of the best solution vs the optimum
best = X[best_index]
fig, ax = plt.subplots(figsize=(8, 4.2))
ax.bar(range(1, D + 1), best, color=BLUE, width=0.7)
ax.axhline(0, color=INK, lw=1)
for b in (lb, ub):
    ax.axhline(b, color=MUTED, ls='--', lw=1)
ax.text(D + 1.3, ub, 'Upper bound', va='center', fontsize=8, color=MUTED)
ax.text(D + 1.3, lb, 'Lower bound', va='center', fontsize=8, color=MUTED)
ax.text(D + 1.3, 0, 'Optimum (0)', va='center', fontsize=8, color=INK)
ax.set_xlim(0, D + 1.2); ax.set_ylim(-6, 6)
ax.set_xlabel('Dimension'); ax.set_ylabel('Value of x')
ax.set_title(f'Best of 50 solutions: still far from 0 (fitness {fitness[best_index]:.1f})',
             color=INK)
ax.grid(axis='y', color=GRID); ax.set_axisbelow(True)
fig.tight_layout();
fig.savefig('fig5_best_solution.png', dpi=150, bbox_inches='tight')