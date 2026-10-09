# Figures for the Week 2 Part 1 wiki page (uses the same seed as sphere_part1.py)
import numpy as np
from random import random, seed
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

seed(42)
D, lower_bound, upper_bound = 30, -5.12, 5.12
x = np.array([lower_bound + random() * (upper_bound - lower_bound) for _ in range(D)])
f = float((x ** 2).sum())

BLUE, ORANGE = '#2a78d6', '#eb6834'
INK, MUTED, GRID, SURF = '#0b0b0b', '#52514e', '#e4e3df', '#fcfcfb'
plt.rcParams.update({'axes.edgecolor': MUTED, 'axes.labelcolor': INK,
                     'xtick.color': MUTED, 'ytick.color': MUTED,
                     'axes.spines.top': False, 'axes.spines.right': False,
                     'figure.facecolor': SURF, 'axes.facecolor': SURF})
dims = np.arange(1, D + 1)

# Figure 1: values of the random solution vs the optimum
fig, ax = plt.subplots(figsize=(8, 4.2))
ax.bar(dims, x, color=BLUE, width=0.7)
ax.axhline(0, color=INK, lw=1)
for b in (lower_bound, upper_bound):
    ax.axhline(b, color=MUTED, ls='--', lw=1)
ax.text(D + 1.3, upper_bound, 'Upper bound', va='center', fontsize=8, color=MUTED)
ax.text(D + 1.3, lower_bound, 'Lower bound', va='center', fontsize=8, color=MUTED)
ax.text(D + 1.3, 0, 'Optimum (0)', va='center', fontsize=8, color=INK)
ax.set_xlim(0, D + 1.2); ax.set_ylim(-6, 6)
ax.set_xlabel('Dimension'); ax.set_ylabel('Value of x')
ax.set_title('Random solution: values spread across the whole range', color=INK)
ax.grid(axis='y', color=GRID); ax.set_axisbelow(True)
fig.tight_layout(); fig.savefig('fig1_random_solution.png', dpi=150, bbox_inches='tight')

# Figure 2: how much each dimension contributes to the fitness (x_i^2)
c = x ** 2
top = set(np.argsort(-c)[:5])
fig, ax = plt.subplots(figsize=(8, 4.2))
ax.bar(dims, c, color=[ORANGE if i in top else BLUE for i in range(D)], width=0.7)
ax.set_xlim(0, D + 1)
ax.set_xlabel('Dimension'); ax.set_ylabel('Contribution to fitness (x²)')
ax.set_title(f'Fitness {f:.1f} = sum of 30 contributions '
             f'(5 largest, in orange, give {c[list(top)].sum() / f:.0%})', color=INK)
ax.grid(axis='y', color=GRID); ax.set_axisbelow(True)
fig.tight_layout(); fig.savefig('fig2_contributions.png', dpi=150, bbox_inches='tight')

# Figure 3: the Sphere function landscape in 2D
g = np.linspace(lower_bound, upper_bound, 300)
G1, G2 = np.meshgrid(g, g)
fig, ax = plt.subplots(figsize=(5.6, 5))
cs = ax.contourf(G1, G2, G1 ** 2 + G2 ** 2, levels=20, cmap='Blues_r')
fig.colorbar(cs, ax=ax, label='Fitness')
ax.scatter([0], [0], marker='*', s=260, color=ORANGE, edgecolor=INK, label='Optimum (0, 0)')
ax.set_xlabel('x1'); ax.set_ylabel('x2')
ax.set_title('Sphere function in 2D: one bowl, one minimum', color=INK)
ax.legend(loc='upper right', fontsize=8); ax.set_aspect('equal')
fig.tight_layout(); fig.savefig('fig3_sphere_2d.png', dpi=150, bbox_inches='tight')