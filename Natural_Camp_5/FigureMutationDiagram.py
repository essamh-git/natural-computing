# Diagram: swap mutation vs inversion mutation on a 6-city route
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

INK, MUTED, GRID, SURF = '#0b0b0b', '#52514e', '#e4e3df', '#fcfcfb'
KEEP, NEW = '#9ec5f4', '#eb6834'
ang = np.linspace(0, 2 * np.pi, 6, endpoint=False) + np.pi / 2
xy = np.c_[np.cos(ang), np.sin(ang)]
labels = 'ABCDEF'


def edges(route):
    return {frozenset((route[i], route[(i + 1) % len(route)])) for i in range(len(route))}


def draw(ax, route, title, base):
    e_base = edges(base)
    for i in range(len(route)):
        a, b = route[i], route[(i + 1) % len(route)]
        new = frozenset((a, b)) not in e_base
        ax.plot(*xy[[a, b]].T, color=NEW if new else KEEP, lw=3 if new else 2, zorder=1)
    for k, (x, y) in enumerate(xy):
        ax.scatter(x, y, s=420, color=SURF, edgecolor=INK, lw=1.2, zorder=2)
        ax.text(x, y, labels[k], ha='center', va='center', fontsize=11, color=INK, zorder=3)
    changed = len(edges(route) - e_base)
    ax.set_title(f"{title}\n{' → '.join(labels[c] for c in route)}  ({changed} new links)",
                 fontsize=9.5, color=INK)
    ax.set_xlim(-1.3, 1.3); ax.set_ylim(-1.3, 1.3)
    ax.set_aspect('equal'); ax.axis('off')


base = [0, 1, 2, 3, 4, 5]
swap = [0, 4, 2, 3, 1, 5]          # swap B and E
inv = [0, 4, 3, 2, 1, 5]           # reverse B..E
fig, axes = plt.subplots(1, 3, figsize=(10, 3.8))
fig.patch.set_facecolor(SURF)
draw(axes[0], base, 'Original route', base)
draw(axes[1], swap, 'Swap mutation (swap B and E)', base)
draw(axes[2], inv, 'Inversion mutation (reverse B to E)', base)
h = [plt.Line2D([], [], color=KEEP, lw=2, label='Link kept'),
     plt.Line2D([], [], color=NEW, lw=3, label='New link')]
fig.legend(handles=h, loc='lower center', ncol=2, fontsize=9, frameon=False)
fig.tight_layout(rect=(0, 0.07, 1, 1))
fig.savefig('fig6_swap_vs_inversion.png', dpi=150, bbox_inches='tight', facecolor=SURF)