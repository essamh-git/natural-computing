# Figures for the TSP challenge page (run TSPChallenge.py first)
import itertools
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from TSPChallenge import CITIES, D, tour_length, brute_force

INK, MUTED, GRID, SURF = '#0b0b0b', '#52514e', '#e4e3df', '#fcfcfb'
BLUE, ORANGE = '#2a78d6', '#eb6834'
plt.rcParams.update({'axes.edgecolor': MUTED, 'axes.labelcolor': INK,
                     'xtick.color': MUTED, 'ytick.color': MUTED,
                     'figure.facecolor': SURF, 'axes.facecolor': SURF})

# APPROXIMATE (LONGITUDE, LATITUDE): ONLY USED TO DRAW THE MAP, NOT FOR ANY DISTANCES
POS = {'Brighton': (-0.14, 50.82), 'Bristol': (-2.59, 51.45), 'Cambridge': (0.12, 52.21),
       'Glasgow': (-4.25, 55.86), 'Liverpool': (-2.98, 53.41), 'London': (-0.13, 51.51),
       'Manchester': (-2.24, 53.48), 'Oxford': (-1.26, 51.75)}
best, tours, lengths, _ = brute_force(closed=True)
route = list(tours[0]) + [tours[0][0]]

fig, (ax, ax2) = plt.subplots(1, 2, figsize=(11, 5), gridspec_kw={'width_ratios': [1, 1.4]})
xy = np.array([POS[c] for c in CITIES])
for a, b in zip(route, route[1:]):
    ax.annotate('', xy=xy[b], xytext=xy[a],
                arrowprops=dict(arrowstyle='->', color=BLUE, lw=1.8, shrinkA=6, shrinkB=6))
    v = xy[b] - xy[a]
    perp = np.array([v[1], -v[0]]) / np.linalg.norm(v)      # RIGHT-HAND SIDE OF THE DIRECTION OF TRAVEL
    mid = (xy[a] + xy[b]) / 2 + 0.22 * perp
    ax.text(*mid, str(D[a, b]), fontsize=7.5, color=MUTED, ha='center', va='center',
            bbox=dict(boxstyle='round,pad=0.15', fc=SURF, ec='none'))
ax.scatter(xy[:, 0], xy[:, 1], s=40, color=INK, zorder=3)
offsets = {'Brighton': (6, -10), 'London': (8, 2), 'Cambridge': (6, 4), 'Oxford': (-40, 6),
           'Bristol': (-40, -12), 'Liverpool': (-52, 0), 'Manchester': (8, 0), 'Glasgow': (8, 0)}
for c, (x, y) in POS.items():
    ax.annotate(c, (x, y), xytext=offsets[c], textcoords='offset points', fontsize=9, color=INK)
ax.scatter(*xy[0], s=140, facecolor='none', edgecolor=ORANGE, lw=2, zorder=4)
ax.set_title(f'Shortest round trip: {best} km\n(start at Brighton, circled)', fontsize=10, color=INK)
ax.set_aspect(1 / np.cos(np.radians(53))); ax.set_xticks([]); ax.set_yticks([])
ax.set_xlim(-5.2, 1.3); ax.set_ylim(50.4, 56.3)
for s in ax.spines.values():
    s.set_color(GRID)

# EACH LOOP ONCE: START AT BRIGHTON (0) AND KEEP ONE OF ITS TWO DIRECTIONS
uniq = np.array([tour_length((0,) + t) for t in itertools.permutations(range(1, 8)) if t[0] < t[-1]])
ax2.hist(uniq, bins=40, color=BLUE, alpha=0.85)
ax2.axvline(best, color=ORANGE, lw=2)
ax2.text(best + 20, ax2.get_ylim()[1] * 0.92, f'Shortest\n{best} km', color=INK, fontsize=9, va='top')
ax2.axvline(uniq.mean(), color=MUTED, ls='--', lw=1)
ax2.text(uniq.mean() + 20, ax2.get_ylim()[1] * 0.97, f'Average {uniq.mean():.0f} km', color=MUTED,
         fontsize=8.5, va='top')
ax2.set_xlabel('Round-trip length (km)'); ax2.set_ylabel('Number of tours')
ax2.set_title(f'All {len(uniq):,} possible round trips (brute force)', fontsize=10, color=INK)
ax2.spines[['top', 'right']].set_visible(False)
ax2.grid(axis='y', color=GRID); ax2.set_axisbelow(True)
fig.tight_layout(); fig.savefig('fig7_tsp_challenge.png', dpi=150, bbox_inches='tight')
print('unique tours', len(uniq), 'min', uniq.min(), 'how many at min', (uniq == uniq.min()).sum())