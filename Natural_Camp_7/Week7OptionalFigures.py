# Figures for the Week 7 optional page (reads week7_optional.npz made by Week7Optional.py)
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

r = np.load('week7_optional.npz')
INK, MUTED, GRID, SURF = '#0b0b0b', '#52514e', '#e4e3df', '#fcfcfb'
BLUE, ORANGE, GREY = '#2a78d6', '#eb6834', '#b7b6b1'
plt.rcParams.update({'axes.edgecolor': MUTED, 'axes.labelcolor': INK,
                     'xtick.color': MUTED, 'ytick.color': MUTED,
                     'axes.spines.top': False, 'axes.spines.right': False,
                     'figure.facecolor': SURF, 'axes.facecolor': SURF})
MODES = [('standard', 'Standard SDS'), ('context_free', 'Context-free'),
         ('context_sensitive', 'Context-sensitive')]

# Figure 4: one example run per mode
fig, axes = plt.subplots(1, 3, figsize=(13, 4), sharey=True)
run, T = 3, 300
it = np.arange(1, T + 1)
for ax, (m, title) in zip(axes, MODES):
    s1, s2 = r[f'{m}_s1'][run, :T], r[f'{m}_s2'][run, :T]
    ax.plot(it, s1, color=BLUE, lw=1.4, label='First "sds"')
    ax.plot(it, s2, color=ORANGE, lw=1.4, label='Second "sds"')
    ax.plot(it, 100 - s1 - s2, color=GREY, lw=1.2, label='Elsewhere (searching)')
    ax.set_title(title, fontsize=10, color=INK)
    ax.set_xlabel('Iteration'); ax.set_ylim(0, 100)
    ax.grid(color=GRID); ax.set_axisbelow(True)
axes[0].set_ylabel('Agents (%)')
axes[0].legend(fontsize=8, loc='upper right')
fig.suptitle('Two matches, 100 agents: one example run per mode', color=INK)
fig.tight_layout(); fig.savefig('fig4_sds_modes_example.png', dpi=150, bbox_inches='tight')

# Figure 5: where the agents are (left) and how balanced the two clusters are (right)
fig, (ax, ax2) = plt.subplots(1, 2, figsize=(12, 4.3))
labels = [t for _, t in MODES]
half = slice(500, None)
a1 = [r[f'{m}_s1'][:, half].mean() for m, _ in MODES]
a2 = [r[f'{m}_s2'][:, half].mean() for m, _ in MODES]
el = [100 - x - y for x, y in zip(a1, a2)]
x = np.arange(3)
ax.bar(x, a1, color=BLUE, width=0.55, label='First "sds"')
ax.bar(x, a2, bottom=a1, color=ORANGE, width=0.55, label='Second "sds"')
ax.bar(x, el, bottom=np.add(a1, a2), color=GREY, width=0.55, label='Elsewhere (searching)')
for i in range(3):
    for bottom, h in ((0, a1[i]), (a1[i], a2[i]), (a1[i] + a2[i], el[i])):
        if h > 4:
            ax.text(i, bottom + h / 2, f'{h:.0f}%', ha='center', va='center', fontsize=9, color='white')
ax.set_xticks(x); ax.set_xticklabels(labels); ax.set_ylabel('Agents (%), average over iterations 500–1000')
ax.set_title('Where the agents are', fontsize=10, color=INK)
ax.legend(fontsize=8, loc='upper center', bbox_to_anchor=(0.5, -0.1), ncol=3, frameon=False)
ax.set_ylim(0, 100)

rng = np.random.default_rng(0)
for i, (m, _) in enumerate(MODES):
    s1, s2 = r[f'{m}_s1'][:, -1], r[f'{m}_s2'][:, -1]
    tot = s1 + s2
    bal = 100 * np.minimum(s1, s2) / np.where(tot > 0, tot, 1)
    ax2.scatter(i + rng.uniform(-0.15, 0.15, len(bal)), bal, s=12, color=BLUE, alpha=0.5, lw=0)
    ax2.hlines(np.median(bal), i - 0.28, i + 0.28, color=INK, lw=2)
    ax2.text(i + 0.3, np.median(bal), f'median\n{np.median(bal):.0f}%', va='center', fontsize=8, color=INK)
ax2.axhline(50, color=MUTED, ls='--', lw=1)
ax2.text(2.55, 51, 'equal split', fontsize=8, color=MUTED, ha='right', va='bottom')
ax2.set_xticks(x); ax2.set_xticklabels(labels); ax2.set_xlim(-0.5, 2.7); ax2.set_ylim(-2, 55)
ax2.set_ylabel('Smaller cluster as % of both clusters')
ax2.set_title('How evenly the two matches are shared (end of each of 200 runs)', fontsize=10, color=INK)
ax2.grid(axis='y', color=GRID); ax2.set_axisbelow(True)
fig.tight_layout(); fig.savefig('fig5_sds_modes_allocation.png', dpi=150, bbox_inches='tight')