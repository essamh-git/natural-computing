# Figures for the Week 7 SDS page (reads week7_results.npz made by Week7Experiments.py)
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

r = np.load('week7_results.npz')
INK, MUTED, GRID, SURF = '#0b0b0b', '#52514e', '#e4e3df', '#fcfcfb'
BLUE, ORANGE, AQUA = '#2a78d6', '#eb6834', '#1baf7a'
plt.rcParams.update({'axes.edgecolor': MUTED, 'axes.labelcolor': INK,
                     'xtick.color': MUTED, 'ytick.color': MUTED,
                     'axes.spines.top': False, 'axes.spines.right': False,
                     'figure.facecolor': SURF, 'axes.facecolor': SURF})


def band(ax, A, col, label, n=None):
    A = A[:, :n] if n else A
    it = np.arange(1, A.shape[1] + 1)
    ax.fill_between(it, np.percentile(A, 10, 0), np.percentile(A, 90, 0), color=col, alpha=0.15, lw=0)
    ax.plot(it, np.median(A, 0), color=col, lw=2, label=label)


# Figure 1: Q2, how long until every agent is active
fig, (ax, ax2) = plt.subplots(1, 2, figsize=(11, 4))
band(ax, r['q2_activity'], BLUE, "Median of 1,000 runs (shaded: 10th–90th percentile)", n=40)
ax.set_xlabel('Iteration'); ax.set_ylabel('Active agents (%)'); ax.set_ylim(0, 105)
ax.set_title("Searching for 'sds' with 10 agents", fontsize=10, color=INK)
ax.legend(fontsize=8, loc='lower right'); ax.grid(color=GRID); ax.set_axisbelow(True)
ff = r['q2_first']
ax2.hist(ff, bins=np.arange(0.5, ff.max() + 1.5), color=BLUE)
ax2.axvline(np.median(ff), color=ORANGE, lw=2)
ax2.text(np.median(ff) + 0.6, ax2.get_ylim()[1] * 0.9, f'Median {np.median(ff):.0f}', color=INK, fontsize=9)
ax2.set_xlabel('First iteration with all 10 agents active'); ax2.set_ylabel('Number of runs')
ax2.set_title('When does every agent become active? (1,000 runs)', fontsize=10, color=INK)
ax2.grid(axis='y', color=GRID); ax2.set_axisbelow(True)
fig.tight_layout(); fig.savefig('fig1_sds_all_active.png', dpi=150, bbox_inches='tight')

# Figure 2: Q3, no perfect match (averages over 1,000 runs)
fig, ax = plt.subplots(figsize=(8, 4.2))
for A, col, lab in ((r['q2_activity'], BLUE, "Active agents: 'sds' (perfect match)"),
                    (r['q3_activity'], ORANGE, "Active agents: 'fine' (no perfect match)"),
                    (r['q3_share'], AQUA, "'fine': agents pointing at 'find'")):
    ax.plot(np.arange(1, A.shape[1] + 1), A.mean(0), color=col, lw=2, label=lab)
ax.axhline(75, color=MUTED, ls='--', lw=1)
ax.text(40, 76.5, "'find' matches 3 of 4 letters (75%)", ha='left', va='bottom', fontsize=8, color=MUTED)
ax.set_xlabel('Iteration'); ax.set_ylabel('% of the 10 agents (average of 1,000 runs)')
ax.set_ylim(0, 105); ax.set_xlim(1, 200)
ax.set_title("No perfect match: most agents still gather on 'find', but activity stays well below 100%",
             fontsize=9.5, color=INK)
ax.legend(fontsize=8, loc='lower right'); ax.grid(color=GRID); ax.set_axisbelow(True)
fig.tight_layout(); fig.savefig('fig2_sds_no_perfect_match.png', dpi=150, bbox_inches='tight')

# Figure 3: Q4, two matches
fig, axes = plt.subplots(1, 3, figsize=(13, 4))
for ax, N, iters in ((axes[0], 10, 60), (axes[1], 100, 60)):
    S = r[f'q4_N{N}_share1']
    for k in range(25):
        ax.plot(np.arange(1, iters + 1), S[k, :iters], color=BLUE, lw=1, alpha=0.6)
    ax.set_ylim(-3, 103); ax.set_xlabel('Iteration'); ax.set_ylabel('Agents on the first "sds" (%)')
    ax.set_title(f'{N} agents: 25 example runs', fontsize=10, color=INK)
    ax.grid(color=GRID); ax.set_axisbelow(True)
fin = r['q4_N100_share1'][:, -1]
axes[2].hist(fin, bins=np.arange(0, 101, 5), color=BLUE)
axes[2].set_xlabel('Agents on the first "sds" after 1,000 iterations (%)'); axes[2].set_ylabel('Number of runs')
axes[2].set_title('100 agents: final split, 200 runs', fontsize=10, color=INK)
axes[2].grid(axis='y', color=GRID); axes[2].set_axisbelow(True)
fig.suptitle('Two perfect matches: the agents split between them, and the split freezes', color=INK)
fig.tight_layout(); fig.savefig('fig3_sds_two_matches.png', dpi=150, bbox_inches='tight')