# Figures for the Week 4 wiki page (reads week4_results.npz made by Week4Experiments.py)
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from DFOImage import blob, brightness_map

r = np.load('week4_results.npz')
INK, MUTED, GRID, SURF = '#0b0b0b', '#52514e', '#e4e3df', '#fcfcfb'
FLY, BEST, TRUE = '#1baf7a', '#e34948', '#eda100'
DELTA_COL = {0.001: '#9ec5f4', 0.01: '#2a78d6', 0.1: '#0d366b'}
plt.rcParams.update({'axes.edgecolor': MUTED, 'axes.labelcolor': INK,
                     'xtick.color': MUTED, 'ytick.color': MUTED,
                     'figure.facecolor': SURF, 'axes.facecolor': SURF})


def show(ax, img, pos=None, title='', true=None):
    ax.imshow(img, cmap='gray', vmin=0, vmax=img.max())
    if pos is not None:
        B = brightness_map(img)
        fit = [B[int(round(p[1])), int(round(p[0]))] for p in pos]
        b = int(np.argmax(fit))
        ax.scatter(pos[:, 0], pos[:, 1], s=10, color=FLY, lw=0)
        ax.scatter(*pos[b], s=40, color=BEST, edgecolor='white', lw=0.8, zorder=3)
    if true is not None:
        ax.scatter(*true, s=260, facecolor='none', edgecolor=TRUE, lw=1.6)
    ax.set_title(title, fontsize=9, color=INK)
    ax.set_xticks([]); ax.set_yticks([])


def true_peak(img):
    B = brightness_map(img)
    y, x = np.unravel_index(np.argmax(B), B.shape)
    return (x, y)


def legend_row(fig):
    h = [plt.Line2D([], [], marker='o', ls='', color=FLY, ms=4, label='Fly'),
         plt.Line2D([], [], marker='o', ls='', color=BEST, mec='white', ms=6, label='Best fly'),
         plt.Line2D([], [], marker='o', ls='', mfc='none', mec=TRUE, ms=11, label='True brightest area')]
    fig.legend(handles=h, loc='lower center', ncol=3, fontsize=8, frameon=False)


# Figure 1: Task 1, the swarm finding the brightest area
img = r['t1_img']; P = r['t1_positions']
fig, axes = plt.subplots(1, 3, figsize=(9.5, 3.6))
for ax, it in zip(axes, [0, 10, 99]):
    show(ax, img, P[it], f'Iteration {it}', true_peak(img))
legend_row(fig)
fig.tight_layout(rect=(0, 0.07, 1, 1))
fig.savefig('fig1_find_brightest.png', dpi=150, bbox_inches='tight')

# Figure 2: Task 1, success rate against the disturbance threshold
fig, ax = plt.subplots(figsize=(6.8, 3.8))
d = r['t1_deltas']; rate = 100 * r['t1_rates']
bars = ax.bar(range(len(d)), rate, color='#2a78d6', width=0.6)
for b, v in zip(bars, rate):
    ax.text(b.get_x() + b.get_width() / 2, v + 1.5, f'{v:.0f}%', ha='center', fontsize=9, color=INK)
ax.set_xticks(range(len(d))); ax.set_xticklabels([f'{x:g}' for x in d])
ax.set_ylim(0, 110); ax.set_xlabel('Disturbance threshold (Δ)')
ax.set_ylabel('Runs that found the brightest area (%)')
ax.set_title('2D image search needs much more disturbance than Week 3 (90 runs per Δ)',
             fontsize=10, color=INK)
ax.spines[['top', 'right']].set_visible(False)
ax.grid(axis='y', color=GRID); ax.set_axisbelow(True)
fig.tight_layout(); fig.savefig('fig2_delta_success.png', dpi=150, bbox_inches='tight')

# Figure 3: Task 2, other input images
names = ['Synthetic_2', 'Synthetic_3', 'Coins', 'Hubble_Deep_Field']
fig, axes = plt.subplots(1, 4, figsize=(12, 3.5))
for ax, n in zip(axes, names):
    im = r[f't2_{n}_img']; k = r[f't2_{n}_rates']
    show(ax, im, r[f't2_{n}_final'],
         f"{n.replace('_', ' ')}\n{k[0]}/30 runs (100 it.), {k[1]}/30 (300 it.)", true_peak(im))
legend_row(fig)
fig.tight_layout(rect=(0, 0.07, 1, 1))
fig.savefig('fig3_other_images.png', dpi=150, bbox_inches='tight')

# Figure 4: Task 3, switching the image half-way
A, Bimg = r['t3_imgA'], r['t3_imgB']
fig, axes = plt.subplots(2, 3, figsize=(9.5, 6.6))
for row, dl in enumerate([0.001, 0.1]):
    P = r[f't3_d{dl}_positions_run0']
    for col, (it, im) in enumerate([(99, A), (110, Bimg), (199, Bimg)]):
        label = 'image 1' if it < 100 else 'image 2'
        show(axes[row, col], im, P[it], f'Δ = {dl}: iteration {it} ({label})', true_peak(im))
legend_row(fig)
fig.tight_layout(rect=(0, 0.05, 1, 1), h_pad=2.5)
fig.savefig('fig4_switch_snapshots.png', dpi=150, bbox_inches='tight')

# Figure 5: dynamic tasks, % of runs at the brightest area over time
fig, axes = plt.subplots(1, 3, figsize=(12, 3.8), sharey=True)
panels = [('t3', 'Task 3: image switched at iteration 100', 100),
          ('t4', 'Advanced 4: moving object', None),
          ('t5', 'Advanced 5: A fades, B emerges (it. 50–150)', None)]
for ax, (key, title, switch) in zip(axes, panels):
    for dl, c in DELTA_COL.items():
        ok = 100 * np.mean(r[f'{key}_d{dl}_ratio'] >= 0.95, axis=0)
        ax.plot(ok, color=c, lw=2, label=f'Δ = {dl}')
    if switch:
        ax.axvline(switch, color=MUTED, ls='--', lw=1)
    if key == 't5':
        ax.axvspan(50, 150, color=GRID, alpha=0.6, lw=0)
    ax.set_title(title, fontsize=9.5, color=INK)
    ax.set_xlabel('Iteration'); ax.set_ylim(-3, 103)
    ax.spines[['top', 'right']].set_visible(False)
    ax.grid(color=GRID); ax.set_axisbelow(True)
axes[0].set_ylabel('Runs at the brightest area (%)')
axes[0].legend(fontsize=8, loc='center left', bbox_to_anchor=(0.03, 0.35))
fig.tight_layout(); fig.savefig('fig5_dynamic.png', dpi=150, bbox_inches='tight')

# Figure 6: Advanced 6, two equally bright spots
img6 = r['t6_img']; bests = r['t6_bests']; c = r['t6_counts']
fig, axes = plt.subplots(1, 2, figsize=(8, 4.3))
show(axes[0], img6, title=f'Best fly at the end of 30 runs\n(spot 1: {c[0]} runs, spot 2: {c[1]} runs)')
axes[0].scatter(bests[:, 0], bests[:, 1], s=40, color=BEST, edgecolor='white', lw=0.8)
show(axes[1], r['t6_masked'], title='Sequential niching: blank out the first spot,\nthen run DFO again')
for p, lab in [(r['t6_first'], '1st run'), (r['t6_second'], '2nd run')]:
    axes[1].scatter(*p, s=60, color=BEST, edgecolor='white', lw=0.8)
    axes[1].annotate(lab, p, xytext=(8, -12), textcoords='offset points', color='white', fontsize=9)
fig.tight_layout(); fig.savefig('fig6_two_spots.png', dpi=150, bbox_inches='tight')

# Animation: Advanced 4, the swarm tracking a moving object (delta = 0.01)
bg = r['t4_bg']; cen = r['t4_centres']; P = r['t4_d0.01_positions_run0']
H, W = bg.shape
fig, ax = plt.subplots(figsize=(3.4, 3.4), dpi=80)
fig.subplots_adjust(0, 0, 1, 0.92)
im = ax.imshow(bg, cmap='gray', vmin=0, vmax=255 + bg.max())
flies = ax.scatter([], [], s=8, color=FLY, lw=0)
ax.set_xticks([]); ax.set_yticks([])
title = ax.set_title('', fontsize=9)


def frame(t):
    im.set_data(bg + blob(H, W, *cen[t], 10, 255))
    flies.set_offsets(P[t])
    title.set_text(f'Tracking a moving object: iteration {t}')
    return im, flies, title


anim = FuncAnimation(fig, frame, frames=range(0, 200, 2))
anim.save('anim_moving_object.gif', writer=PillowWriter(fps=12))