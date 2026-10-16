"""
Week 4 experiments: DFO finding the brightest area in an image.
Task 1: find the brightest area (and the effect of the disturbance threshold in 2D)
Task 2: other input images (synthetic and real photos)
Task 3: change the input image half-way through
Advanced 4: trace a moving object
Advanced 5: a fading object and an emerging object
Advanced 6: two equally bright spots (and sequential niching to find both)
Saves all results to week4_results.npz for week4_figures.py.
"""
import numpy as np
from DFOImage import synthetic_image, real_image, brightness_map, blob, dfo_image

RUNS = 30
N = 30
out = {}


def success(img, run, frac=0.95):
    return run['best_fit'][-1] >= frac * brightness_map(img).max()


# Task 1
print('Task 1: success rate (brightness >= 95% of the maximum) on synthetic images 1-3')
synth = {s: synthetic_image(s)[0] for s in (1, 2, 3)}
DELTAS = [0.0, 0.001, 0.01, 0.05, 0.1, 0.2]
rates = []
for dl in DELTAS:
    ok = sum(success(synth[s], dfo_image(lambda t: synth[s], N=N, delta=dl, seed=r))
             for s in synth for r in range(RUNS))
    rates.append(ok / (len(synth) * RUNS))
    print(f'  delta = {dl:<6} success {ok}/{len(synth) * RUNS} = {100 * ok / (len(synth) * RUNS):.0f}%')
out['t1_deltas'] = np.array(DELTAS); out['t1_rates'] = np.array(rates)
r1 = dfo_image(lambda t: synth[1], N=N, delta=0.1, seed=0)
out['t1_positions'] = r1['positions']; out['t1_img'] = synth[1]

# Task 2
print('\nTask 2: other input images (delta = 0.1)')
images = {'Synthetic 2': synth[2], 'Synthetic 3': synth[3],
          'Coins': real_image('coins'), 'Hubble Deep Field': real_image('hubble_deep_field')}
for name, img in images.items():
    row = []
    for it in (100, 300):
        ok = sum(success(img, dfo_image(lambda t: img, N=N, delta=0.1, seed=r, maxIterations=it))
                 for r in range(RUNS))
        row.append(ok)
    print(f'  {name:<18} 100 iterations: {row[0]}/30   300 iterations: {row[1]}/30')
    key = name.replace(' ', '_')
    out[f't2_{key}_img'] = img
    out[f't2_{key}_rates'] = np.array(row)
    out[f't2_{key}_final'] = dfo_image(lambda t: img, N=N, delta=0.1, seed=1)['positions'][-1]


# dynamic tasks helper
def dynamic(frames, deltas, name, iters, target_at=None):
    """Run DFO on a changing image; record brightness ratio (and tracking error) per iteration."""
    Bmax = np.array([brightness_map(f).max() for f in frames])
    for dl in deltas:
        ratios, errors, finals = [], [], []
        for r in range(RUNS):
            res = dfo_image(lambda t: frames[min(t, len(frames) - 1)], N=N,
                            maxIterations=iters, delta=dl, seed=r)
            ratios.append(res['best_fit'] / Bmax[:iters])
            if target_at is not None:
                tgt = np.array([target_at(t) for t in range(iters)])
                errors.append(np.linalg.norm(res['best_pos'] - tgt, axis=1))
            finals.append(res)
        out[f'{name}_d{dl}_ratio'] = np.array(ratios)
        if errors:
            out[f'{name}_d{dl}_error'] = np.array(errors)
        out[f'{name}_d{dl}_positions_run0'] = finals[0]['positions']
        end_ok = np.mean(np.array(ratios)[:, -1] >= 0.95)
        msg = f'  delta = {dl:<6} success at the end: {100 * end_ok:.0f}%'
        if errors:
            tracked = np.mean(np.array(errors)[:, iters // 2:] < 10)
            msg += f'   time within 10 px of the object (2nd half): {100 * tracked:.0f}%'
        print(msg)


# Task 3
print('\nTask 3: switch from image 1 to image 2 at iteration 100 (200 iterations)')
frames3 = [synth[1]] * 100 + [synth[2]] * 100
out['t3_imgA'], out['t3_imgB'] = synth[1], synth[2]
dynamic(frames3, [0.001, 0.01, 0.1], 't3', 200)


# Advanced 4: moving object
print('\nAdvanced 4: bright object moving in a circle (radius 60 px, one lap in 200 iterations)')
bg, _ = synthetic_image(4, bright=False)
bg = 0.55 * bg
H, W = bg.shape
centre = lambda t: (100 + 60 * np.cos(2 * np.pi * t / 200), 100 + 60 * np.sin(2 * np.pi * t / 200))
frames4 = [bg + blob(H, W, *centre(t), 10, 255) for t in range(200)]
dynamic(frames4, [0.001, 0.01, 0.1], 't4', 200, target_at=centre)
out['t4_centres'] = np.array([centre(t) for t in range(200)])
out['t4_bg'] = bg


# Advanced 5: fading / emerging
print('\nAdvanced 5: object A fades out and object B emerges between iterations 50 and 150')
pA, pB = (50, 150), (150, 50)
def frame5(t):
    a = np.clip(1 - (t - 50) / 100, 0, 1)            # A: 1 -> 0
    return bg + blob(H, W, *pA, 12, 255 * a) + blob(H, W, *pB, 12, 255 * (1 - a))
frames5 = [frame5(t) for t in range(200)]
dynamic(frames5, [0.001, 0.01, 0.1], 't5', 200)


# Advanced 6: two bright spots
print('\nAdvanced 6: two equally bright spots')
s1, s2 = (55, 60), (145, 140)
Bbg = brightness_map(bg)
top = max(Bbg[s1[1], s1[0]], Bbg[s2[1], s2[0]])
amp = lambda p: 255 + top - Bbg[p[1], p[0]]        # TOP UP THE DIMMER SPOT SO BOTH PEAKS MATCH
img6 = bg + blob(H, W, *s1, 12, amp(s1)) + blob(H, W, *s2, 12, amp(s2))
B6 = brightness_map(img6)
print('  peak brightness: spot 1 %.1f, spot 2 %.1f' % (B6[s1[1]-15:s1[1]+16, s1[0]-15:s1[0]+16].max(),
                                                    B6[s2[1]-15:s2[1]+16, s2[0]-15:s2[0]+16].max()))
bests = np.array([dfo_image(lambda t: img6, N=N, delta=0.1, seed=r)['best_pos'][-1] for r in range(RUNS)])
near1 = np.sum(np.linalg.norm(bests - s1, axis=1) < 15)
near2 = np.sum(np.linalg.norm(bests - s2, axis=1) < 15)
print(f'  final best fly: spot 1 in {near1} runs, spot 2 in {near2} runs (never both in one run)')
# SEQUENTIAL NICHING: FIND ONE SPOT, BLANK IT OUT, RUN AGAIN
first = dfo_image(lambda t: img6, N=N, delta=0.1, seed=0)['best_pos'][-1]
yy, xx = np.mgrid[0:H, 0:W]
masked = img6.copy()
masked[(xx - first[0]) ** 2 + (yy - first[1]) ** 2 < 25 ** 2] = 0
second = dfo_image(lambda t: masked, N=N, delta=0.1, seed=0)['best_pos'][-1]
print(f'  sequential niching: run 1 found {np.round(first)}, run 2 (spot masked) found {np.round(second)}')
out.update(t6_img=img6, t6_bests=bests, t6_first=first, t6_second=second, t6_masked=masked,
           t6_counts=np.array([near1, near2]))

np.savez_compressed('week4_results.npz', **out)