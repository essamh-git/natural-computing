"""
Week 7 Part 2 experiments: which micro-feature works best when searching an image for a small image?
Search space: the 'camera' test image (scikit-image), resized to 128x128.
Model: the 16x16 block containing the cameraman's face and camera (top-left corner x=64, y=38).
Three conditions for the large image: exact copy, Gaussian noise (sd 10), brighter (+40).
30 runs per micro-feature per condition, 100 agents, 1500 iterations.
Saves week7_micro_results.npz for Week7MicroFigures.py.
"""
import numpy as np
from multiprocessing import Pool
from skimage import data, transform
from SDSImage import sds_image, MICRO, TOL

RUNS, N, ITERS = 30, 100, 1500
TX, TY, S = 64, 38, 16
clean = transform.resize(data.camera(), (128, 128), anti_aliasing=True, preserve_range=True).round()
model = clean[TY:TY + S, TX:TX + S].copy()
noise = np.random.default_rng(123).normal(0, 10, clean.shape)
CONDITIONS = {
    'Exact': clean,
    'Noisy (sd 10)': np.clip(clean + noise, 0, 255).round(),
    'Brighter (+40)': np.clip(clean + 40, 0, 255),
}


def job(args):
    cond, kind, seed = args
    act, pos, size = sds_image(CONDITIONS[cond], model, kind, N=N, maxIter=ITERS, seed=seed)
    on_target = (np.abs(pos[:, 0] - TX) <= 1) & (np.abs(pos[:, 1] - TY) <= 1)
    found = on_target & (size >= 50)       # A CLUSTER OF AT LEAST HALF THE AGENTS ON THE FACE
    first = int(np.flatnonzero(found)[0]) + 1 if found.any() else -1
    exact_pos = int(pos[-1, 0] == TX and pos[-1, 1] == TY)
    return cond, kind, seed, act, size, on_target, first, exact_pos


if __name__ == '__main__':
    jobs = [(c, k, s) for c in CONDITIONS for k in MICRO for s in range(RUNS)]
    with Pool() as pool:
        res = pool.map(job, jobs, chunksize=2)
    out = {'model': model, 'target': np.array([TX, TY]), 'tol': TOL}
    for c, img in CONDITIONS.items():
        out[f'img_{c}'] = img
    print(f'{RUNS} runs, {N} agents, {ITERS} iterations; found = cluster of >= 50% of agents within 1 px of the face\n')
    print(f"{'Condition':<16}{'Micro-feature':<14}{'Found':>7}{'Median it.':>12}{'Exact pos.':>12}{'Final activity':>16}")
    for c in CONDITIONS:
        for k in MICRO:
            rs = [r for r in res if r[0] == c and r[1] == k]
            first = np.array([r[6] for r in rs])
            ok = first > 0
            exact = sum(r[7] for r in rs if r[6] > 0)
            act = np.mean([r[3][-1] for r in rs])
            med = f'{np.median(first[ok]):.0f}' if ok.any() else '-'
            print(f'{c:<16}{k:<14}{ok.sum():>4}/{RUNS}{med:>12}{exact:>9}/{max(ok.sum(), 1)}{act:>15.0f}%')
            key = f'{c.split()[0]}_{k}'
            out[f'{key}_first'] = first
            out[f'{key}_activity'] = np.array([r[3] for r in rs])
            out[f'{key}_size'] = np.array([r[4] * r[5] for r in rs])     # SIZE OF CLUSTER ON THE FACE
        print()
    # HOW OFTEN DOES ONE MICRO-FEATURE TEST PASS AT THE TRUE POSITION? (THE 'SCORE' OF THE MATCH)
    from SDSImage import make_test
    rng = np.random.default_rng(0)
    print('Chance that one micro-feature test passes at the true position (10,000 tests):')
    for c, img in CONDITIONS.items():
        row = []
        for k in MICRO:
            t = make_test(img, model, k, rng)
            p = np.mean([t(TX, TY) for _ in range(10000)])
            out[f'pass_{c.split()[0]}_{k}'] = p
            row.append(f'{k} {100 * p:.0f}%')
        print(f'  {c:<16}' + ', '.join(row))
    np.savez_compressed('week7_micro_results.npz', **out)