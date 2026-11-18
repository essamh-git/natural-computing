"""
Week 7 Part 2: SDS when the search space is a large image and the model is a small image.

Same structure as the lecturer's SDS.py (github.com/mohmaj/SDS):
  - hypothesis = (x, y), the top-left corner where the small image might sit in the large one
  - test phase: each agent checks ONE micro-feature of the model at its hypothesis
  - diffusion phase: an inactive agent polls a random agent; it copies that agent's hypothesis
    if it is active, otherwise picks a new random hypothesis; active agents re-test
Only the micro-feature changes between versions. Four micro-features are compared:

  'pixel_exact'   one random pixel of the model must have exactly the same grey value
  'pixel_tol'     one random pixel must be within TOL grey levels
  'patch_tol'     the mean of a random 3x3 block must be within TOL grey levels
  'pixel_pair'    pick two random pixels of the model; the same one must be brighter in the
                  image too (only the ORDER matters, so it ignores brightness changes)
"""
import numpy as np

TOL = 12        # GREY-LEVEL TOLERANCE (OUT OF 255)
MICRO = ['pixel_exact', 'pixel_tol', 'patch_tol', 'pixel_pair']


def make_test(ss, model, kind, rng):
    """Return a function test(x, y) -> True/False that checks one random micro-feature."""
    h, w = model.shape
    ss, model = ss.astype(float), model.astype(float)

    if kind == 'pixel_exact':
        def test(x, y):
            i, j = rng.integers(h), rng.integers(w)
            return ss[y + i, x + j] == model[i, j]
    elif kind == 'pixel_tol':
        def test(x, y):
            i, j = rng.integers(h), rng.integers(w)
            return abs(ss[y + i, x + j] - model[i, j]) <= TOL
    elif kind == 'patch_tol':
        def test(x, y):
            i, j = rng.integers(h - 2), rng.integers(w - 2)
            return abs(ss[y + i:y + i + 3, x + j:x + j + 3].mean() - model[i:i + 3, j:j + 3].mean()) <= TOL
    elif kind == 'pixel_pair':
        def test(x, y):
            (i1, i2), (j1, j2) = rng.integers(h, size=2), rng.integers(w, size=2)
            dm = model[i1, j1] - model[i2, j2]
            di = ss[y + i1, x + j1] - ss[y + i2, x + j2]
            return np.sign(dm) == np.sign(di)
    else:
        raise ValueError(kind)
    return test


def sds_image(ss, model, kind, N=100, maxIter=1500, seed=0):
    rng = np.random.default_rng(seed)
    H, W = ss.shape
    h, w = model.shape
    maxX, maxY = W - w + 1, H - h + 1
    test = make_test(ss, model, kind, rng)

    hx, hy = rng.integers(maxX, size=N), rng.integers(maxY, size=N)
    status = np.zeros(N, dtype=bool)
    activity, best_pos, best_size = [], [], []

    for itr in range(maxIter):
        active = 0
        for i in range(N):        # TEST PHASE
            status[i] = test(hx[i], hy[i])
            active += status[i]
        for i in range(N):        # DIFFUSION PHASE
            if not status[i]:
                r = rng.integers(N)
                if status[r]:
                    hx[i], hy[i] = hx[r], hy[r]
                else:
                    hx[i], hy[i] = rng.integers(maxX), rng.integers(maxY)
            else:
                status[i] = test(hx[i], hy[i])

        # LARGEST CLUSTER: THE HYPOTHESIS HELD BY THE MOST AGENTS
        keys = hy * maxX + hx
        vals, counts = np.unique(keys, return_counts=True)
        k = vals[np.argmax(counts)]
        activity.append(100 * active / N)
        best_pos.append((k % maxX, k // maxX))
        best_size.append(100 * counts.max() / N)
    return np.array(activity), np.array(best_pos), np.array(best_size)