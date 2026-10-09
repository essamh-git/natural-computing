"""
Week 3 Part 3: DFO parameter study
  - Impact of dimensionality:        D = 2, 10, 30, 50, 100        (delta = 0.001)
  - Impact of disturbance threshold: delta = 0, 1e-4, 1e-3, 1e-2, 0.1   (D = 30)
15 runs per setting, on Sphere and Rastrigin. Everything else as in Part 2:
N = 100 flies, 1000 iterations, bounds [-5.12, 5.12].
The DFO loop is the lecturer's code (github.com/mohmaj/DFO, Python/DFO.py) with D and
delta passed in as arguments.
"""
import numpy as np
from multiprocessing import Pool

N = 100
maxIterations = 1000
RUNS = 15
DIMS = [2, 10, 30, 50, 100]
DELTAS = [0.0, 0.0001, 0.001, 0.01, 0.1]


def sphere(x):
    return float(np.sum(x ** 2))


def rastrigin(x):
    return float(10 * len(x) + np.sum(x ** 2 - 10 * np.cos(2 * np.pi * x)))


FUNCTIONS = {'Sphere': sphere, 'Rastrigin': rastrigin}


def run_dfo(args):
    func_name, seed, D, delta = args
    f = FUNCTIONS[func_name]
    np.random.seed(seed)
    lowerB, upperB = [-5.12] * D, [5.12] * D
    history = []

    X = np.empty([N, D])
    fitness = [None] * N
    for i in range(N):
        for d in range(D):
            X[i, d] = np.random.uniform(lowerB[d], upperB[d])

    for itr in range(maxIterations):
        for i in range(N):
            fitness[i] = f(X[i, ])
        s = np.argmin(fitness)
        history.append(fitness[s])

        for i in range(N):
            if i == s:
                continue
            left, right = (i - 1) % N, (i + 1) % N
            bNeighbour = right if fitness[right] < fitness[left] else left
            for d in range(D):
                if np.random.rand() < delta:
                    X[i, d] = np.random.uniform(lowerB[d], upperB[d])
                    continue
                u = np.random.rand()
                X[i, d] = X[bNeighbour, d] + u * (X[s, d] - X[i, d])
                if X[i, d] < lowerB[d] or X[i, d] > upperB[d]:
                    X[i, d] = np.random.uniform(lowerB[d], upperB[d])

    for i in range(N):
        fitness[i] = f(X[i, ])
    history.append(min(fitness))
    return args, min(fitness), history


def summary(v):
    v = np.array(v)
    return v.mean(), np.median(v), v.min(), v.max(), v.std(ddof=1)


if __name__ == '__main__':
    jobs = []
    for name in FUNCTIONS:
        for D in DIMS:
            jobs += [(name, seed, D, 0.001) for seed in range(RUNS)]
        for dl in DELTAS:
            if dl != 0.001:               # D = 30, delta = 0.001 IS ALREADY IN THE DIMS SET
                jobs += [(name, seed, 30, dl) for seed in range(RUNS)]
    jobs.sort(key=lambda j: -j[2])        # LONGEST RUNS FIRST

    with Pool() as pool:
        results = pool.map(run_dfo, jobs, chunksize=1)

    store = {}
    for (name, seed, D, dl), best, hist in results:
        store.setdefault((name, D, dl), []).append((seed, best, hist))

    out = {}
    for (name, D, dl), runs in store.items():
        runs.sort()
        key = f'{name}_D{D}_delta{dl}'
        out[key + '_final'] = np.array([r[1] for r in runs])
        out[key + '_curves'] = np.array([r[2] for r in runs])
    np.savez('dfo_parameter_study.npz', **out)

    for title, settings in [('Impact of dimensionality (delta = 0.001)', [(D, 0.001) for D in DIMS]),
                            ('Impact of disturbance threshold (D = 30)', [(30, dl) for dl in DELTAS])]:
        print('\n' + title)
        print(f"{'Function':<11}{'D':>5}{'delta':>8}{'Mean':>12}{'Median':>12}{'Min':>12}"
              f"{'Max':>12}{'Std':>12}")
        for name in FUNCTIONS:
            for D, dl in settings:
                m = summary(out[f'{name}_D{D}_delta{dl}_final'])
                print(f'{name:<11}{D:>5}{dl:>8g}' + ''.join(f'{x:>12.3g}' for x in m))