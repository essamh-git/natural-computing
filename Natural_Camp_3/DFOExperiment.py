"""
Week 3: Dispersive Flies Optimisation (DFO), 30 runs on Sphere and Rastrigin

The DFO update loop is the lecturer's code (github.com/mohmaj/DFO, Python/DFO.py),
unchanged except that it is wrapped in a function so it can be run 30 times with
a different seed each run, and it records the best fitness at every iteration.
"""
import numpy as np
from multiprocessing import Pool

N = 100                 # POPULATION SIZE
D = 30                  # DIMENSIONALITY
delta = 0.001           # DISTURBANCE THRESHOLD
maxIterations = 1000    # ITERATIONS ALLOWED
lowerB = [-5.12] * D    # LOWER BOUND (IN ALL DIMENSIONS)
upperB = [ 5.12] * D    # UPPER BOUND (IN ALL DIMENSIONS)
RUNS = 30


# FITNESS FUNCTIONS (MINIMISED, GLOBAL OPTIMUM = 0 AT THE ORIGIN)
def sphere(x):
    return float(np.sum(x ** 2))


def rastrigin(x):
    return float(10 * len(x) + np.sum(x ** 2 - 10 * np.cos(2 * np.pi * x)))


FUNCTIONS = {'Sphere': sphere, 'Rastrigin': rastrigin}


def run_dfo(args):
    func_name, seed, dlt = args
    f = FUNCTIONS[func_name]
    np.random.seed(seed)
    history = []

    # INITIALISATION PHASE
    X = np.empty([N, D])
    fitness = [None] * N
    for i in range(N):
        for d in range(D):
            X[i, d] = np.random.uniform(lowerB[d], upperB[d])

    # MAIN DFO LOOP
    for itr in range(maxIterations):
        for i in range(N):               # EVALUATION
            fitness[i] = f(X[i, ])
        s = np.argmin(fitness)           # FIND BEST FLY
        history.append(fitness[s])

        for i in range(N):
            if i == s:
                continue                 # ELITIST STRATEGY

            # FIND BEST NEIGHBOUR (RING TOPOLOGY)
            left = (i - 1) % N
            right = (i + 1) % N
            bNeighbour = right if fitness[right] < fitness[left] else left

            for d in range(D):                   # UPDATE EACH DIMENSION SEPARATELY
                if np.random.rand() < dlt:       # DISTURBANCE: RANDOM RESTART OF THIS COMPONENT
                    X[i, d] = np.random.uniform(lowerB[d], upperB[d])
                    continue
                u = np.random.rand()
                X[i, d] = X[bNeighbour, d] + u * (X[s, d] - X[i, d])

                # OUT OF BOUND CONTROL
                if X[i, d] < lowerB[d] or X[i, d] > upperB[d]:
                    X[i, d] = np.random.uniform(lowerB[d], upperB[d])

    for i in range(N):
        fitness[i] = f(X[i, ])          # FINAL EVALUATION
    s = np.argmin(fitness)
    history.append(fitness[s])
    return func_name, seed, fitness[s], history


if __name__ == '__main__':
    jobs = [(name, seed, delta) for name in FUNCTIONS for seed in range(RUNS)]
    with Pool() as pool:
        results = pool.map(run_dfo, jobs)

    finals = {name: [] for name in FUNCTIONS}
    curves = {name: [] for name in FUNCTIONS}
    for name, seed, best, hist in results:
        finals[name].append(best)
        curves[name].append(hist)

    print(f'DFO: N={N}, D={D}, delta={delta}, {maxIterations} iterations, {RUNS} runs\n')
    print(f"{'Function':<11}{'Mean':>14}{'Median':>14}{'Min':>14}{'Max':>14}{'Std':>14}")
    for name in FUNCTIONS:
        v = np.array(finals[name])
        print(f'{name:<11}{v.mean():>14.4g}{np.median(v):>14.4g}{v.min():>14.4g}'
              f'{v.max():>14.4g}{v.std(ddof=1):>14.4g}')

    # EXTRA CHECK: RASTRIGIN WITH THE DISTURBANCE SWITCHED OFF (10 RUNS)
    with Pool() as pool:
        no_dist = pool.map(run_dfo, [('Rastrigin', seed, 0.0) for seed in range(10)])
    v = np.array([res[2] for res in no_dist])
    print(f'\nRastrigin with delta = 0 (10 runs): mean {v.mean():.2f}, '
          f'min {v.min():.2f}, max {v.max():.2f}')

    np.savez('dfo_results.npz',
             **{f'{n}_final': np.array(finals[n]) for n in FUNCTIONS},
             **{f'{n}_curves': np.array(curves[n]) for n in FUNCTIONS})