"""
Week 5: Genetic Algorithm (GA) vs Dispersive Flies Optimisation (DFO)

GA:  the Yarpiz course code (ga.py, unchanged) by Mostapha Kalami Heris.
DFO: the lecturer's code (github.com/mohmaj/DFO, Python/DFO.py), as used in Week 3.

Both: 30 dimensions, bounds [-5.12, 5.12], population 100, 1000 iterations,
which is 100,000 fitness evaluations each. 30 runs per algorithm per function.
"""

import io
import contextlib
import numpy as np
from multiprocessing import Pool
from scipy.stats import mannwhitneyu
from ypstruct import structure
import ga

D = 30
NPOP = 100
MAXIT = 1000
LB, UB = -5.12, 5.12
RUNS = 30


# FITNESS FUNCTIONS (MINIMISED, GLOBAL OPTIMUM = 0 AT THE ORIGIN)
def sphere(x):
    return float(np.sum(x ** 2))


def rastrigin(x):
    return float(10 * len(x) + np.sum(x ** 2 - 10 * np.cos(2 * np.pi * x)))


FUNCTIONS = {'Sphere': sphere, 'Rastrigin': rastrigin}


def run_ga(func_name, seed):
    np.random.seed(seed)
    problem = structure()
    problem.costfunc = FUNCTIONS[func_name]
    problem.nvar = D
    problem.varmin = LB
    problem.varmax = UB

    params = structure()
    params.maxit = MAXIT
    params.npop = NPOP
    params.beta = 1
    params.pc = 1
    params.gamma = 0.1
    params.mu = 0.01
    params.sigma = 0.1

    with contextlib.redirect_stdout(io.StringIO()):   # ga.py PRINTS EVERY ITERATION
        out = ga.run(problem, params)
    return out.bestcost


def run_dfo(func_name, seed, delta=0.001):
    f = FUNCTIONS[func_name]
    np.random.seed(seed)
    N = NPOP
    lowerB, upperB = [LB] * D, [UB] * D
    X = np.empty([N, D])
    fitness = [None] * N
    for i in range(N):
        for d in range(D):
            X[i, d] = np.random.uniform(lowerB[d], upperB[d])

    history = []
    for itr in range(MAXIT):
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
    return np.array(history)


def job(args):
    alg, func_name, seed = args
    curve = run_ga(func_name, seed) if alg == 'GA' else run_dfo(func_name, seed)
    return args, curve


if __name__ == '__main__':
    jobs = [(alg, fn, s) for alg in ('GA', 'DFO') for fn in FUNCTIONS for s in range(RUNS)]
    with Pool() as pool:
        results = pool.map(job, jobs, chunksize=1)

    out = {}
    for (alg, fn, s), curve in results:
        out.setdefault(f'{alg}_{fn}', []).append(curve)
    out = {k: np.array(v) for k, v in out.items()}
    np.savez('week5_results.npz', **out)

    print(f'D = {D}, population = {NPOP}, {MAXIT} iterations (100,000 evaluations), {RUNS} runs\n')
    print(f"{'Function':<11}{'Algorithm':<11}{'Mean':>12}{'Median':>12}{'Min':>12}{'Max':>12}{'Std':>12}")
    for fn in FUNCTIONS:
        for alg in ('GA', 'DFO'):
            v = out[f'{alg}_{fn}'][:, -1]
            print(f'{fn:<11}{alg:<11}{v.mean():>12.4g}{np.median(v):>12.4g}{v.min():>12.4g}'
                  f'{v.max():>12.4g}{v.std(ddof=1):>12.4g}')
        p = mannwhitneyu(out[f'GA_{fn}'][:, -1], out[f'DFO_{fn}'][:, -1]).pvalue
        print(f'{"":<11}Mann-Whitney U test p-value: {p:.3g}\n')