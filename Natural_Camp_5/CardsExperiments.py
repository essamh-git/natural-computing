"""
Card game experiments: 50 runs each of the generational GA, the steady-state (Microbial) GA
and random search, with the same budget of fitness evaluations.
Prints the results table and saves cards_results.npz for cards_figures.py.
"""
import itertools
import numpy as np
import CardsGA as C

RUNS = 50
BUDGET = 400     # = 20 GENERATIONS OF 20 (GENERATIONAL) OR 380 ITERATIONS (STEADY STATE)
LONG = 3000      # LONGER RUNS FOR THE SUCCESS-RATE CURVE

# BRUTE FORCE: CHECK ALL 1024 GENOTYPES
all_g = [list(g) for g in itertools.product([0, 1], repeat=C.L)]
errs = np.array([C.error(g) for g in all_g])
opt = all_g[int(np.argmin(errs))]
print(f'Search space: {len(all_g)} genotypes; perfect solutions: {np.sum(errs == 0)}')
print(f'  sum pile {C.decode(opt)[0]} = 36, product pile {C.decode(opt)[1]} = 360\n')

METHODS = {'Generational GA': C.generational_ga, 'Steady-state GA': C.steady_state_ga,
           'Random search': C.random_search}
out = {'all_errors': errs, 'optimum': np.array(opt)}
print(f'{BUDGET} evaluations per run, {RUNS} runs')
print(f"{'Method':<17}{'Solved':>8}{'Mean':>9}{'Median':>9}{'Min':>7}{'Max':>8}{'Variance':>11}{'Std':>8}")
for name, f in METHODS.items():
    H = np.array([f(s, LONG) for s in range(RUNS)])
    fin = H[:, BUDGET - 1]
    key = name.split()[0]
    out[f'{key}_history'] = H
    print(f'{name:<17}{np.sum(fin == 0):>5}/{RUNS}{fin.mean():>9.4f}{np.median(fin):>9.4f}{fin.min():>7.3f}'
          f'{fin.max():>8.4f}{fin.var(ddof=1):>11.6f}{fin.std(ddof=1):>8.4f}')
np.savez('cards_results.npz', **out)

print('\nRuns solved by number of evaluations:')
for k in (100, 400, 1000, 3000):
    print(f'  {k:>5}: ' + ', '.join(f"{n.split()[0]} {np.sum(out[n.split()[0] + '_history'][:, k - 1] == 0)}"
                                     for n in METHODS))