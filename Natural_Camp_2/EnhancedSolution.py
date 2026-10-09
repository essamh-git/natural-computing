"""
Week 2 Advanced Exercise: enhancing the random solutions
Compares, with the SAME budget of fitness evaluations:
  - Random search: keep sampling new random solutions, keep the best
  - Hill climber ((1+1)-Evolution Strategy): start from the best of the 50
    random solutions, add small Gaussian changes, keep the change only if it
    improves fitness. The step size adapts with the 1/5 success rule.
Run on both the Sphere (func 1) and Rastrigin (func 2) functions.
"""
import math as m
import random as rnd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

D, N = 30, 50
lower_bound, upper_bound = -5.12, 5.12
BUDGET = 5000          # EXTRA FITNESS EVALUATIONS AFTER THE FIRST 50
RUNS = 10              # INDEPENDENT RUNS (DIFFERENT SEEDS)


def fitness_function(func_no, x):
    if func_no == 1:   # SPHERE
        s = 0.0
        for v in x:
            s += v * v
    elif func_no == 2: # RASTRIGIN
        s = 10.0 * len(x)
        for v in x:
            s += v * v - 10.0 * m.cos(2 * m.pi * v)
    return s


def random_solution():
    return [lower_bound + rnd.random() * (upper_bound - lower_bound) for _ in range(D)]


def initial_population(func_no):
    X = [random_solution() for _ in range(N)]
    fit = [fitness_function(func_no, x) for x in X]
    b = int(np.argmin(fit))
    return X[b], fit[b]


def random_search(func_no, best, best_f):
    history = []
    for _ in range(BUDGET):
        x = random_solution()
        f = fitness_function(func_no, x)
        if f < best_f:
            best, best_f = x, f
        history.append(best_f)
    return best, history


def hill_climber(func_no, best, best_f, sigma=1.0):
    history, successes = [], 0
    for t in range(1, BUDGET + 1):
        # MUTATE: SMALL GAUSSIAN CHANGE, CLIPPED TO THE BOUNDS
        y = [min(upper_bound, max(lower_bound, v + rnd.gauss(0, sigma))) for v in best]
        f = fitness_function(func_no, y)
        if f < best_f:                       # KEEP ONLY IMPROVEMENTS
            best, best_f = y, f
            successes += 1
        if t % 20 == 0:                      # 1/5 SUCCESS RULE EVERY 20 STEPS
            sigma *= 1.22 if successes / 20 > 0.2 else 0.82
            successes = 0
        history.append(best_f)
    return best, history


results = {}
for func_no, name in [(1, 'Sphere'), (2, 'Rastrigin')]:
    for alg_name, alg in [('Random search', random_search), ('Hill climber', hill_climber)]:
        finals, curves = [], []
        for run in range(RUNS):
            rnd.seed(run)
            best, best_f = initial_population(func_no)   # SAME START FOR BOTH
            _, hist = alg(func_no, best, best_f)
            finals.append(hist[-1])
            curves.append([best_f] + hist)
        results[(name, alg_name)] = (np.array(finals), np.array(curves))

print(f'Best fitness after {N} + {BUDGET} evaluations, over {RUNS} runs')
print(f"{'Function':<11}{'Algorithm':<15}{'Start (best of 50)':>20}{'Mean final':>14}{'Std':>10}{'Best run':>12}")
for (name, alg_name), (finals, curves) in results.items():
    print(f'{name:<11}{alg_name:<15}{curves[:, 0].mean():>20.2f}{finals.mean():>14.4g}'
          f'{finals.std():>10.4g}{finals.min():>12.4g}')

# FIGURE: CONVERGENCE (MEDIAN OVER RUNS, SHADED = MIN TO MAX)
BLUE, ORANGE = '#2a78d6', '#eb6834'
INK, MUTED, GRID, SURF = '#0b0b0b', '#52514e', '#e4e3df', '#fcfcfb'
plt.rcParams.update({'axes.edgecolor': MUTED, 'axes.labelcolor': INK,
                     'xtick.color': MUTED, 'ytick.color': MUTED,
                     'axes.spines.top': False, 'axes.spines.right': False,
                     'figure.facecolor': SURF, 'axes.facecolor': SURF})
fig, axes = plt.subplots(1, 2, figsize=(11, 4.3))
evals = np.arange(N, N + BUDGET + 1)
for ax, name in zip(axes, ['Sphere', 'Rastrigin']):
    for alg_name, col in [('Random search', BLUE), ('Hill climber', ORANGE)]:
        c = results[(name, alg_name)][1]
        med = np.median(c, axis=0)
        ax.fill_between(evals, c.min(0), c.max(0), color=col, alpha=0.15, lw=0)
        ax.plot(evals, med, color=col, lw=2, label=alg_name)
        ax.text(evals[-1] * 1.02, med[-1], f'{med[-1]:.3g}', va='center',
                fontsize=8, color=INK)
    ax.set_yscale('log')
    ax.set_xlim(0, N + BUDGET + 700)
    ax.set_xlabel('Fitness evaluations')
    ax.set_ylabel('Best fitness so far (log scale)')
    ax.set_title(name, color=INK)
    ax.grid(color=GRID, which='major'); ax.set_axisbelow(True)
    ax.legend(fontsize=8, loc='lower left')
fig.suptitle('Random search vs hill climber with the same budget (median of 10 runs)',
             color=INK)
fig.tight_layout()
fig.savefig('fig8_convergence.png', dpi=150, bbox_inches='tight')