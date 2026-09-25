"""
No Free Lunch Theorem - empirical demonstration
Natural Computing, Week 1 - Mohammed Essam Hasin

Search space X = {0, ..., 7} (8 points on a line), cost values Y = {0, 1, 2, 3}.
That gives 4^8 = 65,536 possible objective functions, so we can test EVERY one.

Four deterministic, non-revisiting black-box algorithms try to find the maximum:
  1. Sweep          - evaluates points left to right (ignores the values it sees)
  2. Random order   - evaluates points in a fixed shuffled order (random search)
  3. Hill climber   - moves towards better values (uses what it sees)
  4. Hill descender - deliberately moves towards WORSE values

NFL predicts all four have identical performance when averaged over all 65,536 functions,
but different performance on structured subsets of functions.
"""

import itertools
import random
import numpy as np
import matplotlib.pyplot as plt

N = 8                       # size of search space
Y = [0, 1, 2, 3]            # possible cost values
START = 3                   # starting point for the local-search algorithms


# algorithms
# Each algorithm takes a function f (tuple of length N) and returns the ORDER
# in which it evaluates the points. None of them revisits a point.

def sweep(f):
    return list(range(N))


RANDOM_ORDER = list(range(N))
random.Random(42).shuffle(RANDOM_ORDER)


def random_order(f):
    return RANDOM_ORDER


def local_search(f, prefer_better=True):
    """
    Hill climber (prefer_better=True) or hill descender (False).
    Always explores the unvisited neighbours of its current point; if it moves
    to a better (or, for the descender, worse) point it continues from there.
    When stuck, it jumps to the nearest unvisited point.
    """
    visited = [START]
    current = START
    while len(visited) < N:
        nbrs = [x for x in (current - 1, current + 1)
                if 0 <= x < N and x not in visited]
        if not nbrs:  # stuck: jump to nearest unvisited point
            nbrs = [min((x for x in range(N) if x not in visited),
                        key=lambda x: (abs(x - current), x))]
        nxt = nbrs[0]
        visited.append(nxt)
        better = f[nxt] > f[current]
        worse = f[nxt] < f[current]
        if (prefer_better and better) or (not prefer_better and worse):
            current = nxt
    return visited


def hill_climber(f):
    return local_search(f, prefer_better=True)


def hill_descender(f):
    return local_search(f, prefer_better=False)


ALGORITHMS = {
    "Sweep": sweep,
    "Random order": random_order,
    "Hill climber": hill_climber,
    "Hill descender": hill_descender,
}


# measures
def best_so_far(f, order):
    """
    Best value found after 1, 2, ..., N evaluations.
    """
    return np.maximum.accumulate([f[x] for x in order])


def evals_to_optimum(f, order):
    """
    Number of evaluations needed to first see the global maximum.
    """
    target = max(f)
    for i, x in enumerate(order, start=1):
        if f[x] == target:
            return i


# problem subsets
def is_unimodal(f):
    """
    Rises (or stays flat) to a single peak, then falls (or stays flat).
    """
    peak = f.index(max(f))
    return (all(f[i] <= f[i + 1] for i in range(peak)) and
            all(f[i] >= f[i + 1] for i in range(peak, N - 1)))


def is_deceptive(f):
    """
    Unique maximum hidden in a 'valley': every neighbour of the peak
    has the lowest possible value (0).
    """
    m = max(f)
    if m == 0 or f.count(m) != 1:
        return False
    p = f.index(m)
    return all(f[q] == 0 for q in (p - 1, p + 1) if 0 <= q < N)


# run
def main():
    all_functions = list(itertools.product(Y, repeat=N))
    print(f"Testing {len(all_functions):,} functions")

    subsets = {
        "All functions": lambda f: True,
        "Unimodal": is_unimodal,
        "Deceptive": is_deceptive,
    }

    curves = {s: {a: np.zeros(N) for a in ALGORITHMS} for s in subsets}
    evals = {s: {a: [] for a in ALGORITHMS} for s in subsets}
    counts = {s: 0 for s in subsets}

    for f in all_functions:
        members = [s for s, test in subsets.items() if test(f)]
        for s in members:
            counts[s] += 1
        for name, alg in ALGORITHMS.items():
            order = alg(f)
            bsf = best_so_far(f, order)
            e = evals_to_optimum(f, order)
            for s in members:
                curves[s][name] += bsf
                evals[s][name].append(e)

    for s in subsets:
        for a in ALGORITHMS:
            curves[s][a] /= counts[s]

    # results table
    print("\nMean evaluations needed to find the global maximum (lower is better)")
    header = f"{'Subset':<15}{'n':>8}" + "".join(f"{a:>16}" for a in ALGORITHMS)
    print(header)
    rows = []
    for s in subsets:
        means = [np.mean(evals[s][a]) for a in ALGORITHMS]
        rows.append((s, counts[s], means))
        print(f"{s:<15}{counts[s]:>8,}" + "".join(f"{m:>16.4f}" for m in means))

    with open("nfl_results.csv", "w") as fh:
        fh.write("Subset,Functions," + ",".join(ALGORITHMS) + "\n")
        for s, n, means in rows:
            fh.write(f"{s},{n}," + ",".join(f"{m:.4f}" for m in means) + "\n")

    colours = {"Sweep": "#4C72B0", "Random order": "#55A868",
               "Hill climber": "#C44E52", "Hill descender": "#8172B2"}
    styles = {"Sweep": "-", "Random order": "--",
              "Hill climber": ":", "Hill descender": "-."}
    x = np.arange(1, N + 1)

    # Figure 1: all functions - curves overlap exactly
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for a in ALGORITHMS:
        ax.plot(x, curves["All functions"][a], styles[a], color=colours[a],
                lw=2.5, marker="o", ms=4, label=a)
    ax.set_xlabel("Number of evaluations")
    ax.set_ylabel("Mean best value found")
    ax.set_title(f"Averaged over all {counts['All functions']:,} functions:\n"
                 "every algorithm performs identically")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig("fig1_all_functions.png", dpi=150)

    # Figure 2: structured subsets - bar chart of evaluations to optimum
    fig, ax = plt.subplots(figsize=(8, 4.8))
    names = list(subsets)
    width = 0.2
    for i, a in enumerate(ALGORITHMS):
        vals = [np.mean(evals[s][a]) for s in names]
        bars = ax.bar(np.arange(len(names)) + (i - 1.5) * width, vals, width,
                      color=colours[a], label=a)
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, v + 0.05, f"{v:.2f}",
                    ha="center", va="bottom", fontsize=7)
    ax.set_xticks(np.arange(len(names)))
    ax.set_xticklabels([f"{s}\n(n={counts[s]:,})" for s in names])
    ax.set_ylabel("Mean evaluations to find maximum\n(lower is better)")
    ax.set_title("Equal on average, but rankings change on structured problems")
    ax.legend(ncol=4, fontsize=8, loc="upper center",
              bbox_to_anchor=(0.5, -0.18))
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig("fig2_subsets.png", dpi=150)

    # Figure 3: per-function difference, hill climber vs random order
    diff = (np.array(evals["All functions"]["Random order"]) -
            np.array(evals["All functions"]["Hill climber"]))
    fig, ax = plt.subplots(figsize=(7, 4.5))
    bins = np.arange(diff.min() - 0.5, diff.max() + 1.5)
    ax.hist(diff, bins=bins, color="#C44E52", edgecolor="white")
    ax.axvline(0, color="black", lw=1)
    ax.set_xlabel("Evaluations saved by hill climber vs random order\n"
                  "(positive = hill climber faster, negative = slower)")
    ax.set_ylabel("Number of functions")
    ax.set_title(f"Every gain is paid for elsewhere: mean difference = "
                 f"{diff.mean():.4f}")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig("fig3_tradeoff.png", dpi=150)

    print(f"\nHill climber faster on {np.sum(diff > 0):,} functions, "
          f"slower on {np.sum(diff < 0):,}, equal on {np.sum(diff == 0):,}")
    print(f"Mean difference: {diff.mean():.6f}")


if __name__ == "__main__":
    main()