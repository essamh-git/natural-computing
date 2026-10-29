"""
Improved versions of the tutorial's GA for the Travelling Salesman Problem.
Same steps as the tutorial (elites + fitness-proportionate selection, ordered crossover,
per-city mutation), with three optional improvements that can be switched on one at a time:

  fix_elitism  the tutorial mutates routes in place, including the elite routes, and the same
               route object can appear several times in the mating pool, so the best route
               can get worse. Here children are copies and elites are never mutated.
  inversion    instead of swapping two cities, reverse the section of the route between them
               (keeps more of the route's edges intact).
  two_opt      apply 2-opt local search to the best route every generation (a "memetic" GA).

Speed-ups in every version: routes are lists of city indices, distances come from a
precomputed numpy distance matrix, and selection uses numpy instead of a pandas table.
"""
import random
import numpy as np


def distance_matrix(xy):
    xy = np.asarray(xy, dtype=float)
    return np.sqrt(((xy[:, None, :] - xy[None, :, :]) ** 2).sum(-1))


def route_length(route, Dm):
    r = np.asarray(route)
    return float(Dm[r, np.roll(r, -1)].sum())


def breed(p1, p2):           # ORDERED CROSSOVER, AS IN THE TUTORIAL
    a, b = int(random.random() * len(p1)), int(random.random() * len(p1))
    start, end = min(a, b), max(a, b)
    part = p1[start:end]
    used = set(part)
    return part + [c for c in p2 if c not in used]


def mutate(route, rate, inversion):
    route = list(route)
    n = len(route)
    for i in range(n):
        if random.random() < rate:
            j = int(random.random() * n)
            if inversion:
                a, b = min(i, j), max(i, j)
                route[a:b + 1] = route[a:b + 1][::-1]
            else:
                route[i], route[j] = route[j], route[i]
    return route


def two_opt(route, Dm):
    route = list(route)
    n = len(route)
    improved = True
    while improved:
        improved = False
        for i in range(n - 1):
            for j in range(i + 2, n if i > 0 else n - 1):
                a, b, c, d = route[i], route[i + 1], route[j], route[(j + 1) % n]
                if Dm[a, c] + Dm[b, d] < Dm[a, b] + Dm[c, d] - 1e-12:
                    route[i + 1:j + 1] = route[i + 1:j + 1][::-1]
                    improved = True
    return route


def run_ga(xy, popSize=100, eliteSize=20, mutationRate=0.01, generations=500,
           fix_elitism=True, inversion=False, use_two_opt=False):
    Dm = distance_matrix(xy)
    n = len(xy)
    pop = [random.sample(range(n), n) for _ in range(popSize)]
    progress = []

    for g in range(generations + 1):
        lengths = np.array([route_length(r, Dm) for r in pop])
        order = np.argsort(lengths)
        if use_two_opt and g < generations:
            best = order[0]
            pop[best] = two_opt(pop[best], Dm)
            lengths[best] = route_length(pop[best], Dm)
            order = np.argsort(lengths)
        progress.append(lengths[order[0]])
        if g == generations:
            break

        # SELECTION: ELITES FIRST, THEN ROULETTE WHEEL ON FITNESS = 1 / LENGTH (AS IN THE TUTORIAL)
        fit = 1.0 / lengths[order]
        cum_perc = 100 * np.cumsum(fit) / fit.sum()
        chosen = list(order[:eliteSize])
        for _ in range(popSize - eliteSize):
            chosen.append(order[np.searchsorted(cum_perc, 100 * random.random())])
        matingpool = [pop[i] for i in chosen]

        # CROSSOVER
        shuffled = random.sample(matingpool, len(matingpool))
        children = [list(r) for r in matingpool[:eliteSize]]
        for i in range(popSize - eliteSize):
            children.append(breed(shuffled[i], shuffled[popSize - i - 1]))

        # MUTATION
        if fix_elitism:
            pop = children[:eliteSize] + [mutate(c, mutationRate, inversion)
                                          for c in children[eliteSize:]]
        else:
            pop = [mutate(c, mutationRate, inversion) for c in children]

    best = pop[int(np.argmin([route_length(r, Dm) for r in pop]))]
    return best, progress