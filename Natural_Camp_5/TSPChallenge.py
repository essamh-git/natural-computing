"""
Week 5 TSP challenge: 8 UK cities, private-jet distances in km (from the lecture slide).
1) Brute force: check every possible round trip, so the answer is guaranteed to be the shortest.
2) GA: run the GA from the TSP tutorial (improved version) 30 times to see if it finds the same answer.
The challenge asks to visit each city once; a TSP route also returns to the start, so both
the round-trip and the one-way (no return) answers are given.
"""
import itertools
import random
import numpy as np

CITIES = ['Brighton', 'Bristol', 'Cambridge', 'Glasgow', 'Liverpool', 'London', 'Manchester', 'Oxford']
UPPER = [[0, 172, 145, 607, 329, 72, 312, 120],
         [0, 0, 192, 494, 209, 158, 216, 92],
         [0, 0, 0, 490, 237, 75, 205, 100],
         [0, 0, 0, 0, 286, 545, 296, 489],
         [0, 0, 0, 0, 0, 421, 49, 208],
         [0, 0, 0, 0, 0, 0, 249, 75],
         [0, 0, 0, 0, 0, 0, 0, 194],
         [0, 0, 0, 0, 0, 0, 0, 0]]
D = np.array(UPPER)
D = D + D.T        # THE SLIDE ONLY GIVES THE UPPER TRIANGLE


def tour_length(route, closed=True):
    legs = list(zip(route, route[1:])) + ([(route[-1], route[0])] if closed else [])
    return int(sum(D[a, b] for a, b in legs))


def brute_force(closed=True):
    n = len(CITIES)
    if closed:          # FIX THE START CITY: ROTATIONS ARE THE SAME TOUR
        tours = [(0,) + p for p in itertools.permutations(range(1, n))]
    else:
        tours = list(itertools.permutations(range(n)))
    lengths = np.array([tour_length(t, closed) for t in tours])
    best = int(lengths.min())
    return best, [t for t, L in zip(tours, lengths) if L == best], lengths, len(tours)


def ga(seed, popSize=50, eliteSize=10, mutationRate=0.02, generations=200):
    """Improved GA from the TSP tutorial page (fixed elitism, inversion mutation), using D."""
    random.seed(seed)
    n = len(CITIES)
    pop = [random.sample(range(n), n) for _ in range(popSize)]
    for g in range(generations):
        lengths = np.array([tour_length(r) for r in pop])
        order = np.argsort(lengths)
        fit = 1.0 / lengths[order]
        cum = 100 * np.cumsum(fit) / fit.sum()
        pool = [pop[i] for i in order[:eliteSize]] + \
               [pop[order[np.searchsorted(cum, 100 * random.random())]] for _ in range(popSize - eliteSize)]
        shuffled = random.sample(pool, len(pool))
        children = [list(r) for r in pool[:eliteSize]]
        for i in range(popSize - eliteSize):
            a, b = sorted((int(random.random() * n), int(random.random() * n)))
            part = shuffled[i][a:b]
            child = part + [c for c in shuffled[popSize - i - 1] if c not in part]
            for k in range(n):         # INVERSION MUTATION
                if random.random() < mutationRate:
                    j = int(random.random() * n)
                    lo, hi = min(k, j), max(k, j)
                    child[lo:hi + 1] = child[lo:hi + 1][::-1]
            children.append(child)
        pop = children
    best = min(pop, key=tour_length)
    return best, tour_length(best)


def name_route(route, closed=True):
    # START THE PRINTED ROUTE AT BRIGHTON (CITY 0) FOR READABILITY
    r = list(route)
    if closed:
        k = r.index(0); r = r[k:] + r[:k]
        if r[1] > r[-1]:        # SAME LOOP EITHER DIRECTION: PICK ONE CONSISTENTLY
            r = [r[0]] + r[1:][::-1]
        r = r + [r[0]]
    return ' → '.join(CITIES[c] for c in r)


if __name__ == '__main__':
    best, tours, lengths, count = brute_force(closed=True)
    print(f'Round trip: checked {count} orders ({count // 2} different tours, each in 2 directions)')
    print(f'  shortest = {best} km, found {len(tours)} times (the same loop in both directions)')
    print('  route:', name_route(tours[0]))
    print('  as indices:', list(tours[0]) + [tours[0][0]])
    print(f'  longest = {lengths.max()} km, average = {lengths.mean():.0f} km')

    best_o, tours_o, _, count_o = brute_force(closed=False)
    print(f'\nOne-way (no return): checked {count_o} orders, shortest = {best_o} km')
    print('  route:', name_route(tours_o[0], closed=False))

    results = [ga(s) for s in range(30)]
    found = sum(L == best for _, L in results)
    print(f'\nGA: found the {best} km tour in {found}/30 runs '
          f'(lengths found: {sorted(set(L for _, L in results))})')
    np.save('all_tour_lengths.npy', lengths)