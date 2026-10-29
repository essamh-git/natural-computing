"""
Week 5 Challenge II: the card game (Harvey, 2011)
Divide cards 1-10 into two piles: the SUM of pile 0 should be close to 36,
and the PRODUCT of pile 1 should be close to 360.

Genotype: 10 bits, one per card. Bit = 0 -> sum pile, bit = 1 -> product pile.
Search space: 2^10 = 1024 possible genotypes.
Fitness (error, minimised): |sum - 36| / 36 + |product - 360| / 360, so 0 = perfect.
(An empty product pile has product 1.)

Two GA modes, both with tournament selection of size 2, uniform crossover and bit-flip
mutation (rate 1/10), and the same budget of fitness evaluations:
  Generational GA:   the whole population is replaced each generation (best one kept).
  Steady-state GA:   Microbial GA (Harvey, 2011): each iteration two random individuals
                     compete; the loser copies some genes from the winner, is mutated,
                     and replaces itself. One new evaluation per iteration.
Random search is included as a baseline.
"""
import math
import random
import numpy as np

CARDS = list(range(1, 11))
L = 10
POP = 20
MUT = 1 / L
PC = 0.7


def error(g):
    s = sum(c for c, b in zip(CARDS, g) if b == 0)
    p = math.prod(c for c, b in zip(CARDS, g) if b == 1)
    return abs(s - 36) / 36 + abs(p - 360) / 360


def mutate(g):
    return [1 - b if random.random() < MUT else b for b in g]


def uniform_crossover(a, b):
    return [x if random.random() < 0.5 else y for x, y in zip(a, b)]


class Budget:
    """Counts fitness evaluations and records the best error after each one."""
    def __init__(self, n):
        self.n, self.best, self.history = n, float('inf'), []

    def eval(self, g):
        e = error(g)
        self.best = min(self.best, e)
        self.history.append(self.best)
        return e

    def left(self):
        return len(self.history) < self.n


def generational_ga(seed, budget=400):
    random.seed(seed)
    B = Budget(budget)
    pop = [[random.randint(0, 1) for _ in range(L)] for _ in range(POP)]
    fit = [B.eval(g) for g in pop]
    while B.left():
        b = int(np.argmin(fit))
        new, new_fit = [pop[b]], [fit[b]]        # ELITISM: KEEP THE BEST
        while len(new) < POP and B.left():
            p1 = min(random.sample(range(POP), 2), key=lambda i: fit[i])   # TOURNAMENTS OF 2
            p2 = min(random.sample(range(POP), 2), key=lambda i: fit[i])
            child = uniform_crossover(pop[p1], pop[p2]) if random.random() < PC else list(pop[p1])
            child = mutate(child)
            new.append(child); new_fit.append(B.eval(child))
        pop, fit = new, new_fit       # WHOLE POPULATION REPLACED
    return np.array(B.history)


def steady_state_ga(seed, budget=400):
    """Microbial GA (Harvey, 2011)."""
    random.seed(seed)
    B = Budget(budget)
    pop = [[random.randint(0, 1) for _ in range(L)] for _ in range(POP)]
    fit = [B.eval(g) for g in pop]
    while B.left():
        a, b = random.sample(range(POP), 2)
        win, lose = (a, b) if fit[a] <= fit[b] else (b, a)
        child = [w if random.random() < 0.5 else l for w, l in zip(pop[win], pop[lose])]
        pop[lose] = mutate(child)
        fit[lose] = B.eval(pop[lose])
    return np.array(B.history)


def random_search(seed, budget=400):
    random.seed(seed)
    B = Budget(budget)
    while B.left():
        B.eval([random.randint(0, 1) for _ in range(L)])
    return np.array(B.history)


def decode(g):
    return ([c for c, b in zip(CARDS, g) if b == 0], [c for c, b in zip(CARDS, g) if b == 1])