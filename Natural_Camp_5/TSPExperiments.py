"""
Week 5 optional challenge: GA and the Travelling Salesman Problem
  Q1: do we always get the same result for the same cities?  (20 runs of the tutorial code)
  Q2: how can the algorithm be improved?  (tutorial vs three improvements, 20 runs each)
Reference: OR-Tools (guided local search, 30 s) gives a best-known route for comparison.
Saves everything to week5_tsp_results.npz for week5_tsp_figures.py.
"""
import random
import time
import numpy as np
from multiprocessing import Pool
import TSPTutorial as T
import TSPImproved as I

RUNS = 50
N_CITIES = 25
SETTINGS = dict(popSize=100, eliteSize=20, mutationRate=0.01, generations=500)

# THE SAME 25 CITIES FOR EVERY RUN (SAME WAY THE TUTORIAL MAKES THEM)
random.seed(42)
XY = [(int(random.random() * 200), int(random.random() * 200)) for _ in range(N_CITIES)]

VARIANTS = {
    'Tutorial': None,
    'Fixed elitism': dict(fix_elitism=True, inversion=False, use_two_opt=False),
    '+ Inversion mutation': dict(fix_elitism=True, inversion=True, use_two_opt=False),
    '+ 2-opt (memetic)': dict(fix_elitism=True, inversion=True, use_two_opt=True),
}


def job(args):
    name, seed = args
    random.seed(seed)
    t = time.time()
    if VARIANTS[name] is None:
        cities = [T.City(x, y) for x, y in XY]
        best, prog = T.run_ga(cities, **SETTINGS)
        route = [cities.index(c) for c in best]
    else:
        route, prog = I.run_ga(XY, **SETTINGS, **VARIANTS[name])
    return name, seed, time.time() - t, np.array(prog), np.array(route)


def best_known(xy, seconds=30):
    from ortools.constraint_solver import pywrapcp, routing_enums_pb2
    Dm = (I.distance_matrix(xy) * 1000).round().astype(int)
    man = pywrapcp.RoutingIndexManager(len(xy), 1, 0)
    rt = pywrapcp.RoutingModel(man)
    cb = rt.RegisterTransitCallback(lambda i, j: int(Dm[man.IndexToNode(i), man.IndexToNode(j)]))
    rt.SetArcCostEvaluatorOfAllVehicles(cb)
    p = pywrapcp.DefaultRoutingSearchParameters()
    p.local_search_metaheuristic = routing_enums_pb2.LocalSearchMetaheuristic.GUIDED_LOCAL_SEARCH
    p.time_limit.seconds = seconds
    sol = rt.SolveWithParameters(p)
    idx, route = rt.Start(0), []
    while not rt.IsEnd(idx):
        route.append(man.IndexToNode(idx)); idx = sol.Value(rt.NextVar(idx))
    return route, I.route_length(route, I.distance_matrix(xy))


if __name__ == '__main__':
    jobs = [(name, s) for name in VARIANTS for s in range(RUNS)]
    jobs.sort(key=lambda j: j[0] != 'Tutorial')            # SLOWEST JOBS FIRST
    with Pool() as pool:
        results = pool.map(job, jobs, chunksize=1)
    ref_route, ref_len = best_known(XY)

    out = {'xy': np.array(XY), 'ref_route': np.array(ref_route), 'ref_len': ref_len}
    print(f'{N_CITIES} cities, {RUNS} runs per version, settings {SETTINGS}')
    print(f'Best-known route (OR-Tools): {ref_len:.1f}\n')
    print(f"{'Version':<22}{'Mean':>8}{'Median':>8}{'Min':>8}{'Max':>8}{'Std':>7}"
          f"{'Optimal':>9}{'Got worse':>11}{'Time (s)':>10}")
    for name in VARIANTS:
        rs = sorted([r for r in results if r[0] == name], key=lambda r: r[1])
        final = np.array([r[3][-1] for r in rs])
        worse = np.array([np.sum(np.diff(r[3]) > 1e-9) for r in rs])
        secs = np.array([r[2] for r in rs])
        optimal = np.sum(final <= ref_len + 1e-6)
        key = name.replace(' ', '_').replace('+', 'plus').replace('(', '').replace(')', '').replace('-', '')
        out[f'{key}_progress'] = np.array([r[3] for r in rs])
        out[f'{key}_routes'] = np.array([r[4] for r in rs])
        out[f'{key}_time'] = secs
        print(f'{name:<22}{final.mean():>8.1f}{np.median(final):>8.1f}{final.min():>8.1f}'
              f'{final.max():>8.1f}{final.std(ddof=1):>7.1f}{optimal:>6}/{RUNS}'
              f'{worse.mean():>11.1f}{secs.mean():>10.2f}')
        if name == 'Tutorial':
            distinct = len({round(f, 6) for f in final})
            print(f'    distinct final distances: {distinct}/{RUNS}; '
                  f'runs that ended worse than their own best: {np.sum(final > np.array([r[3].min() for r in rs]) + 1e-9)}/{RUNS}')
    np.savez('week5_tsp_results.npz', **out)