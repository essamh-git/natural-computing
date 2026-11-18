"""
Week 7 optional: identifying two instances with context-free and context-sensitive SDS.
Search space with two perfect matches of 'sds'. 200 runs per mode, 100 agents, 1000 iterations.
Saves Week7Optional.npz for Week7OptionalFigures.py.
"""
import numpy as np
from SDSModes import sds_run

SS2 = 'try to find sds in this sds sentence'
p1, p2 = SS2.index('sds'), SS2.rindex('sds')
RUNS, N, ITERS = 200, 100, 1000
MODES = ['standard', 'context_free', 'context_sensitive']
out = {}
print(f'{SS2!r}: matches at {p1} and {p2}; {RUNS} runs, {N} agents, {ITERS} iterations\n')
print(f"{'Mode':<19}{'Active':>8}{'Match 1':>9}{'Match 2':>9}{'Elsewhere':>11}{'Both kept':>11}"
      f"{'Balance':>9}{'Moves':>8}")
for mode in MODES:
    A, S1, S2 = [], [], []
    for s in range(RUNS):
        act, hyp = sds_run(SS2, 'sds', N=N, maxIter=ITERS, seed=s, mode=mode)
        A.append(act)
        S1.append(100 * np.mean(hyp == p1, axis=1))
        S2.append(100 * np.mean(hyp == p2, axis=1))
    A, S1, S2 = map(np.array, (A, S1, S2))
    late = slice(ITERS // 2, None)                      # SECOND HALF OF EACH RUN
    both = np.mean((S1[:, -1] >= 10) & (S2[:, -1] >= 10)) * 100
    # BALANCE: SHARE OF THE SMALLER CLUSTER AMONG AGENTS ON EITHER MATCH (50% = PERFECTLY EQUAL)
    tot = S1[:, -1] + S2[:, -1]
    bal = np.mean(np.minimum(S1[:, -1], S2[:, -1]) / np.where(tot > 0, tot, 1)) * 100
    moves = np.mean(np.abs(np.diff(S1[:, late], axis=1)).sum(axis=1))   # HOW MUCH CLUSTER 1 CHANGES
    print(f'{mode:<19}{A[:, late].mean():>7.1f}%{S1[:, late].mean():>8.1f}%{S2[:, late].mean():>8.1f}%'
          f'{100 - S1[:, late].mean() - S2[:, late].mean():>10.1f}%{both:>10.0f}%{bal:>8.0f}%{moves:>8.0f}')
    out.update({f'{mode}_activity': A, f'{mode}_s1': S1, f'{mode}_s2': S2})
print('\nActive = % of agents active; Match 1/2 = % of agents on each match; Both kept = runs where each')
print('match still has at least 10% of agents at the end; Balance = smaller cluster as a share of the two')
print('(50% = equal); Moves = total change in cluster 1 over the second half (0 = frozen).')

# SINGLE MATCH: HOW MUCH OF THE POPULATION EACH MODE PUTS ON IT
SS1 = 'try to find sds in this sentence'
print(f"\nOne match ({SS1!r}), 100 runs: average % of agents on 'sds' in the second half")
for mode in MODES:
    sh = [100 * np.mean(sds_run(SS1, 'sds', N=N, maxIter=ITERS, seed=s, mode=mode)[1][ITERS // 2:]
                        == SS1.index('sds')) for s in range(100)]
    print(f'  {mode:<19}{np.mean(sh):.1f}%')
    out[f'{mode}_single'] = np.array(sh)
np.savez_compressed('week7_optional.npz', **out)