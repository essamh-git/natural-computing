"""
Week 7: Stochastic Diffusion Search experiments (1000 runs each, N = 10 agents as in SDS.py)
Q2: how many iterations until all agents are active?
Q3: searching for a model with no perfect match ('fine' instead of 'find')
Q4: two perfect matches of the model in the search space
Saves week7_results.npz for week7_figures.py.
"""
import numpy as np
from SDS import sds_run

RUNS = 1000
SS = 'try to find sds in this sentence'
out = {}


def first(cond):
    idx = np.flatnonzero(cond)
    return idx[0] if len(idx) else -1


# Q2
print("Q2: model 'sds', search space:", repr(SS))
true_pos = SS.index('sds')
first_full, settled, acts = [], [], []
for s in range(RUNS):
    act, hyp = sds_run(SS, 'sds', N=10, maxIter=100, seed=s)
    acts.append(act)
    first_full.append(first(act == 100))
    on_target = np.all(hyp == true_pos, axis=1)
    # SETTLED = FROM THIS ITERATION ON, EVERY AGENT ALWAYS HOLDS THE CORRECT POSITION
    later_wrong = np.flatnonzero(~on_target)
    settled.append(later_wrong[-1] + 1 if len(later_wrong) else 0)
first_full, settled = np.array(first_full) + 1, np.array(settled) + 1   # COUNT ITERATIONS FROM 1
print(f'  first iteration with 100% active: median {np.median(first_full):.0f}, mean {first_full.mean():.1f}, '
      f'90% of runs by {np.percentile(first_full, 90):.0f}, never: {np.sum(first_full < 1)}')
print(f'  all agents settled on "sds" for good: median {np.median(settled):.0f}, mean {settled.mean():.1f}, '
      f'90% of runs by {np.percentile(settled, 90):.0f}')
out.update(q2_activity=np.array(acts), q2_first=first_full, q2_settled=settled)

# Q3
print("\nQ3: model 'fine' (no perfect match; best partial match 'find' = 3 of 4 letters)")
find_pos = SS.index('find')
acts, share = [], []
for s in range(RUNS):
    act, hyp = sds_run(SS, 'fine', N=10, maxIter=200, seed=s)
    acts.append(act)
    share.append(100 * np.mean(hyp == find_pos, axis=1))
acts, share = np.array(acts), np.array(share)
print(f'  activity, iterations 100-200: mean {acts[:, 100:].mean():.1f}%, '
      f'reached 100% at least once in {np.mean((acts == 100).any(axis=1)) * 100:.0f}% of runs')
print(f"  agents pointing at 'find', iterations 100-200: mean {share[:, 100:].mean():.1f}%")
print(f'  (theory: an agent on "find" passes a test with probability 3/4)')
act_sds = out['q2_activity'][:, 50:].mean()
print(f"  compare 'sds' (perfect match), iterations 50-100: mean activity {act_sds:.1f}%")
out.update(q3_activity=acts, q3_share=share)

# Q4
SS2 = 'try to find sds in this sds sentence'
p1, p2 = SS2.index('sds'), SS2.rindex('sds')
print(f"\nQ4: two matches, search space: {SS2!r} (positions {p1} and {p2})")
for N, iters in ((10, 300), (100, 1000)):
    runs = RUNS if N == 10 else 200
    s1, s2, fix, winner = [], [], [], []
    for s in range(runs):
        act, hyp = sds_run(SS2, 'sds', N=N, maxIter=iters, seed=s)
        a, b = 100 * np.mean(hyp == p1, axis=1), 100 * np.mean(hyp == p2, axis=1)
        s1.append(a); s2.append(b)
        t = first((a == 100) | (b == 100))
        fix.append(t)
        winner.append(0 if t < 0 else (1 if a[t] == 100 else 2))
    fix, winner = np.array(fix), np.array(winner)
    done = fix >= 0
    fix = np.where(done, fix + 1, -1)      # COUNT ITERATIONS FROM 1
    print(f'  N = {N}: one match took ALL agents in {done.mean() * 100:.0f}% of runs within {iters} iterations '
          f'(median iteration {np.median(fix[done]) if done.any() else float("nan"):.0f}); '
          f'first match won {np.sum(winner == 1)}, second won {np.sum(winner == 2)}')
    split = ~done
    if split.any():
        a = np.array(s1)[split]
        frozen = [np.flatnonzero(np.diff(x) != 0)[-1] + 2 if np.any(np.diff(x) != 0) else 1 for x in a]
        print(f'         split runs ({split.sum()}): share on match 1 between {a[:, -1].min():.0f}% and '
              f'{a[:, -1].max():.0f}%, frozen by iteration {np.median(frozen):.0f} (median)')
    out.update({f'q4_N{N}_share1': np.array(s1), f'q4_N{N}_share2': np.array(s2),
                f'q4_N{N}_fix': fix, f'q4_N{N}_winner': winner})

np.savez_compressed('week7_results.npz', **out)