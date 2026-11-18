"""
Stochastic Diffusion Search (SDS) - the lecturer's code (github.com/mohmaj/SDS, SDS.py),
unchanged inside sds_run(), except that it returns the activity and the agents' hypotheses
at every iteration instead of printing them, and takes the search space, model, population
size, number of iterations and random seed as arguments.
"""
import numpy as np


def sds_run(ss, model, N=10, maxIter=30, seed=None):
    if seed is not None:
        np.random.seed(seed)
    # INITIALISE AGENTS
    hypo = np.random.randint(len(ss) - len(model) + 1, size=(N))
    status = np.zeros((N), dtype=bool)
    activity, hypos = [], []

    for itr in range(maxIter):
        activeAgents = 0
        # TEST PHASE
        for i in range(N):
            microFeature = np.random.randint(len(model))
            if ss[hypo[i] + microFeature] == model[microFeature]:
                status[i] = True
                activeAgents += 1
            else:
                status[i] = False

        # DIFFUSION PHASE
        for i in range(N):
            if status[i] == False:                # INACTIVE AGENT
                rand = np.random.randint(N)       # PICK RANDOM AGENT TO COMMUNICATE
                if status[rand] == True:          # SHARE HYPOTHESIS
                    hypo[i] = hypo[rand]
                else:          # PICK A RANDOM HYPOTHESIS
                    hypo[i] = np.random.randint(len(ss) - len(model) + 1)
            else:              # ACTIVE AGENT
                microFeature = np.random.randint(len(model))
                if ss[hypo[i] + microFeature] == model[microFeature]:
                    status[i] = True
                else:
                    status[i] = False

        activity.append(activeAgents * 100 / N)
        hypos.append(hypo.copy())
    return np.array(activity), np.array(hypos)