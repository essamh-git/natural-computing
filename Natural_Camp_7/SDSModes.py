"""
SDS with the three ways an ACTIVE agent can behave in the diffusion phase:
  'standard'           the lecturer's SDS.py: the active agent re-tests its hypothesis
  'context_free'       the active agent polls a random agent; if that agent is ACTIVE,
                       it becomes inactive and picks a new random hypothesis
  'context_sensitive'  as context free, but only if the polled agent is active AND has the
                       SAME hypothesis (al-Rifaie and Bishop, 2013; Nasuto, 1999)
The test phase and the inactive agents' behaviour are exactly as in the lecturer's code.
"""
import numpy as np


def sds_run(ss, model, N=10, maxIter=30, seed=None, mode='standard'):
    if seed is not None:
        np.random.seed(seed)
    space = len(ss) - len(model) + 1
    hypo = np.random.randint(space, size=(N))
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
            if status[i] == False:                       # INACTIVE AGENT (UNCHANGED)
                rand = np.random.randint(N)
                if status[rand] == True:
                    hypo[i] = hypo[rand]
                else:
                    hypo[i] = np.random.randint(space)
            elif mode == 'standard':                     # ACTIVE AGENT, LECTURER'S CODE
                microFeature = np.random.randint(len(model))
                status[i] = ss[hypo[i] + microFeature] == model[microFeature]
            else:                                        # ACTIVE AGENT, CONTEXT MODES
                rand = np.random.randint(N)
                same = hypo[rand] == hypo[i]
                if status[rand] and (mode == 'context_free' or same):
                    status[i] = False
                    hypo[i] = np.random.randint(space)

        activity.append(activeAgents * 100 / N)
        hypos.append(hypo.copy())
    return np.array(activity), np.array(hypos)