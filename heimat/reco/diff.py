from scipy.stats import spearmanr, kendalltau
import numpy as np

def sim_psk(xs1, xs2):
    """ gegeben zwei Series ergibt die Koeffizienten für:
        - Pearson
        - Kentall
        - Spearman
    """
    p = np.corrcoef(xs1, xs2)
    s = spearmanr(xs1, xs2)
    k = kendalltau(xs1, xs2)
    return p, s, k

def dist_em(xs1, xs2):
    euklid = np.linalg.norm(xs1 - xs2)
    manhattan = sum(abs(e - s) for s,e in zip(xs1, xs2))
    return euklid, manhattan