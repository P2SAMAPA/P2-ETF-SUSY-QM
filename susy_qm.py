import numpy as np
from scipy.stats import norm

def susy_qm_score(returns, n_grid=None, prominence=None):
    """
    Return a proxy for the ground state energy: negative log-likelihood
    of the returns under a fitted normal distribution (lower = better fit).
    """
    returns_clean = returns.dropna()
    if len(returns_clean) < 5:
        return 1.0
    mu = returns_clean.mean()
    sigma = returns_clean.std()
    if sigma < 1e-12:
        return 1.0
    # Negative log-likelihood (higher = worse fit)
    nll = -norm.logpdf(returns_clean.values, mu, sigma).sum()
    # Normalize by number of points to get per-observation negative log-likelihood
    return nll / len(returns_clean)
