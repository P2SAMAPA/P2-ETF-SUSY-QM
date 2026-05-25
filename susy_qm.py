import numpy as np
from scipy.stats import norm

def susy_qm_score(returns, n_grid=None, prominence=None):
    """
    Compute a proxy for ground state energy: negative log‑likelihood
    of returns under a fitted normal distribution.
    Lower values indicate better fit (more Gaussian, less structure).
    Higher values indicate deviation from normality – more "exotic" regimes.
    """
    returns_clean = returns.dropna()
    if len(returns_clean) < 5:
        return 1.0
    mu = returns_clean.mean()
    sigma = returns_clean.std()
    if sigma < 1e-12:
        return 1.0
    # Negative log-likelihood (higher = less likely = more "energy")
    nll = -norm.logpdf(returns_clean.values, mu, sigma).sum()
    # Return per-observation average
    return nll / len(returns_clean)
