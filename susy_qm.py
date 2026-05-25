import numpy as np
from sklearn.neighbors import KernelDensity
from scipy.signal import argrelextrema

def superpotential_from_returns(returns, grid_min=None, grid_max=None, n_grid=200):
    """
    Estimate superpotential W(x) = -log(p(x)) + constant.
    Returns x_grid and W evaluated on it.
    """
    returns_clean = returns.dropna().values.reshape(-1, 1)
    if len(returns_clean) < 5:
        return np.linspace(-0.05, 0.05, n_grid), np.zeros(n_grid)
    if grid_min is None:
        grid_min = np.percentile(returns_clean, 1)
    if grid_max is None:
        grid_max = np.percentile(returns_clean, 99)
    # Add small margin
    margin = max(0.01, (grid_max - grid_min) * 0.1)
    grid_min -= margin
    grid_max += margin
    x_grid = np.linspace(grid_min, grid_max, n_grid)
    kde = KernelDensity(kernel='gaussian', bandwidth='scott').fit(returns_clean)
    log_prob = kde.score_samples(x_grid.reshape(-1, 1))
    W = -log_prob
    # Normalise to have minimum zero
    W -= np.min(W)
    return x_grid, W

def count_local_minima(x, W):
    """Count number of local minima (valleys) of W(x)."""
    # Find indices where W is smaller than both neighbours
    minima_idx = argrelextrema(W, np.less)[0]
    # Filter out minima that are too shallow (less than 5% of total range)
    if len(W) == 0:
        return 0
    w_range = np.ptp(W)
    threshold = 0.05 * w_range if w_range > 0 else 0.0
    deep_minima = [i for i in minima_idx if W[i] - np.min(W) > threshold]  # local minima that are not just flat
    # Also require that the minimum is not at the very edge (artefact)
    valid = [i for i in deep_minima if 0 < i < len(W)-1]
    return len(valid)

def witten_index_from_superpotential(returns, n_grid=200):
    """
    Compute Witten index as number of local minima of the superpotential.
    """
    x_grid, W = superpotential_from_returns(returns, n_grid=n_grid)
    return count_local_minima(x_grid, W)

def susy_qm_score(returns, n_grid=200, threshold=None):
    """
    Returns the number of metastable regimes (local minima of -log pdf).
    """
    returns_clean = returns.dropna()
    if len(returns_clean) < 10:
        return 0.0
    idx = witten_index_from_superpotential(returns_clean, n_grid=n_grid)
    return int(idx) if isinstance(idx, np.integer) else float(idx)
