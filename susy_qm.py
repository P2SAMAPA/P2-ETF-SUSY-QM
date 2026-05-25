import numpy as np
from sklearn.neighbors import KernelDensity
from scipy.signal import find_peaks

def count_density_modes(returns, n_grid=200, prominence=0.05):
    """
    Estimate number of modes (peaks) in the return density.
    Each mode corresponds to a metastable regime.
    """
    returns_clean = returns.dropna().values.reshape(-1, 1)
    if len(returns_clean) < 10:
        return 0
    # Grid based on data range
    grid_min = np.percentile(returns_clean, 1)
    grid_max = np.percentile(returns_clean, 99)
    margin = max(0.01, (grid_max - grid_min) * 0.1)
    grid_min -= margin
    grid_max += margin
    x_grid = np.linspace(grid_min, grid_max, n_grid)
    kde = KernelDensity(kernel='gaussian', bandwidth='scott').fit(returns_clean)
    log_density = kde.score_samples(x_grid.reshape(-1, 1))
    density = np.exp(log_density)
    # Find peaks with minimum prominence
    peaks, _ = find_peaks(density, prominence=prominence * np.max(density))
    return len(peaks)

def susy_qm_score(returns, n_grid=200, prominence=0.05):
    """Return number of density modes (metastable regimes)."""
    returns_clean = returns.dropna()
    if len(returns_clean) < 10:
        return 0.0
    n_modes = count_density_modes(returns_clean, n_grid=n_grid, prominence=prominence)
    return float(n_modes)
