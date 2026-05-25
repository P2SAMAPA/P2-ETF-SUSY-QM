import numpy as np
from sklearn.neighbors import KernelDensity
from scipy.signal import find_peaks

def count_density_modes(returns, n_grid=200, prominence=0.01):
    """
    Estimate number of modes (peaks) in the return density.
    Uses a more sensitive prominence and adjusts bandwidth.
    """
    returns_clean = returns.dropna().values.reshape(-1, 1)
    if len(returns_clean) < 10:
        return 0
    # Use a smaller bandwidth than 'scott' for more detail
    # Scott's bandwidth = n^{-1/(d+4)} * std, often too smooth for bimodality detection
    # Use a fixed bandwidth of 0.01 or adaptive based on IQR
    iqr = np.percentile(returns_clean, 75) - np.percentile(returns_clean, 25)
    bw = max(0.005, iqr * 0.5)  # bandwidth as half of IQR, minimum 0.005
    kde = KernelDensity(kernel='gaussian', bandwidth=bw).fit(returns_clean)
    grid_min = np.percentile(returns_clean, 1)
    grid_max = np.percentile(returns_clean, 99)
    margin = max(0.01, (grid_max - grid_min) * 0.1)
    grid_min -= margin
    grid_max += margin
    x_grid = np.linspace(grid_min, grid_max, n_grid)
    log_density = kde.score_samples(x_grid.reshape(-1, 1))
    density = np.exp(log_density)
    # Find peaks with low prominence to catch small modes
    peaks, properties = find_peaks(density, prominence=prominence * np.max(density), width=1)
    # Also consider peaks that are at least 10% of max height
    if len(peaks) == 0:
        # Try lower prominence
        peaks, _ = find_peaks(density, prominence=0.005 * np.max(density))
    return len(peaks)

def susy_qm_score(returns, n_grid=200, prominence=0.01):
    """Return number of density modes (metastable regimes)."""
    returns_clean = returns.dropna()
    if len(returns_clean) < 10:
        return 0.0
    n_modes = count_density_modes(returns_clean, n_grid=n_grid, prominence=prominence)
    return float(n_modes)
