import numpy as np
from scipy.linalg import eigh_tridiagonal
from sklearn.neighbors import KernelDensity

def superpotential_from_returns(returns, n_grid=200):
    returns_clean = returns.dropna().values.reshape(-1, 1)
    if len(returns_clean) < 5:
        return np.linspace(-0.1, 0.1, n_grid), np.zeros(n_grid)
    grid_min = np.percentile(returns_clean, 1)
    grid_max = np.percentile(returns_clean, 99)
    margin = max(0.01, (grid_max - grid_min) * 0.2)
    grid_min -= margin
    grid_max += margin
    x_grid = np.linspace(grid_min, grid_max, n_grid)
    kde = KernelDensity(kernel='gaussian', bandwidth=0.01).fit(returns_clean)
    log_prob = kde.score_samples(x_grid.reshape(-1, 1))
    W = -log_prob
    W -= np.min(W)
    return x_grid, W

def discretized_hamiltonian_eigenvalues(x, W, hbar=1.0, m=1.0):
    dx = x[1] - x[0]
    Wp = np.gradient(W, dx)
    Wpp = np.gradient(Wp, dx)
    V_eff = Wp**2 - Wpp
    hbar_sq_over_2m = (hbar**2) / (2*m)
    n = len(x)
    diag = np.zeros(n)
    off_diag = np.zeros(n-1)
    for i in range(n):
        diag[i] = V_eff[i] + 2 * hbar_sq_over_2m / dx**2
    for i in range(n-1):
        off_diag[i] = - hbar_sq_over_2m / dx**2
    eigvals, _ = eigh_tridiagonal(diag, off_diag, select='a', select_range=None)
    return eigvals

def witten_index(eigvals, threshold=1e-6):
    # Count eigenvalues below threshold (near zero)
    return np.sum(eigvals < threshold)

def susy_qm_score(returns, n_grid=200, threshold=1e-6):
    returns_clean = returns.dropna()
    if len(returns_clean) < 10:
        return 0.0
    x_grid, W = superpotential_from_returns(returns_clean, n_grid=n_grid)
    eigvals = discretized_hamiltonian_eigenvalues(x_grid, W)
    idx = witten_index(eigvals, threshold)
    # convert to Python int
    return float(idx)
