import numpy as np
from scipy.linalg import eigh_tridiagonal
from sklearn.neighbors import KernelDensity

def superpotential_from_returns(returns, grid_min=-0.05, grid_max=0.05, n_grid=100):
    """Estimate superpotential W(x) = -log(p(x)) + constant."""
    kde = KernelDensity(kernel='gaussian', bandwidth='scott').fit(returns.values.reshape(-1, 1))
    x_grid = np.linspace(grid_min, grid_max, n_grid)
    log_prob = kde.score_samples(x_grid.reshape(-1, 1))
    W = -log_prob
    W -= np.min(W)
    return x_grid, W

def discretized_hamiltonian(x, W, hbar=1.0, m=1.0):
    """Construct discrete Hamiltonian and return eigenvalues."""
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

def witten_index(eigvals, threshold=0.01):
    """Number of eigenvalues near zero."""
    return np.sum(np.abs(eigvals) < threshold)

def susy_qm_score(returns, n_grid=100, threshold=0.01):
    """Return Python int (or float) Witten index."""
    returns_clean = returns.dropna()
    if len(returns_clean) < 10:
        return 0.0
    ret_min, ret_max = returns_clean.quantile(0.01), returns_clean.quantile(0.99)
    margin = max(0.01, (ret_max - ret_min) * 0.1)
    grid_min = ret_min - margin
    grid_max = ret_max + margin
    x_grid, W = superpotential_from_returns(returns_clean, grid_min, grid_max, n_grid)
    eigvals = discretized_hamiltonian(x_grid, W)
    idx = witten_index(eigvals, threshold)
    # Convert from numpy int64 to Python int
    return int(idx) if isinstance(idx, np.integer) else float(idx)
