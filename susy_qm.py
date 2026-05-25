import numpy as np
from scipy.linalg import eigh_tridiagonal
from sklearn.neighbors import KernelDensity

def superpotential_from_returns(returns, grid_min=-0.05, grid_max=0.05, n_grid=100):
    """
    Estimate superpotential W(x) = -log(p(x)) + constant,
    where p(x) is the kernel density estimate of returns.
    Returns W evaluated on the grid.
    """
    kde = KernelDensity(kernel='gaussian', bandwidth='scott').fit(returns.values.reshape(-1, 1))
    x_grid = np.linspace(grid_min, grid_max, n_grid)
    log_prob = kde.score_samples(x_grid.reshape(-1, 1))
    # W(x) = -log(p(x)) (up to constant)
    W = -log_prob
    # Normalise to zero minimum
    W -= np.min(W)
    return x_grid, W

def discretized_hamiltonian(x, W, hbar=1.0, m=1.0):
    """
    Construct discrete Hamiltonian H = - (ħ²/2m) d²/dx² + V_eff(x)
    where V_eff = W'² - W''   (SUSY partner potential)
    Compute derivatives via finite differences.
    """
    dx = x[1] - x[0]
    # first derivative
    Wp = np.gradient(W, dx)
    # second derivative
    Wpp = np.gradient(Wp, dx)
    V_eff = Wp**2 - Wpp
    # Kinetic term: - (ħ²/2m) * second derivative matrix (tridiagonal)
    hbar_sq_over_2m = (hbar**2) / (2*m)
    n = len(x)
    # main diagonal
    diag = np.zeros(n)
    off_diag = np.zeros(n-1)
    for i in range(n):
        diag[i] = V_eff[i] + 2 * hbar_sq_over_2m / dx**2
    for i in range(n-1):
        off_diag[i] = - hbar_sq_over_2m / dx**2
    # Solve for eigenvalues
    eigvals, _ = eigh_tridiagonal(diag, off_diag, select='a', select_range=None)
    return eigvals

def witten_index(eigvals, threshold=0.01):
    """Number of eigenvalues near zero."""
    return np.sum(np.abs(eigvals) < threshold)

def susy_qm_score(returns, n_grid=100, threshold=0.01):
    """
    Compute per-ETF Witten index from its return series.
    """
    returns_clean = returns.dropna()
    if len(returns_clean) < 10:
        return 0.0
    # Estimate grid limits from data
    ret_min, ret_max = returns_clean.quantile(0.01), returns_clean.quantile(0.99)
    margin = max(0.01, (ret_max - ret_min) * 0.1)
    grid_min = ret_min - margin
    grid_max = ret_max + margin
    x_grid, W = superpotential_from_returns(returns_clean, grid_min, grid_max, n_grid)
    eigvals = discretized_hamiltonian(x_grid, W)
    index = witten_index(eigvals, threshold)
    return index
