import numpy as np
from scipy.linalg import eigh_tridiagonal
from sklearn.neighbors import KernelDensity

def compute_hamiltonian_eigenvalues(returns, n_grid=200):
    """Compute eigenvalues of the SUSY Hamiltonian H = -d²/dx² + V_eff(x)."""
    returns_clean = returns.dropna().values.reshape(-1, 1)
    if len(returns_clean) < 10:
        return np.array([1.0])  # dummy
    # Estimate superpotential
    kde = KernelDensity(kernel='gaussian', bandwidth=0.01).fit(returns_clean)
    grid_min = np.percentile(returns_clean, 1)
    grid_max = np.percentile(returns_clean, 99)
    margin = 0.05
    grid_min -= margin
    grid_max += margin
    x_grid = np.linspace(grid_min, grid_max, n_grid)
    log_prob = kde.score_samples(x_grid.reshape(-1, 1))
    W = -log_prob
    W -= np.min(W)
    dx = x_grid[1] - x_grid[0]
    Wp = np.gradient(W, dx)
    Wpp = np.gradient(Wp, dx)
    V_eff = Wp**2 - Wpp
    # Kinetic term
    hbar_sq_over_2m = 0.5
    n = n_grid
    diag = np.zeros(n)
    off_diag = np.zeros(n-1)
    for i in range(n):
        diag[i] = V_eff[i] + 2 * hbar_sq_over_2m / dx**2
    for i in range(n-1):
        off_diag[i] = - hbar_sq_over_2m / dx**2
    eigvals, _ = eigh_tridiagonal(diag, off_diag, select='a', select_range=(0, 9))
    return eigvals

def witten_index(returns, n_grid=200, epsilon=0.05):
    """
    Approximate Witten index as the number of eigenvalues below epsilon * max_eigenvalue.
    """
    eigvals = compute_hamiltonian_eigenvalues(returns, n_grid)
    if len(eigvals) == 0:
        return 0
    max_eig = max(eigvals)
    if max_eig < 1e-12:
        return 0
    threshold = epsilon * max_eig
    index = np.sum(eigvals < threshold)
    return int(index)

def susy_qm_score(returns, n_grid=200, epsilon=0.05):
    """Return integer Witten index (0,1,2,...)."""
    returns_clean = returns.dropna()
    if len(returns_clean) < 10:
        return 0
    return witten_index(returns_clean, n_grid, epsilon)
