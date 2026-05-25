import numpy as np
from scipy.linalg import eigh_tridiagonal
from sklearn.neighbors import KernelDensity

def compute_ground_state_energy(returns, n_grid=200):
    """
    Discretise the SUSY Hamiltonian H = -d²/dx² + V_eff(x)
    and return the smallest eigenvalue (ground state energy).
    """
    returns_clean = returns.dropna().values.reshape(-1, 1)
    if len(returns_clean) < 10:
        return 1.0  # fallback

    # KDE for superpotential W(x) = -log(p(x))
    kde = KernelDensity(kernel='gaussian', bandwidth=0.01).fit(returns_clean)
    grid_min = np.percentile(returns_clean, 1)
    grid_max = np.percentile(returns_clean, 99)
    margin = 0.05 * (grid_max - grid_min)
    grid_min -= margin
    grid_max += margin
    x_grid = np.linspace(grid_min, grid_max, n_grid)
    log_prob = kde.score_samples(x_grid.reshape(-1, 1))
    W = -log_prob
    W -= np.min(W)                     # normalise

    dx = x_grid[1] - x_grid[0]
    Wp = np.gradient(W, dx)
    Wpp = np.gradient(Wp, dx)
    V_eff = Wp**2 - Wpp

    # Kinetic term: -0.5 * d²/dx² (ħ=1, m=1)
    hbar_sq_over_2m = 0.5
    n = n_grid
    diag = np.zeros(n)
    off_diag = np.zeros(n-1)
    for i in range(n):
        diag[i] = V_eff[i] + 2 * hbar_sq_over_2m / dx**2
    for i in range(n-1):
        off_diag[i] = - hbar_sq_over_2m / dx**2

    # Compute smallest eigenvalue only
    eigvals, _ = eigh_tridiagonal(diag, off_diag, select='a', select_range=(0, 0))
    return float(eigvals[0])

def susy_qm_score(returns, n_grid=200, epsilon=None):
    """
    Return the ground state energy of the SUSY Hamiltonian.
    Lower energy = flatter potential = more metastability.
    """
    returns_clean = returns.dropna()
    if len(returns_clean) < 10:
        return 1.0
    return compute_ground_state_energy(returns_clean, n_grid)
