# Supersymmetric Quantum Mechanics Engine for ETFs

Applies supersymmetric quantum mechanics to ETF return distributions. The per‑ETF **Witten index** (number of near‑zero energy states) measures the richness of metastable regimes – a novel complexity signal.

## Features
- Three ETF universes
- Seven rolling windows (63–4536 days)
- Kernel density estimation of return distribution → superpotential
- Discrete Hamiltonian (SUSY partner) eigenvalue problem
- Witten index = count of eigenvalues below threshold
- Best window automatically selected (largest absolute raw signal)
- Two‑tab Streamlit dashboard (auto best + manual window selection)
- Results stored on Hugging Face: `P2SAMAPA/p2-etf-susy-qm-results`

## Usage

1. Set `HF_TOKEN` environment variable.
2. Run training: `python train.py`
3. Launch dashboard: `streamlit run streamlit_app.py`
4. GitHub Actions runs daily.

## Interpretation

- The **Witten index** counts the number of ground states of a supersymmetric system.
- In finance, each near‑zero energy state corresponds to a metastable regime in the return dynamics.
- ETFs with high index are more complex (multiple coexisting regimes) and may offer higher predictability.
- This is the first known application of supersymmetric quantum mechanics to quantitative finance.

## Requirements

See `requirements.txt`.
