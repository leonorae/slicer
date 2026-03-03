"""Mathematical analysis tools for LLM projection geometry.

Phase 0 — Effective Rank Analysis
    effective_rank(S)          → Shannon-entropy erank (Roy & Vetterli 2007)
    spectrum_noise_floor(S)    → number of singular values above noise floor

All functions are pure numpy: no model loading, no disk I/O.
They can be safely imported by any module without side effects.
"""

from __future__ import annotations

import numpy as np


def effective_rank(S: np.ndarray) -> float:
    """Shannon-entropy effective rank (Roy & Vetterli 2007).

    erank = exp(-sum_k p_k * log(p_k))   where p_k = sigma_k / sum(sigma_j)

    Intuitively: if only r singular values dominate, erank ≈ r; if all k
    singular values are equal, erank = k.  Unlike the hard rank, erank is
    continuous and differentiable in the singular values.

    Parameters
    ----------
    S : 1-D array of singular values (non-negative, need not be sorted).
        Values ≤ 1e-10 are treated as zero and excluded before computing
        the normalised probability distribution, preventing log(0).

    Returns
    -------
    Effective rank as a float in [1, len(S)].
    Returns 0.0 if S is empty or all-zero.
    """
    s = np.asarray(S, dtype=np.float64).ravel()
    s = s[s > 1e-10]
    if s.size == 0:
        return 0.0
    p = s / s.sum()
    return float(np.exp(-(p * np.log(p)).sum()))


def spectrum_noise_floor(S: np.ndarray, threshold: float = 1e-6) -> int:
    """Number of singular values strictly above ``threshold * S.max()``.

    This is *n_visible* — the count of directions that carry meaningful
    signal above the numerical noise floor of the Ridge solution.

    Used by Phase 2 null space analysis to partition W's column space into
    a *visible* subspace (directions the projection actively uses) and a
    *null* subspace (directions the LLM activation geometry does not reach).

    Parameters
    ----------
    S         : 1-D array of singular values.
    threshold : Relative cutoff as a fraction of the largest singular value.
                Default 1e-6 matches typical float32 numerical precision.

    Returns
    -------
    Integer count of visible directions.  0 if S is empty.
    """
    s = np.asarray(S, dtype=np.float64).ravel()
    if s.size == 0:
        return 0
    return int(np.sum(s > threshold * s.max()))
