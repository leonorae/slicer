"""
Quantitative geometric metrics for the visualization pipeline.

Three families of metrics:

1. Intrinsic dimensionality   – from PCA explained-variance spectrum
2. Attention–MLP separability – silhouette score and centroid-distance ratio
3. Position dependence        – cross-validated R² of a linear position predictor
"""

from typing import Dict, List, Optional, Tuple

import numpy as np


# ---------------------------------------------------------------------------
# 1. Intrinsic dimensionality
# ---------------------------------------------------------------------------

def compute_intrinsic_dimensionality(
    explained_variance_ratio: np.ndarray,
    threshold: float = 0.90,
) -> dict:
    """
    Compute intrinsic-dimensionality estimates from a PCA explained-variance spectrum.

    Parameters
    ----------
    explained_variance_ratio : 1-D array of per-component variance fractions (sums to ≤1)
    threshold                : cumulative variance fraction defining the "effective rank"

    Returns
    -------
    dict with:
        n_to_90_percent    : int   – components needed to reach ``threshold`` variance
        participation_ratio: float – (Σλ)² / Σλ²  (larger → more spread across dimensions)
        top_5_variance     : float – cumulative variance in the first 5 components
        cumulative_variance: list  – running cumulative variances
    """
    ev = np.asarray(explained_variance_ratio, dtype=float)
    cumsum = np.cumsum(ev)

    # Number of components to reach threshold
    hits = np.where(cumsum >= threshold)[0]
    n_to_threshold = int(hits[0]) + 1 if len(hits) > 0 else len(ev)

    # Participation ratio: (Σλ_i)² / Σλ_i²
    # Uses the explained-variance ratios as proxies for eigenvalues.
    pr = float((ev.sum() ** 2) / (np.sum(ev ** 2) + 1e-12))

    top5 = float(cumsum[min(4, len(cumsum) - 1)])

    return {
        "n_to_90_percent": n_to_threshold,
        "participation_ratio": pr,
        "top_5_variance": top5,
        "cumulative_variance": cumsum.tolist(),
    }


# ---------------------------------------------------------------------------
# 2. Attention–MLP separability
# ---------------------------------------------------------------------------

def compute_separability(
    attn_acts: np.ndarray,
    mlp_acts: np.ndarray,
    max_samples: int = 5_000,
    random_state: int = 42,
) -> dict:
    """
    Compute separability between attention-output and MLP-output point clouds.

    Two complementary measures:
    - **Silhouette score** (−1…1): 1 = perfectly separated, 0 = overlapping, <0 = mixed.
    - **Separability ratio**: centroid distance / mean intra-cluster radius.
      Values >1 indicate distinct clusters; <1 indicates substantial overlap.

    Parameters
    ----------
    attn_acts   : ndarray (n, dim) – attention-output activations
    mlp_acts    : ndarray (n, dim) – MLP-output activations
    max_samples : total sample cap (split evenly between classes)

    Returns
    -------
    dict with silhouette_score, separability_ratio, centroid_distance, avg_intra_distance
    """
    from sklearn.metrics import silhouette_score
    from sklearn.preprocessing import StandardScaler

    rng = np.random.default_rng(random_state)

    n_per = max_samples // 2
    idx_a = rng.choice(len(attn_acts), min(n_per, len(attn_acts)), replace=False)
    idx_m = rng.choice(len(mlp_acts), min(n_per, len(mlp_acts)), replace=False)
    A = attn_acts[idx_a]
    M = mlp_acts[idx_m]

    combined = np.vstack([A, M])
    labels = np.array([0] * len(A) + [1] * len(M))

    # Standardise before silhouette computation
    scaler = StandardScaler()
    combined_s = scaler.fit_transform(combined)

    sil = float(
        silhouette_score(
            combined_s,
            labels,
            sample_size=min(5_000, len(labels)),
            random_state=random_state,
        )
    )

    c_a = A.mean(axis=0)
    c_m = M.mean(axis=0)
    centroid_dist = float(np.linalg.norm(c_a - c_m))

    intra_a = float(np.mean(np.linalg.norm(A - c_a, axis=1)))
    intra_m = float(np.mean(np.linalg.norm(M - c_m, axis=1)))
    avg_intra = (intra_a + intra_m) / 2.0
    sep_ratio = centroid_dist / (avg_intra + 1e-8)

    return {
        "silhouette_score": sil,
        "separability_ratio": sep_ratio,
        "centroid_distance": centroid_dist,
        "avg_intra_distance": avg_intra,
    }


def compute_attention_mlp_separability(
    activations: Dict[str, np.ndarray],
    max_samples: int = 5_000,
) -> dict:
    """
    Aggregate attention and MLP outputs across all layers and compute separability.

    Looks for keys containing 'attn_out' and 'mlp_out' in the activations dict.

    Returns
    -------
    dict from compute_separability, or {'error': reason} if data is missing.
    """
    attn_parts = [arr for k, arr in activations.items() if "attn_out" in k]
    mlp_parts = [arr for k, arr in activations.items() if "mlp_out" in k]

    if not attn_parts or not mlp_parts:
        return {"error": "no 'attn_out' or 'mlp_out' keys found in activations"}

    attn_all = np.vstack(attn_parts)
    mlp_all = np.vstack(mlp_parts)
    return compute_separability(attn_all, mlp_all, max_samples=max_samples)


# ---------------------------------------------------------------------------
# 3. Position dependence
# ---------------------------------------------------------------------------

def compute_position_dependence(
    per_seq_acts: List[np.ndarray],
    per_seq_positions: List[np.ndarray],
    max_tokens: int = 20_000,
    random_state: int = 42,
) -> dict:
    """
    Predict token position from activations via cross-validated Ridge regression.

    For models with learned absolute positional embeddings (GPT-2), the
    residual stream carries strong positional information → high R².
    For RoPE models (LLaMA, Mistral), position is encoded in attention patterns
    rather than residual stream activations → near-zero R².

    Parameters
    ----------
    per_seq_acts      : list of (seq_len, hidden_dim) arrays, one per sequence
    per_seq_positions : list of (seq_len,) integer position arrays  (0 … seq_len−1)
    max_tokens        : cap on total tokens used (random subsample if exceeded)

    Returns
    -------
    dict with position_r2_mean, position_r2_std, n_tokens_used
    """
    from sklearn.decomposition import PCA as _PCA
    from sklearn.linear_model import Ridge
    from sklearn.model_selection import cross_val_score
    from sklearn.preprocessing import StandardScaler

    if not per_seq_acts:
        return {"error": "no per-sequence activations provided"}

    # Use relative position [0, 1] within each sequence instead of absolute
    # indices.  Absolute indices (0…seq_len−1) create incompatible y-ranges
    # across CV folds: a fold whose test set contains long sequences but
    # whose training set has only short ones predicts ŷ≈30 for tokens at
    # y=200, driving R² to large negative values.  Relative position is
    # scale-invariant across sequence lengths.
    X_parts, y_parts = [], []
    for acts, pos in zip(per_seq_acts, per_seq_positions):
        if len(pos) == 0:
            continue
        rel = pos.astype(float) / max(float(len(pos) - 1), 1.0)
        X_parts.append(acts)
        y_parts.append(rel)

    if not X_parts:
        return {"error": "no per-sequence activations provided"}

    X = np.vstack(X_parts)
    y = np.concatenate(y_parts)

    if len(X) > max_tokens:
        rng = np.random.default_rng(random_state)
        idx = rng.choice(len(X), max_tokens, replace=False)
        X, y = X[idx], y[idx]

    n = len(X)

    # Reduce to top-K PCA components so n >> p (ridge is underdetermined
    # when raw 768-dim features outnumber training samples per fold).
    n_components = min(48, n // 10)
    if n_components < 2:
        return {"error": f"too few tokens ({n}) for reliable position R²; need ≥ 20"}

    scaler = StandardScaler()
    X_s = scaler.fit_transform(X)

    pca_dim = _PCA(n_components=n_components, random_state=random_state)
    X_pca = pca_dim.fit_transform(X_s)

    alpha = max(1.0, np.sqrt(n))
    model = Ridge(alpha=alpha)
    scores = cross_val_score(model, X_pca, y, cv=5, scoring="r2", n_jobs=-1)

    return {
        "position_r2_mean": float(np.mean(scores)),
        "position_r2_std": float(np.std(scores)),
        "n_tokens_used": int(n),
        "n_pca_components": int(n_components),
        "ridge_alpha": float(alpha),
    }
