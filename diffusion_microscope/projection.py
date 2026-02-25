"""
Linear projection from LLM hidden-state space to CLIP text-embedding space.

The projection is a simple affine map  y = X @ W.T + b  trained via Ridge
regression (L2-regularised least squares).  This deliberately minimal bridge
preserves the *linear* geometry of the LLM's representations: directions,
distances, and interpolations in activation space map faithfully to
directions in CLIP space.  Any non-linear warping in the resulting images
is therefore attributable to the diffusion decoder, not the bridge.

Typical dimensions:
    GPT-2        768  →  CLIP ViT-L/14   768   (SD 1.5)
    Llama-2-7B  4096  →  CLIP ViT-H/14  1024   (SD 2.1)
    Mistral-7B  4096  →  CLIP ViT-H/14  1024   (SD 2.1)
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Optional

import numpy as np


class LinearProjection:
    """
    Affine map  LLM activation → CLIP embedding  trained with Ridge regression.

    Parameters
    ----------
    alpha : Ridge regularisation strength.  Higher → smoother, more generalised
            map; lower → closer fit to training pairs but risk of overfitting
            and mode collapse in generated images.
    """

    def __init__(self, alpha: float = 1.0):
        self.alpha = alpha
        self.W: Optional[np.ndarray] = None       # (clip_dim, llm_dim)
        self.b: Optional[np.ndarray] = None       # (clip_dim,)
        self.scaler_mean: Optional[np.ndarray] = None   # (llm_dim,)
        self.scaler_scale: Optional[np.ndarray] = None  # (llm_dim,)
        self._meta: Dict = {}

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def fit(
        self,
        llm_activations: np.ndarray,
        clip_embeddings: np.ndarray,
        *,
        normalize_llm: bool = True,
        normalize_clip: bool = False,
    ) -> "LinearProjection":
        """
        Train the linear map from paired (LLM activation, CLIP embedding) data.

        Parameters
        ----------
        llm_activations : (n_samples, llm_dim)
        clip_embeddings : (n_samples, clip_dim)
        normalize_llm   : z-score LLM activations per dimension before fitting
        normalize_clip   : L2-normalise CLIP embeddings (usually already normed)
        """
        from sklearn.linear_model import Ridge

        X = llm_activations.astype(np.float64)
        y = clip_embeddings.astype(np.float64)

        if X.shape[0] != y.shape[0]:
            raise ValueError(
                f"sample count mismatch: {X.shape[0]} LLM vs {y.shape[0]} CLIP"
            )

        # Optional z-score of LLM activations
        if normalize_llm:
            self.scaler_mean = X.mean(axis=0)
            self.scaler_scale = X.std(axis=0)
            self.scaler_scale[self.scaler_scale < 1e-8] = 1.0  # avoid div-by-zero
            X = (X - self.scaler_mean) / self.scaler_scale
        else:
            self.scaler_mean = np.zeros(X.shape[1])
            self.scaler_scale = np.ones(X.shape[1])

        # Optional L2-normalisation of CLIP targets
        if normalize_clip:
            norms = np.linalg.norm(y, axis=1, keepdims=True)
            norms[norms < 1e-8] = 1.0
            y = y / norms

        reg = Ridge(alpha=self.alpha)
        reg.fit(X, y)

        self.W = reg.coef_.astype(np.float32)      # (clip_dim, llm_dim)
        self.b = reg.intercept_.astype(np.float32)  # (clip_dim,)

        self._meta = {
            "n_train": int(X.shape[0]),
            "llm_dim": int(X.shape[1]),
            "clip_dim": int(y.shape[1]),
            "alpha": float(self.alpha),
            "normalize_llm": normalize_llm,
            "normalize_clip": normalize_clip,
        }

        return self

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def transform(self, llm_activations: np.ndarray) -> np.ndarray:
        """
        Project LLM activations into CLIP space.

        Parameters
        ----------
        llm_activations : (n, llm_dim)  or  (llm_dim,) for a single vector

        Returns
        -------
        (n, clip_dim) or (clip_dim,)  projected embeddings
        """
        if self.W is None:
            raise RuntimeError("projection not trained — call .fit() first")

        squeeze = llm_activations.ndim == 1
        X = llm_activations[np.newaxis] if squeeze else llm_activations
        X = X.astype(np.float64)

        # Apply same normalisation as training
        X = (X - self.scaler_mean) / self.scaler_scale

        out = (X @ self.W.T + self.b).astype(np.float32)
        return out.squeeze(0) if squeeze else out

    # ------------------------------------------------------------------
    # Validation helpers
    # ------------------------------------------------------------------

    def validate(
        self,
        llm_val: np.ndarray,
        clip_val: np.ndarray,
        k_neighbors: int = 5,
    ) -> Dict:
        """
        Quick sanity checks on a held-out set.

        Returns dict with:
            cosine_dist_mean/std  – pairwise cosine distance among projected
                                    vectors (near 0 → collapse)
            projection_r2         – R² of projected vs true CLIP embeddings
            nn_recall_at_k        – fraction of projected vectors whose true
                                    CLIP match is within the k nearest neighbours
        """
        from sklearn.metrics.pairwise import cosine_distances

        proj = self.transform(llm_val)

        # 1. Collapse check — pairwise cosine distance
        dists = cosine_distances(proj)
        np.fill_diagonal(dists, np.nan)
        mean_dist = float(np.nanmean(dists))
        std_dist = float(np.nanstd(dists))

        # 2. R² of reconstruction
        ss_res = np.sum((clip_val - proj) ** 2)
        ss_tot = np.sum((clip_val - clip_val.mean(axis=0)) ** 2)
        r2 = float(1 - ss_res / ss_tot) if ss_tot > 0 else 0.0

        # 3. Nearest-neighbour recall
        #    For each projected vector, check if its ground-truth CLIP match
        #    is among the k closest vectors in the projected set.
        n = len(proj)
        cos_sim = 1.0 - cosine_distances(proj, clip_val)  # (n, n)
        hits = 0
        for i in range(n):
            top_k = np.argsort(cos_sim[i])[-k_neighbors:]
            if i in top_k:
                hits += 1
        nn_recall = float(hits / n) if n > 0 else 0.0

        return {
            "cosine_dist_mean": mean_dist,
            "cosine_dist_std": std_dist,
            "projection_r2": r2,
            f"nn_recall_at_{k_neighbors}": nn_recall,
            "n_val_samples": n,
        }

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, directory: str) -> str:
        """Save projection weights and metadata to *directory*."""
        d = Path(directory)
        d.mkdir(parents=True, exist_ok=True)

        np.save(d / "W.npy", self.W)
        np.save(d / "b.npy", self.b)
        np.save(d / "scaler_mean.npy", self.scaler_mean)
        np.save(d / "scaler_scale.npy", self.scaler_scale)

        with open(d / "meta.json", "w") as f:
            json.dump(self._meta, f, indent=2)

        return str(d)

    @classmethod
    def load(cls, directory: str) -> "LinearProjection":
        """Load a previously saved projection."""
        d = Path(directory)
        proj = cls()
        proj.W = np.load(d / "W.npy")
        proj.b = np.load(d / "b.npy")
        proj.scaler_mean = np.load(d / "scaler_mean.npy")
        proj.scaler_scale = np.load(d / "scaler_scale.npy")

        meta_path = d / "meta.json"
        if meta_path.exists():
            with open(meta_path) as f:
                proj._meta = json.load(f)
            proj.alpha = proj._meta.get("alpha", 1.0)

        return proj

    # ------------------------------------------------------------------
    # Repr
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        if self.W is None:
            return "LinearProjection(untrained)"
        return (
            f"LinearProjection(llm_dim={self.W.shape[1]}, "
            f"clip_dim={self.W.shape[0]}, alpha={self.alpha}, "
            f"n_train={self._meta.get('n_train', '?')})"
        )
