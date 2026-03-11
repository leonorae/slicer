"""
Linear projection from LLM hidden-state space to CLIP text-embedding space.

The projection is a simple affine map  y = X @ W.T + b  trained via Ridge
regression (L2-regularised least squares).  This deliberately minimal bridge
preserves the *linear* geometry of the LLM's representations: directions,
distances, and interpolations in activation space map faithfully to
directions in CLIP space.  Any non-linear warping in the resulting images
is therefore attributable to the diffusion decoder, not the bridge.

Typical dimensions (default: SD 1.5 / CLIP ViT-L/14):
    GPT-2        768  →  CLIP ViT-L/14   768   (SD 1.5)
    Llama-2-7B  4096  →  CLIP ViT-L/14   768   (SD 1.5)
    Mistral-7B  4096  →  CLIP ViT-L/14   768   (SD 1.5)
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional, Union

import numpy as np


# Alpha grid used by RidgeCV when alpha="auto" for any projection type.
_ALPHA_GRID: List[float] = [0.01, 0.1, 1.0, 10.0, 100.0, 1000.0]


def _auto_alpha(X_train: np.ndarray, y_train: np.ndarray) -> float:
    """
    Select the best Ridge alpha for (X_train, y_train) via RidgeCV LOO-CV.

    X_train is z-scored internally so the selected alpha is comparable across
    layers with different activation magnitudes.
    """
    from sklearn.linear_model import RidgeCV

    mean = X_train.mean(axis=0)
    scale = X_train.std(axis=0)
    scale[scale < 1e-8] = 1.0
    X_norm = (X_train - mean) / scale

    rcv = RidgeCV(alphas=_ALPHA_GRID, scoring="r2")
    rcv.fit(X_norm, y_train)
    return float(rcv.alpha_)


class LinearProjection:
    """
    Affine map  LLM activation → CLIP embedding  trained with Ridge regression.

    Parameters
    ----------
    alpha : Ridge regularisation strength.  Higher → smoother, more generalised
            map; lower → closer fit to training pairs but risk of overfitting
            and mode collapse in generated images.

    After fitting, the SVD of W is computed automatically:
        W = U S Vt   (thin SVD, full_matrices=False)
    The singular value spectrum S is stored as S.npy alongside W.npy, and the
    right singular vectors Vt are stored as Vt.npy (needed for Phase 2 null
    space analysis).  Properties erank, condition_number, and n_visible expose
    the key scalars computed from S.
    """

    def __init__(self, alpha: float = 1.0):
        self.alpha = alpha
        self.W: Optional[np.ndarray] = None       # (clip_dim, llm_dim)
        self.b: Optional[np.ndarray] = None       # (clip_dim,)
        self.scaler_mean: Optional[np.ndarray] = None   # (llm_dim,)
        self.scaler_scale: Optional[np.ndarray] = None  # (llm_dim,)
        self._meta: Dict = {}
        # SVD of W — populated by fit() and load()
        self._S: Optional[np.ndarray] = None    # (k,) singular values, k=min(clip,llm)
        self._Vt: Optional[np.ndarray] = None   # (k, llm_dim) right singular vectors
        # Corpus statistics for confounder analysis — set externally after fit()
        self.corpus_clip_centroid: Optional[np.ndarray] = None  # (clip_dim,)

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

        # SVD of W for Phase 0 (erank) and Phase 2 (null space).
        # Computed in float64 for numerical accuracy, stored in float32.
        _U, _S, _Vt = np.linalg.svd(self.W.astype(np.float64), full_matrices=False)
        self._S = _S.astype(np.float32)    # shape (min(clip_dim, llm_dim),)
        self._Vt = _Vt.astype(np.float32)  # shape (min(...), llm_dim)

        from .analysis import effective_rank, spectrum_noise_floor
        _erank = effective_rank(self._S)
        _n_vis = spectrum_noise_floor(self._S)
        _cond = (
            float(self._S[0] / self._S[-1])
            if float(self._S[-1]) > 1e-10
            else float("inf")
        )

        self._meta = {
            "n_train": int(X.shape[0]),
            "llm_dim": int(X.shape[1]),
            "clip_dim": int(y.shape[1]),
            "alpha": float(self.alpha),
            "normalize_llm": normalize_llm,
            "normalize_clip": normalize_clip,
            # Phase 0 geometric properties
            "erank": _erank,
            "condition_number": _cond,
            "n_visible": _n_vis,
        }

        return self

    # ------------------------------------------------------------------
    # SVD / geometric properties  (Phase 0)
    # ------------------------------------------------------------------

    @property
    def singular_values(self) -> Optional[np.ndarray]:
        """Singular values of W in descending order, shape (k,).  None if not fitted."""
        return self._S

    @property
    def erank(self) -> Optional[float]:
        """Shannon-entropy effective rank of W.  None if not fitted."""
        return self._meta.get("erank") if self._meta else None

    @property
    def condition_number(self) -> Optional[float]:
        """Condition number of W (sigma_max / sigma_min).  None if not fitted."""
        return self._meta.get("condition_number") if self._meta else None

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

    def corpus_distance(self, probe_activation: np.ndarray) -> float:
        """
        Normalised L2 distance of *probe_activation* from the corpus centroid
        in activation space.

        Uses the z-score parameters (scaler_mean, scaler_scale) that were fitted
        on the training corpus as the corpus statistics.  Equivalent to a
        spherical Mahalanobis distance (diagonal covariance approximation).

        A value of 1.0 means the probe sits one pooled standard deviation away
        from the corpus mean.  Values > 3 indicate out-of-distribution probes.
        """
        if self.scaler_mean is None or self.scaler_scale is None:
            raise RuntimeError("projection not trained — call .fit() first")
        x = probe_activation.astype(np.float64)
        z = (x - self.scaler_mean) / self.scaler_scale
        return float(np.linalg.norm(z))

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
        """Save projection weights, SVD spectra, and metadata to *directory*."""
        d = Path(directory)
        d.mkdir(parents=True, exist_ok=True)

        np.save(d / "W.npy", self.W)
        np.save(d / "b.npy", self.b)
        np.save(d / "scaler_mean.npy", self.scaler_mean)
        np.save(d / "scaler_scale.npy", self.scaler_scale)

        # SVD arrays for Phase 0 (erank) and Phase 2 (null space)
        if self._S is not None:
            np.save(d / "S.npy", self._S)
        if self._Vt is not None:
            np.save(d / "Vt.npy", self._Vt)

        # Corpus statistics for confounder analysis
        if self.corpus_clip_centroid is not None:
            np.save(d / "corpus_clip_centroid.npy", self.corpus_clip_centroid)

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

        # Load SVD arrays if present (absent in projections saved before Phase 0)
        s_path, vt_path = d / "S.npy", d / "Vt.npy"
        if s_path.exists():
            proj._S = np.load(s_path)
        if vt_path.exists():
            proj._Vt = np.load(vt_path)

        meta_path = d / "meta.json"
        if meta_path.exists():
            with open(meta_path) as f:
                proj._meta = json.load(f)
            proj.alpha = proj._meta.get("alpha", 1.0)

        centroid_path = d / "corpus_clip_centroid.npy"
        if centroid_path.exists():
            proj.corpus_clip_centroid = np.load(centroid_path)

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


class LayerProjectionSet:
    """
    One LinearProjection per LLM layer, all trained against the same CLIP targets.

    Training separate Ridge maps per layer lets each layer's statistical
    distribution be captured independently.  Early layers and late layers can
    have very different activation magnitudes and directions; a single shared
    map would be dominated by the layer it was trained on (usually the last).

    Parameters
    ----------
    alpha : Ridge regularisation for every layer (float), or ``"auto"`` to pick
            the best value per layer via RidgeCV leave-one-out cross-validation
            over a log-spaced grid.
    """

    def __init__(self, alpha: Union[float, str] = 1.0):
        self.alpha = alpha
        self.projections: Dict[int, LinearProjection] = {}

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def fit(
        self,
        llm_activations_per_layer: Dict[int, np.ndarray],
        clip_embeddings: np.ndarray,
        *,
        val_fraction: float = 0.15,
        normalize_llm: bool = True,
        normalize_clip: bool = False,
    ) -> Dict[int, Dict]:
        """
        Train one LinearProjection per layer using the same CLIP targets.

        Parameters
        ----------
        llm_activations_per_layer : {layer_idx: (n, llm_dim)}
        clip_embeddings           : (n, clip_dim) — same for every layer
        val_fraction              : fraction of data held out for per-layer metrics

        Returns
        -------
        {layer_idx: validation_metrics_dict}  (includes 'alpha' key per layer)
        """
        n = len(clip_embeddings)
        n_val = max(1, int(n * val_fraction))
        rng = np.random.default_rng(42)
        idx = rng.permutation(n)
        train_idx, val_idx = idx[n_val:], idx[:n_val]

        clip_train = clip_embeddings[train_idx]
        clip_val = clip_embeddings[val_idx]

        # Corpus centroid in CLIP space — used for confounder analysis at generate time
        clip_centroid = clip_train.mean(axis=0).astype(np.float32)

        val_metrics: Dict[int, Dict] = {}

        for layer_idx in sorted(llm_activations_per_layer.keys()):
            X = llm_activations_per_layer[layer_idx]
            X_train, X_val = X[train_idx], X[val_idx]

            if self.alpha == "auto":
                chosen_alpha = self._select_alpha(X_train, clip_train)
            else:
                chosen_alpha = float(self.alpha)

            proj = LinearProjection(alpha=chosen_alpha)
            proj.fit(
                X_train, clip_train,
                normalize_llm=normalize_llm,
                normalize_clip=normalize_clip,
            )
            proj.corpus_clip_centroid = clip_centroid
            self.projections[layer_idx] = proj

            metrics = proj.validate(X_val, clip_val)
            metrics["alpha"] = chosen_alpha
            val_metrics[layer_idx] = metrics

            print(
                f"[ProjectionSet]   layer {layer_idx:3d}  alpha={chosen_alpha:.3g}"
                f"  R²={metrics['projection_r2']:+.4f}",
                flush=True,
            )

        return val_metrics

    def _select_alpha(
        self, X_train: np.ndarray, y_train: np.ndarray
    ) -> float:
        """Pick the best alpha via RidgeCV (leave-one-out) on z-scored X."""
        return _auto_alpha(X_train, y_train)

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def transform(self, layer_idx: int, llm_activations: np.ndarray) -> np.ndarray:
        """Project activations at *layer_idx* into CLIP space."""
        if layer_idx not in self.projections:
            raise KeyError(
                f"no projection trained for layer {layer_idx}. "
                f"Available: {self.layers()}"
            )
        return self.projections[layer_idx].transform(llm_activations)

    def layers(self) -> List[int]:
        return sorted(self.projections.keys())

    def __contains__(self, layer_idx: int) -> bool:
        return layer_idx in self.projections

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, directory: str) -> str:
        """Save all per-layer projections plus an index.json manifest."""
        d = Path(directory)
        d.mkdir(parents=True, exist_ok=True)

        for layer_idx, proj in self.projections.items():
            proj.save(str(d / f"layer_{layer_idx:04d}"))

        with open(d / "index.json", "w") as f:
            json.dump(
                {
                    "type": "LayerProjectionSet",
                    "alpha": self.alpha,
                    "layers": self.layers(),
                },
                f,
                indent=2,
            )
        return str(d)

    @classmethod
    def load(cls, directory: str) -> "LayerProjectionSet":
        """Load a previously saved LayerProjectionSet."""
        d = Path(directory)
        with open(d / "index.json") as f:
            meta = json.load(f)
        lps = cls(alpha=meta.get("alpha", 1.0))
        for layer_idx in meta["layers"]:
            lps.projections[layer_idx] = LinearProjection.load(
                str(d / f"layer_{layer_idx:04d}")
            )
        return lps

    @staticmethod
    def is_saved_at(directory: str) -> bool:
        """Return True if *directory* holds a LayerProjectionSet (has index.json)."""
        return (Path(directory) / "index.json").exists()

    # ------------------------------------------------------------------
    # Repr
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        ls = self.layers()
        if not ls:
            return "LayerProjectionSet(untrained)"
        return (
            f"LayerProjectionSet(n_layers={len(ls)}, "
            f"layers={ls[0]}..{ls[-1]}, alpha={self.alpha!r})"
        )
