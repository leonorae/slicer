"""
PCA and UMAP dimensionality reduction analyzers.

PCAAnalyzer fits a separate PCA per extraction point, producing per-point
explained-variance curves for scree plots and intrinsic-dimensionality estimates.

UMAPAnalyzer fits a single UMAP on a combined sample from all extraction points
(ensuring a common embedding space), then transforms each point individually.
"""

from typing import Dict, Optional

import numpy as np
from sklearn.decomposition import PCA


class PCAAnalyzer:
    """
    Per-extraction-point PCA.

    Fitting a separate PCA for each extraction point lets us compare the
    intrinsic dimensionality and spectral profile across layer components
    independently.

    Parameters
    ----------
    n_components : int
        Number of principal components to retain (capped by min(n_samples, n_features)).
    """

    def __init__(self, n_components: int = 50):
        self.n_components = n_components

    def fit_transform(self, activations: Dict[str, np.ndarray]) -> Dict[str, dict]:
        """
        Fit PCA independently on each extraction point.

        Parameters
        ----------
        activations : dict mapping name -> ndarray (n_tokens, hidden_dim)

        Returns
        -------
        results : dict mapping name -> dict with keys:
            'transformed'             ndarray (n_tokens, n_components)
            'explained_variance_ratio' ndarray (n_components,)
            'cumulative_variance'      ndarray (n_components,)
            'components'               ndarray (n_components, hidden_dim)
        """
        results: Dict[str, dict] = {}
        for name, arr in activations.items():
            n_comp = min(self.n_components, arr.shape[0] - 1, arr.shape[1])
            if n_comp < 1:
                continue
            pca = PCA(n_components=n_comp, random_state=42)
            transformed = pca.fit_transform(arr)
            results[name] = {
                "transformed": transformed,
                "explained_variance_ratio": pca.explained_variance_ratio_,
                "cumulative_variance": np.cumsum(pca.explained_variance_ratio_),
                "components": pca.components_,
            }
        return results


class UMAPAnalyzer:
    """
    UMAP reduction fitted on a joint sample drawn from all extraction points.

    Fitting on a combined sample ensures all extraction points share the same
    embedding coordinate system, making cross-point comparison meaningful.

    Parameters
    ----------
    n_components : int        Number of UMAP output dimensions (default 2 for scatter plots).
    n_neighbors  : int        Controls local vs. global structure trade-off.
    min_dist     : float      Controls cluster tightness.
    metric       : str        Distance metric used internally by UMAP.
    """

    def __init__(
        self,
        n_components: int = 2,
        n_neighbors: int = 15,
        min_dist: float = 0.1,
        metric: str = "euclidean",
    ):
        self.n_components = n_components
        self.n_neighbors = n_neighbors
        self.min_dist = min_dist
        self.metric = metric
        self.reducer = None

    def fit_transform(
        self,
        activations: Dict[str, np.ndarray],
        sample_size: int = 5_000,
    ) -> Dict[str, np.ndarray]:
        """
        Fit UMAP on a joint subsample, then transform each extraction point.

        Parameters
        ----------
        activations  : dict mapping name -> ndarray (n_tokens, hidden_dim)
        sample_size  : tokens to sample *per extraction point* when building
                       the fitting dataset (total fitting set = sample_size × n_points,
                       capped by data availability).

        Returns
        -------
        results : dict mapping name -> ndarray (n_tokens, n_components)
        """
        try:
            import umap as umap_lib  # umap-learn
        except ImportError:
            raise ImportError(
                "umap-learn is required. Install with: pip install umap-learn"
            )

        rng = np.random.default_rng(seed=42)

        # Build fitting dataset: sample from each extraction point
        samples = []
        for arr in activations.values():
            n = min(sample_size, len(arr))
            idx = rng.choice(len(arr), n, replace=False)
            samples.append(arr[idx])
        fit_data = np.vstack(samples)

        self.reducer = umap_lib.UMAP(
            n_components=self.n_components,
            n_neighbors=self.n_neighbors,
            min_dist=self.min_dist,
            metric=self.metric,
            random_state=42,
            low_memory=True,
            verbose=False,
        )
        self.reducer.fit(fit_data)

        # Transform each extraction point individually
        results: Dict[str, np.ndarray] = {}
        for name, arr in activations.items():
            results[name] = self.reducer.transform(arr)

        return results
