"""
Visualization utilities for the geometric analysis pipeline.

Produces four plot types saved as high-resolution PNG files:
    1. PCA scree plot (per-component and cumulative explained variance)
    2. UMAP scatter grid – one subplot per extraction point, colored by component type
    3. UMAP layer-evolution grid – successive layers of a single component type
    4. UMAP combined overview – all points in one figure, colored by component type
"""

import re
from pathlib import Path
from typing import Dict, Optional

import matplotlib
matplotlib.use("Agg")  # force non-interactive backend before pyplot is imported
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np

# ---------------------------------------------------------------------------
# Component type → color mapping
# ---------------------------------------------------------------------------
COMPONENT_COLORS: Dict[str, str] = {
    "pre_attn_norm": "#1f77b4",      # steel blue
    "attn_out": "#d62728",           # crimson
    "post_attn_residual": "#ff7f0e", # orange
    "pre_mlp_norm": "#e377c2",       # pink
    "mlp_out": "#2ca02c",            # green
    "post_mlp_residual": "#9467bd",  # purple
    "final_norm": "#8c564b",         # brown
}


def _component_type(name: str) -> str:
    """Map an extraction-point key to its component type label."""
    # Try longest match first to avoid 'attn_out' matching inside 'post_attn_residual'
    ordered = [
        "post_attn_residual",
        "post_mlp_residual",
        "pre_attn_norm",
        "attn_out",
        "pre_mlp_norm",
        "mlp_out",
        "final_norm",
    ]
    for comp in ordered:
        if comp in name:
            return comp
    return "other"


def _safe_name(model_name: str) -> str:
    """Convert model name to a filesystem-safe string."""
    return re.sub(r"[^\w\-]", "_", model_name)


class GeometryVisualizer:
    """
    Produces and saves geometric visualization plots.

    Parameters
    ----------
    save_dir : directory for output PNG files (created if absent)
    """

    def __init__(self, save_dir: str = "./outputs"):
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        plt.rcParams.update({"figure.dpi": 100, "font.size": 9})

    # ------------------------------------------------------------------
    # 1. PCA scree plot
    # ------------------------------------------------------------------

    def plot_pca_scree(
        self,
        pca_results: Dict[str, dict],
        model_name: str,
        max_components: int = 30,
        key_filter: Optional[str] = "post_mlp_residual",
    ) -> str:
        """
        Scree plot with one line per extraction point.

        Left panel : per-component explained variance ratio (log scale to show tail).
        Right panel: cumulative explained variance with 90% threshold line.

        Parameters
        ----------
        key_filter : if set, only plot extraction points whose name contains this
                     string. Defaults to 'post_mlp_residual' to avoid 73-line clutter.
                     Pass None to show all points.
        """
        subset = {
            k: v for k, v in pca_results.items()
            if key_filter is None or key_filter in k or k == "final_norm"
        }
        if not subset:
            subset = pca_results  # fallback: show everything

        colors = cm.viridis(np.linspace(0.1, 0.9, max(len(subset), 1)))
        fig, axes = plt.subplots(1, 2, figsize=(13, 5))

        for (name, res), color in zip(subset.items(), colors):
            ev = res["explained_variance_ratio"][:max_components]
            cum = res["cumulative_variance"][:max_components]
            x = list(range(1, len(ev) + 1))
            # Short label: "L06 post mlp residual" → "L06"
            m = re.search(r"layer_(\d+)", name)
            label = f"L{m.group(1)}" if m else name.replace("_", " ")

            axes[0].plot(x, ev, color=color, alpha=0.8, linewidth=1.4, label=label)
            axes[1].plot(x, cum, color=color, alpha=0.8, linewidth=1.4, label=label)

        axes[0].set_xlabel("Principal Component")
        axes[0].set_ylabel("Explained Variance Ratio (log)")
        axes[0].set_title("Scree Plot")
        axes[0].set_yscale("log")
        axes[0].set_xlim(1, max_components)

        axes[1].set_xlabel("Number of Components")
        axes[1].set_ylabel("Cumulative Explained Variance")
        axes[1].set_title("Cumulative Variance")
        axes[1].axhline(0.9, color="black", linestyle="--", alpha=0.5, label="90%")
        axes[1].set_xlim(1, max_components)
        axes[1].set_ylim(0, 1.05)

        handles, labels = axes[0].get_legend_handles_labels()
        fig.legend(
            handles, labels,
            loc="center right", bbox_to_anchor=(1.12, 0.5), fontsize=8,
        )

        filter_note = f" ({key_filter} layers)" if key_filter else ""
        plt.suptitle(f"{model_name}: PCA Scree{filter_note}", fontsize=11, y=1.01)
        plt.tight_layout()
        path = self.save_dir / f"{_safe_name(model_name)}_pca_scree.png"
        plt.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        return str(path)

    def plot_pca_scatter(
        self,
        pca_results: Dict[str, dict],
        model_name: str,
        key_filter: str = "post_mlp_residual",
        max_layers: int = 12,
        max_points: int = 2_000,
    ) -> str:
        """
        PC1 vs PC2 scatter grids showing how the residual-stream representation
        changes across layers. Useful when UMAP is skipped.
        """
        subset = {
            k: v for k, v in pca_results.items() if key_filter in k
        }
        if not subset:
            return ""

        # Thin to at most max_layers evenly spaced layers
        keys = sorted(subset.keys(), key=lambda x: int(re.search(r"layer_(\d+)", x).group(1)))
        step = max(1, len(keys) // max_layers)
        keys = keys[::step][:max_layers]

        n = len(keys)
        cols = min(4, n)
        rows = (n + cols - 1) // cols
        rng = np.random.default_rng(seed=0)

        fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 3.5 * rows))
        axes_flat = np.array(axes).flatten() if n > 1 else [axes]
        cmap = cm.viridis

        for i, key in enumerate(keys):
            arr = pca_results[key]["transformed"]  # (n_tokens, n_components)
            ax = axes_flat[i]

            if len(arr) > max_points:
                idx = rng.choice(len(arr), max_points, replace=False)
                arr = arr[idx]

            color = cmap(i / max(n - 1, 1))
            ax.scatter(arr[:, 0], arr[:, 1], c=[color], s=3, alpha=0.4,
                       linewidths=0, rasterized=True)

            # Clip axes to 2nd–98th percentile so the outlier dimension
            # (one massive direction in GPT-2's residual stream) doesn't
            # compress the main cloud into a single dot.
            for axis_data, set_lim in (
                (arr[:, 0], ax.set_xlim),
                (arr[:, 1], ax.set_ylim),
            ):
                lo, hi = np.percentile(axis_data, [2, 98])
                margin = max((hi - lo) * 0.1, abs(hi) * 0.05, 1e-3)
                set_lim(lo - margin, hi + margin)

            m = re.search(r"layer_(\d+)", key)
            ax.set_title(f"Layer {m.group(1)}" if m else key, fontsize=9)
            ax.set_xlabel("PC1", fontsize=7)
            ax.set_ylabel("PC2", fontsize=7)
            ax.tick_params(labelsize=6)

        for j in range(n, len(axes_flat)):
            axes_flat[j].set_visible(False)

        plt.suptitle(f"{model_name}: PC1 vs PC2 — {key_filter.replace('_', ' ')} layers",
                     fontsize=11)
        plt.tight_layout()
        path = self.save_dir / f"{_safe_name(model_name)}_pca_scatter.png"
        plt.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        return str(path)

    def plot_positional_embedding_helix(
        self,
        wpe_weight: np.ndarray,
        model_name: str,
        n_positions: int = 256,
    ) -> str:
        """
        Project learned positional embeddings into 3D PCA and plot the helix.

        This reproduces the key finding from Ning et al. (2025): GPT-2's learned
        positional embeddings (wpe.weight, shape [1024, 768]) form a helix when
        projected to 3 principal components, reflecting the periodic structure
        the model uses to encode sequence order.

        Requires no forward passes — only the embedding weight matrix.
        Not applicable to RoPE models (LLaMA, Mistral, etc.).

        Parameters
        ----------
        wpe_weight  : ndarray (n_ctx, hidden_dim) — positional embedding weight matrix
        n_positions : how many positions to plot (default 256; 1024 is noisier)
        """
        from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 — registers '3d' projection
        from sklearn.decomposition import PCA as _PCA

        wpe = wpe_weight[:min(n_positions, wpe_weight.shape[0])]
        n_pos = len(wpe)

        pca3 = _PCA(n_components=3, random_state=42)
        coords = pca3.fit_transform(wpe)   # (n_pos, 3)
        ev = pca3.explained_variance_ratio_

        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection="3d")

        # Draw the trajectory as a colored line (position → color)
        colors = cm.plasma(np.linspace(0, 1, n_pos))
        for i in range(n_pos - 1):
            ax.plot(
                coords[i : i + 2, 0], coords[i : i + 2, 1], coords[i : i + 2, 2],
                color=colors[i], linewidth=0.8, alpha=0.7,
            )

        sc = ax.scatter(
            coords[:, 0], coords[:, 1], coords[:, 2],
            c=np.arange(n_pos), cmap="plasma", s=12, zorder=5, linewidths=0,
        )
        plt.colorbar(sc, ax=ax, label="Position index", shrink=0.65, pad=0.1)

        ax.set_xlabel(f"PC1 ({ev[0]:.1%})", fontsize=9)
        ax.set_ylabel(f"PC2 ({ev[1]:.1%})", fontsize=9)
        ax.set_zlabel(f"PC3 ({ev[2]:.1%})", fontsize=9)
        ax.set_title(
            f"{model_name}: Positional Embedding Helix\n"
            f"3D PCA of wpe.weight (first {n_pos} positions)",
            fontsize=10,
        )
        ax.tick_params(labelsize=7)

        path = self.save_dir / f"{_safe_name(model_name)}_positional_helix.png"
        plt.tight_layout()
        plt.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        return str(path)

    # ------------------------------------------------------------------
    # 2. UMAP grid by extraction point
    # ------------------------------------------------------------------

    def plot_umap_by_component(
        self,
        umap_results: Dict[str, np.ndarray],
        model_name: str,
        max_points: int = 3_000,
    ) -> str:
        """
        Grid of UMAP scatter plots, one subplot per extraction point.
        Color encodes the component type (attention, MLP, residual, …).
        """
        names = list(umap_results.keys())
        n = len(names)
        if n == 0:
            return ""
        cols = min(4, n)
        rows = (n + cols - 1) // cols

        fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 3.5 * rows))
        axes_flat = np.array(axes).flatten() if n > 1 else [axes]

        rng = np.random.default_rng(seed=0)

        for i, name in enumerate(names):
            arr = umap_results[name]
            ax = axes_flat[i]

            if len(arr) > max_points:
                idx = rng.choice(len(arr), max_points, replace=False)
                arr = arr[idx]

            comp = _component_type(name)
            color = COMPONENT_COLORS.get(comp, "#7f7f7f")

            ax.scatter(
                arr[:, 0], arr[:, 1],
                c=color, s=2, alpha=0.4, linewidths=0, rasterized=True,
            )
            short = name.replace("layer_", "L").replace("_", "\n")
            ax.set_title(short, fontsize=7, pad=2)
            ax.set_xticks([])
            ax.set_yticks([])

        # Hide unused subplots
        for j in range(n, len(axes_flat)):
            axes_flat[j].set_visible(False)

        # Component-type legend
        legend_handles = [
            plt.scatter([], [], c=c, s=25, label=comp.replace("_", " "))
            for comp, c in COMPONENT_COLORS.items()
        ]
        fig.legend(
            handles=legend_handles,
            loc="lower center",
            ncol=len(COMPONENT_COLORS),
            bbox_to_anchor=(0.5, -0.03),
            fontsize=8,
        )

        plt.suptitle(f"{model_name}: UMAP by Extraction Point", fontsize=11)
        plt.tight_layout()
        path = self.save_dir / f"{_safe_name(model_name)}_umap_components.png"
        plt.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        return str(path)

    # ------------------------------------------------------------------
    # 3. UMAP layer-evolution grid
    # ------------------------------------------------------------------

    def plot_umap_layer_evolution(
        self,
        umap_results: Dict[str, np.ndarray],
        model_name: str,
        component: str = "post_mlp_residual",
        max_points: int = 3_000,
    ) -> str:
        """
        Grid showing how a single component type's UMAP projection evolves
        through successive layers.  Color cycles from blue (early) to yellow (late).
        """
        # Collect layers matching the requested component type
        layer_data: Dict[int, np.ndarray] = {}
        for name, arr in umap_results.items():
            if component in name:
                m = re.search(r"layer_(\d+)", name)
                if m:
                    layer_data[int(m.group(1))] = arr

        if not layer_data:
            return ""

        layers = sorted(layer_data.keys())
        n = len(layers)
        cols = min(4, n)
        rows = (n + cols - 1) // cols
        cmap = cm.viridis

        fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 3.5 * rows))
        axes_flat = np.array(axes).flatten() if n > 1 else [axes]

        rng = np.random.default_rng(seed=0)

        for i, layer in enumerate(layers):
            arr = layer_data[layer]
            ax = axes_flat[i]

            if len(arr) > max_points:
                idx = rng.choice(len(arr), max_points, replace=False)
                arr = arr[idx]

            frac = i / max(n - 1, 1)
            color = cmap(frac)
            ax.scatter(
                arr[:, 0], arr[:, 1],
                c=[color], s=2, alpha=0.4, linewidths=0, rasterized=True,
            )
            ax.set_title(f"Layer {layer}", fontsize=9)
            ax.set_xticks([])
            ax.set_yticks([])

        for j in range(n, len(axes_flat)):
            axes_flat[j].set_visible(False)

        comp_label = component.replace("_", " ")
        plt.suptitle(
            f"{model_name}: Layer Evolution ({comp_label})", fontsize=11
        )
        plt.tight_layout()
        path = self.save_dir / f"{_safe_name(model_name)}_umap_layer_evolution.png"
        plt.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        return str(path)

    # ------------------------------------------------------------------
    # 4. UMAP combined overview
    # ------------------------------------------------------------------

    def plot_umap_combined(
        self,
        umap_results: Dict[str, np.ndarray],
        model_name: str,
        max_points_per_type: int = 1_000,
    ) -> str:
        """
        Single figure combining all extraction points, colored by component type.
        Useful for a quick high-level overview of cluster structure.
        """
        fig, ax = plt.subplots(figsize=(9, 7))
        rng = np.random.default_rng(seed=0)
        seen_comps = set()

        for name, arr in umap_results.items():
            comp = _component_type(name)
            color = COMPONENT_COLORS.get(comp, "#7f7f7f")

            if len(arr) > max_points_per_type:
                idx = rng.choice(len(arr), max_points_per_type, replace=False)
                arr = arr[idx]

            label = comp.replace("_", " ") if comp not in seen_comps else None
            seen_comps.add(comp)

            ax.scatter(
                arr[:, 0], arr[:, 1],
                c=color, s=1, alpha=0.3,
                linewidths=0, rasterized=True, label=label,
            )

        ax.legend(markerscale=6, fontsize=9, framealpha=0.9)
        ax.set_xlabel("UMAP 1")
        ax.set_ylabel("UMAP 2")
        ax.set_title(f"{model_name}: UMAP Combined Overview")

        plt.tight_layout()
        path = self.save_dir / f"{_safe_name(model_name)}_umap_combined.png"
        plt.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        return str(path)
