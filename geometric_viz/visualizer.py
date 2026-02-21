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
    ) -> str:
        """
        Scree plot with one line per extraction point.

        Left panel : per-component explained variance ratio.
        Right panel: cumulative explained variance with 90% threshold line.
        """
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        colors = cm.tab20(np.linspace(0, 1, max(len(pca_results), 1)))

        for (name, res), color in zip(pca_results.items(), colors):
            ev = res["explained_variance_ratio"][:max_components]
            cum = res["cumulative_variance"][:max_components]
            x = range(1, len(ev) + 1)
            label = name.replace("layer_", "L").replace("_", " ")

            axes[0].plot(x, ev, color=color, alpha=0.75, linewidth=1.2, label=label)
            axes[1].plot(x, cum, color=color, alpha=0.75, linewidth=1.2, label=label)

        axes[0].set_xlabel("Principal Component")
        axes[0].set_ylabel("Explained Variance Ratio")
        axes[0].set_title("Scree Plot")
        axes[0].set_xlim(1, max_components)

        axes[1].set_xlabel("Number of Components")
        axes[1].set_ylabel("Cumulative Explained Variance")
        axes[1].set_title("Cumulative Variance")
        axes[1].axhline(0.9, color="black", linestyle="--", alpha=0.4, label="90%")
        axes[1].set_xlim(1, max_components)
        axes[1].set_ylim(0, 1.05)

        # Shared legend outside the axes
        handles, labels = axes[0].get_legend_handles_labels()
        fig.legend(
            handles,
            labels,
            loc="center right",
            bbox_to_anchor=(1.18, 0.5),
            fontsize=7,
            framealpha=0.9,
        )

        plt.suptitle(f"{model_name}: PCA Analysis", fontsize=11, y=1.01)
        plt.tight_layout()
        path = self.save_dir / f"{_safe_name(model_name)}_pca_scree.png"
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
