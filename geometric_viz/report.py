"""
Report generation: JSON metrics file and human-readable text summary.
"""

import json
import re
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Tuple

import numpy as np


def _to_json_safe(obj: Any) -> Any:
    """Recursively convert numpy types to Python-native types for JSON serialization."""
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, dict):
        return {k: _to_json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_json_safe(v) for v in obj]
    return obj


class GeometryReport:
    """
    Accumulates analysis results and writes a JSON metrics file and a plain-text summary.

    Parameters
    ----------
    model_name : HuggingFace model identifier (used in filenames and headers)
    output_dir : directory in which to write report files
    """

    def __init__(self, model_name: str, output_dir: str):
        self.model_name = model_name
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.data: Dict[str, Any] = {
            "model_name": model_name,
            "timestamp": datetime.now().isoformat(),
            "metrics": {},
            "plot_paths": {},
        }

    def add_metric(self, key: str, value: Any) -> None:
        """Store a metric under ``key``."""
        self.data["metrics"][key] = value

    def add_plot(self, key: str, path: str) -> None:
        """Record the path to a saved plot."""
        if path:  # skip empty paths (e.g. when a plot was skipped)
            self.data["plot_paths"][key] = path

    def save(self) -> Tuple[str, str]:
        """
        Write JSON and text report files to ``output_dir``.

        Returns
        -------
        (json_path, txt_path) as strings
        """
        safe = re.sub(r"[^\w\-]", "_", self.model_name)

        # --- JSON -----------------------------------------------------------
        json_path = self.output_dir / f"{safe}_metrics.json"
        with open(json_path, "w") as f:
            json.dump(_to_json_safe(self.data), f, indent=2)

        # --- Text summary ---------------------------------------------------
        txt_path = self.output_dir / f"{safe}_summary.txt"
        with open(txt_path, "w") as f:
            f.write("Geometric Visualization Pipeline Report\n")
            f.write(f"Model     : {self.model_name}\n")
            f.write(f"Timestamp : {self.data['timestamp']}\n")
            f.write("=" * 60 + "\n\n")

            m = self.data.get("metrics", {})

            # Intrinsic dimensionality
            if "intrinsic_dimensionality" in m:
                idim = m["intrinsic_dimensionality"]
                f.write("Intrinsic Dimensionality\n")
                f.write("-" * 30 + "\n")
                f.write(
                    f"  Components to 90% variance : "
                    f"{idim.get('n_to_90_percent', 'N/A')}\n"
                )
                pr = idim.get("participation_ratio")
                if pr is not None:
                    f.write(f"  Participation ratio         : {pr:.4f}\n")
                tv = idim.get("top_5_variance")
                if tv is not None:
                    f.write(f"  Cumulative var (first 5 PCs): {tv:.4f}\n")
                f.write("\n")

            # Attention–MLP separability
            if "attention_mlp_separability" in m:
                sep = m["attention_mlp_separability"]
                f.write("Attention–MLP Separability\n")
                f.write("-" * 30 + "\n")
                if "error" in sep:
                    f.write(f"  Error: {sep['error']}\n")
                else:
                    sil = sep.get("silhouette_score")
                    rat = sep.get("separability_ratio")
                    cen = sep.get("centroid_distance")
                    if sil is not None:
                        f.write(f"  Silhouette score   : {sil:.4f}\n")
                        f.write(
                            "    (range −1…1; >0.5 = well-separated, ~0 = overlapping)\n"
                        )
                    if rat is not None:
                        f.write(f"  Separability ratio : {rat:.4f}\n")
                        f.write(
                            "    (centroid-distance / mean-intra-radius; >1 = distinct)\n"
                        )
                    if cen is not None:
                        f.write(f"  Centroid distance  : {cen:.4f}\n")
                f.write("\n")

            # Position dependence
            if "position_dependence" in m:
                pos = m["position_dependence"]
                f.write("Position Dependence\n")
                f.write("-" * 30 + "\n")
                if "error" in pos:
                    f.write(f"  Error: {pos['error']}\n")
                else:
                    r2 = pos.get("position_r2_mean")
                    r2s = pos.get("position_r2_std")
                    n = pos.get("n_tokens_used")
                    if r2 is not None:
                        f.write(
                            f"  Ridge R² (position pred.) : {r2:.4f}"
                            + (f" ± {r2s:.4f}" if r2s is not None else "")
                            + "\n"
                        )
                        f.write(
                            "    (near 0 → RoPE-like; near 1 → learned abs. position)\n"
                        )
                    if n is not None:
                        f.write(f"  Tokens used               : {n}\n")
                f.write("\n")

            # Saved plots
            if self.data.get("plot_paths"):
                f.write("Generated Plots\n")
                f.write("-" * 30 + "\n")
                for key, path in self.data["plot_paths"].items():
                    f.write(f"  {key:<25} : {path}\n")
                f.write("\n")

        return str(json_path), str(txt_path)
