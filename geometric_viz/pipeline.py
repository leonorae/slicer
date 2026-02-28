"""
Main orchestration: GeometricFertilityPipeline ties all components together.

Usage (Python API)::

    from geometric_viz import GeometricFertilityPipeline

    pipeline = GeometricFertilityPipeline("gpt2", output_dir="./outputs")
    results = pipeline.run(prompts=[...])
    print(results["metrics"])

See run_pipeline.py for the command-line interface.
"""

import re
import time
from pathlib import Path
from typing import Dict, List, Optional

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from .analyzers import PCAAnalyzer, UMAPAnalyzer
from .extractor import ActivationExtractor
from .metrics import (
    compute_attention_mlp_separability,
    compute_intrinsic_dimensionality,
    compute_position_dependence,
)
from .report import GeometryReport
from .visualizer import GeometryVisualizer


class GeometricFertilityPipeline:
    """
    End-to-end geometric analysis pipeline.

    Loads a HuggingFace model, runs forward passes over a set of text prompts,
    extracts internal activations, applies PCA and UMAP, computes quantitative
    metrics, and saves visualisations and a report.

    Parameters
    ----------
    model_name : HuggingFace model identifier (e.g. ``"gpt2"``).
    output_dir : Root directory for all run outputs (a timestamped subdirectory
                 is created per run).
    device     : ``"cuda"``, ``"cpu"``, or ``None`` for auto-detection.
    max_tokens : Maximum tokens per extraction point after subsampling.
    """

    def __init__(
        self,
        model_name: str,
        output_dir: str = "./outputs",
        device: Optional[str] = None,
        max_tokens: int = 10_000,
    ):
        self.model_name = model_name
        self.output_dir = output_dir
        self.max_tokens = max_tokens

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device

        print(f"[Pipeline] Loading '{model_name}' on {device} …")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        if self.tokenizer.pad_token is None:
            # Many causal-LM tokenizers lack a pad token; use EOS as a stand-in.
            self.tokenizer.pad_token = self.tokenizer.eos_token

        load_kwargs: dict = {"torch_dtype": torch.float32}
        if device != "cpu":
            load_kwargs["device_map"] = device

        self.model = AutoModelForCausalLM.from_pretrained(model_name, **load_kwargs)
        if device == "cpu":
            self.model = self.model.to(device)
        self.model.eval()
        print("[Pipeline] Model loaded.")

    # ------------------------------------------------------------------

    def run(
        self,
        prompts: List[str],
        max_length: int = 128,
        batch_size: int = 4,
        n_pca_components: int = 50,
        umap_sample_size: int = 5_000,
        n_umap_neighbors: int = 15,
        skip_umap: bool = False,
    ) -> dict:
        """
        Run the full analysis pipeline on ``prompts``.

        Parameters
        ----------
        prompts           : List of text strings to forward through the model.
        max_length        : Maximum token length (longer sequences are truncated).
        batch_size        : Forward-pass batch size.
        n_pca_components  : Number of PCA components per extraction point.
        umap_sample_size  : Tokens sampled per point when fitting UMAP.
        n_umap_neighbors  : UMAP ``n_neighbors`` hyperparameter.
        skip_umap         : If True, skip UMAP entirely (faster for quick tests).

        Returns
        -------
        dict with keys:
            run_dir       : str – path to the run output directory
            activations   : dict mapping name -> ndarray (n_tokens, hidden)
            pca_results   : dict mapping name -> PCA result dict
            umap_results  : dict mapping name -> ndarray (n_tokens, 2)
            metrics       : dict of all computed metrics
            plot_paths    : dict of plot-type -> PNG file path
            report_paths  : dict with 'json' and 'txt' keys
        """
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        safe_model = re.sub(r"[^\w\-]", "_", self.model_name)
        run_dir = Path(self.output_dir) / f"{safe_model}_{timestamp}"
        run_dir.mkdir(parents=True, exist_ok=True)

        report = GeometryReport(self.model_name, str(run_dir))
        viz = GeometryVisualizer(str(run_dir))

        # ── 1. Extract activations ─────────────────────────────────────────
        print(f"[Pipeline] Extracting activations from {len(prompts)} prompts …")
        extractor = ActivationExtractor(
            self.model, self.model_name, max_tokens=self.max_tokens
        )
        activations, per_seq_data = extractor.extract(
            prompts, self.tokenizer, max_length=max_length, batch_size=batch_size
        )
        n_points = len(activations)
        total_tokens = sum(arr.shape[0] for arr in activations.values())
        print(
            f"[Pipeline] {n_points} extraction points, "
            f"~{total_tokens:,} total token-vectors collected."
        )

        # ── 2. PCA ─────────────────────────────────────────────────────────
        print("[Pipeline] Running PCA …")
        pca_analyzer = PCAAnalyzer(n_components=n_pca_components)
        pca_results = pca_analyzer.fit_transform(activations)

        # Intrinsic dimensionality from the middle post-MLP residual layer
        post_mlp_keys = sorted(
            [k for k in pca_results if "post_mlp_residual" in k],
            key=lambda x: int(re.search(r"layer_(\d+)", x).group(1)),
        )
        if post_mlp_keys:
            mid_key = post_mlp_keys[len(post_mlp_keys) // 2]
            idim = compute_intrinsic_dimensionality(
                pca_results[mid_key]["explained_variance_ratio"]
            )
            report.add_metric("intrinsic_dimensionality", idim)
            print(
                f"[Pipeline] Intrinsic dim (mid-layer '{mid_key}'): "
                f"n_90%={idim['n_to_90_percent']}, "
                f"PR={idim['participation_ratio']:.2f}"
            )

        # Warn early if token count is low (affects metric reliability)
        n_real_tokens = len(per_seq_data["post_mlp_per_seq"]) and sum(
            len(s) for s in per_seq_data["post_mlp_per_seq"]
        )
        if n_real_tokens < 1_000:
            print(
                f"[Pipeline] WARNING: only {n_real_tokens} real tokens collected. "
                "Metrics (especially position R²) will be unreliable. "
                "Consider adding more/longer prompts or raising --max_length."
            )

        scree_path = viz.plot_pca_scree(pca_results, self.model_name)
        report.add_plot("pca_scree", scree_path)

        pca_scatter_path = viz.plot_pca_scatter(pca_results, self.model_name)
        report.add_plot("pca_scatter", pca_scatter_path)

        # Positional-embedding helix (GPT-2 / absolute-position models only)
        wpe_weight = None
        for attr_path in ["transformer.wpe", "gpt_neox.embed_in"]:
            try:
                mod = self.model
                for part in attr_path.split("."):
                    mod = getattr(mod, part)
                if hasattr(mod, "weight"):
                    wpe_weight = mod.weight.detach().cpu().float().numpy()
                    break
            except AttributeError:
                continue

        if wpe_weight is not None:
            print("[Pipeline] Generating positional-embedding helix …")
            helix_path = viz.plot_positional_embedding_helix(wpe_weight, self.model_name)
            report.add_plot("positional_embedding_helix", helix_path)
            print(f"[Pipeline] Helix saved: {helix_path}")
        else:
            print("[Pipeline] Helix skipped (no wpe.weight found — RoPE model).")

        # ── 3. UMAP ────────────────────────────────────────────────────────
        umap_results: Dict[str, object] = {}
        if not skip_umap:
            print("[Pipeline] Running UMAP (this may take a while) …")
            umap_analyzer = UMAPAnalyzer(n_neighbors=n_umap_neighbors)
            umap_results = umap_analyzer.fit_transform(
                activations, sample_size=umap_sample_size
            )
            p1 = viz.plot_umap_by_component(umap_results, self.model_name)
            p2 = viz.plot_umap_layer_evolution(umap_results, self.model_name)
            p3 = viz.plot_umap_combined(umap_results, self.model_name)
            report.add_plot("umap_by_component", p1)
            report.add_plot("umap_layer_evolution", p2)
            report.add_plot("umap_combined", p3)
        else:
            print("[Pipeline] UMAP skipped (--no_umap).")

        # ── 4. Quantitative metrics ────────────────────────────────────────
        print("[Pipeline] Computing attention–MLP separability …")
        sep = compute_attention_mlp_separability(activations)
        report.add_metric("attention_mlp_separability", sep)
        if "silhouette_score" in sep:
            print(
                f"[Pipeline] Silhouette={sep['silhouette_score']:.3f}, "
                f"sep-ratio={sep['separability_ratio']:.3f}"
            )

        print("[Pipeline] Computing position dependence …")
        pos_dep = compute_position_dependence(
            per_seq_data["post_mlp_per_seq"],
            per_seq_data["seq_positions"],
        )
        report.add_metric("position_dependence", pos_dep)
        if "position_r2_mean" in pos_dep:
            print(
                f"[Pipeline] Position R²={pos_dep['position_r2_mean']:.3f} "
                f"± {pos_dep['position_r2_std']:.3f}"
            )

        # ── 5. Save report ─────────────────────────────────────────────────
        print("[Pipeline] Saving report …")
        json_path, txt_path = report.save()
        print(f"[Pipeline] JSON report : {json_path}")
        print(f"[Pipeline] Text summary: {txt_path}")

        return {
            "run_dir": str(run_dir),
            "activations": activations,
            "pca_results": pca_results,
            "umap_results": umap_results,
            "metrics": report.data["metrics"],
            "plot_paths": report.data["plot_paths"],
            "report_paths": {"json": json_path, "txt": txt_path},
        }
