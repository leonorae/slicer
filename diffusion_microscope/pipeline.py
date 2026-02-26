"""
End-to-end orchestration for the Diffusion Microscope.

Phases:
    1. Data preparation  – extract paired (LLM activation, CLIP embedding) data
    2. Projection training – fit the linear LLM→CLIP map via Ridge regression
    3. Validation          – collapse check, R², nearest-neighbour recall
    4. Generation          – render LLM activations as images via SD
    5. Analysis            – layer sweeps, CFG curves, interpolations, grids
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Optional, Sequence

import numpy as np


class MicroscopePipeline:
    """
    High-level pipeline that orchestrates data extraction, projection training,
    and image generation.

    Parameters
    ----------
    llm_model_name  : HuggingFace model name for the LLM (e.g. 'gpt2')
    sd_model_id     : HuggingFace model ID for Stable Diffusion
    clip_model_name : OpenCLIP architecture (must match SD's text encoder)
    clip_pretrained : OpenCLIP weights tag
    output_dir      : root directory for all outputs
    device          : 'cpu' or 'cuda'
    """

    def __init__(
        self,
        llm_model_name: str = "gpt2",
        sd_model_id: str = "sd-legacy/stable-diffusion-v1-5",
        clip_model_name: str = "ViT-L-14",
        clip_pretrained: str = "openai",
        output_dir: str = "./microscope_outputs",
        device: str = "cpu",
    ):
        self.llm_model_name = llm_model_name
        self.sd_model_id = sd_model_id
        self.clip_model_name = clip_model_name
        self.clip_pretrained = clip_pretrained
        self.output_dir = Path(output_dir)
        self.device = device

    # ------------------------------------------------------------------
    # Phase 1: Data preparation
    # ------------------------------------------------------------------

    def prepare_training_data(
        self,
        texts: list[str],
        *,
        layers: Optional[list[int]] = None,
        max_length: int = 256,
        batch_size: int = 4,
        cache_dir: Optional[str] = None,
    ) -> dict:
        """
        Extract paired LLM activations and CLIP embeddings for training.

        Parameters
        ----------
        texts      : training corpus (5k–10k diverse texts recommended)
        layers     : LLM layers to extract (default: all)
        cache_dir  : if set, save/load extracted data here

        Returns
        -------
        dict with keys:
            llm_activations  : {layer_idx: (n_texts, llm_dim)}
            clip_embeddings  : (n_texts, clip_dim)
            texts            : the input texts
        """
        from .clip_bridge import (
            extract_clip_text_embeddings,
            extract_llm_last_token_activations,
        )

        cache = Path(cache_dir) if cache_dir else None

        # Check cache
        if cache and (cache / "clip_embeddings.npy").exists():
            print("[Microscope] Loading cached training data …")
            clip_embs = np.load(cache / "clip_embeddings.npy")
            llm_acts = {}
            for p in sorted(cache.glob("llm_layer_*.npy")):
                layer_idx = int(p.stem.split("_")[-1])
                llm_acts[layer_idx] = np.load(p)
            return {
                "llm_activations": llm_acts,
                "clip_embeddings": clip_embs,
                "texts": texts,
            }

        # Extract CLIP embeddings
        print(f"[Microscope] Extracting CLIP embeddings ({self.clip_model_name}) …")
        clip_embs = extract_clip_text_embeddings(
            texts,
            clip_model_name=self.clip_model_name,
            pretrained=self.clip_pretrained,
            device=self.device,
        )
        print(f"[Microscope]   → {clip_embs.shape}")

        # Extract LLM activations
        print(f"[Microscope] Extracting LLM activations ({self.llm_model_name}) …")
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(self.llm_model_name)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        model = AutoModelForCausalLM.from_pretrained(
            self.llm_model_name,
            torch_dtype=torch.float32,
        )

        llm_acts = extract_llm_last_token_activations(
            texts, model, tokenizer,
            layers=layers,
            max_length=max_length,
            device=self.device,
            batch_size=batch_size,
        )
        for k, v in sorted(llm_acts.items()):
            print(f"[Microscope]   layer {k:3d} → {v.shape}")

        # Free LLM memory
        del model
        if self.device.startswith("cuda"):
            torch.cuda.empty_cache()

        # Cache if requested
        if cache:
            cache.mkdir(parents=True, exist_ok=True)
            np.save(cache / "clip_embeddings.npy", clip_embs)
            for layer_idx, arr in llm_acts.items():
                np.save(cache / f"llm_layer_{layer_idx:04d}.npy", arr)
            print(f"[Microscope] Cached training data → {cache}")

        return {
            "llm_activations": llm_acts,
            "clip_embeddings": clip_embs,
            "texts": texts,
        }

    # ------------------------------------------------------------------
    # Phase 2 + 3a: Train one projection per layer (recommended)
    # ------------------------------------------------------------------

    def train_layer_projections(
        self,
        data: dict,
        *,
        alpha: str = "auto",
        val_fraction: float = 0.15,
    ) -> dict:
        """
        Train a separate LinearProjection for every LLM layer.

        Each layer gets its own Ridge map trained against the same paired CLIP
        targets.  ``alpha="auto"`` selects the best regularisation per layer via
        RidgeCV cross-validation, which is especially useful because early layers
        typically need stronger regularisation than the final layer.

        Parameters
        ----------
        data         : output of prepare_training_data()
        alpha        : per-layer regularisation — float or ``"auto"``
        val_fraction : held-out fraction for per-layer validation

        Returns
        -------
        dict with:
            projection_set : LayerProjectionSet (one projection per layer)
            validation     : {layer_idx: metrics_dict}
        """
        from .projection import LayerProjectionSet

        llm_acts = data["llm_activations"]
        clip_embs = data["clip_embeddings"]

        n_layers = len(llm_acts)
        print(
            f"[Microscope] Training per-layer projections: "
            f"{n_layers} layers, alpha={alpha!r}",
            flush=True,
        )

        lps = LayerProjectionSet(alpha=alpha)
        val_metrics = lps.fit(llm_acts, clip_embs, val_fraction=val_fraction)

        r2_vals = [m["projection_r2"] for m in val_metrics.values()]
        best_layer = max(val_metrics, key=lambda l: val_metrics[l]["projection_r2"])
        print(
            f"[Microscope] R² range: {min(r2_vals):.4f} → {max(r2_vals):.4f} "
            f"(best: layer {best_layer}  R²={val_metrics[best_layer]['projection_r2']:.4f})",
            flush=True,
        )

        return {"projection_set": lps, "validation": val_metrics}

    # ------------------------------------------------------------------
    # Phase 2 + 3b: Train and validate a single-layer projection (legacy)
    # ------------------------------------------------------------------

    def train_projection(
        self,
        data: dict,
        *,
        target_layer: Optional[int] = None,
        alpha: float = 1.0,
        val_fraction: float = 0.15,
    ) -> dict:
        """
        Train a linear projection from LLM activations (one layer) to CLIP
        embeddings, and validate on a held-out split.

        Parameters
        ----------
        data          : output of prepare_training_data()
        target_layer  : which LLM layer to project from (default: last)
        alpha         : Ridge regularisation
        val_fraction  : fraction of data held out for validation

        Returns
        -------
        dict with:
            projection : trained LinearProjection
            validation : validation metrics dict
            layer      : the layer index used
        """
        from .projection import LinearProjection

        llm_acts = data["llm_activations"]
        clip_embs = data["clip_embeddings"]

        if target_layer is None:
            target_layer = max(llm_acts.keys())
        if target_layer not in llm_acts:
            raise KeyError(
                f"layer {target_layer} not in extracted layers: "
                f"{sorted(llm_acts.keys())}"
            )

        X = llm_acts[target_layer]
        y = clip_embs

        # Train / val split
        n = len(X)
        n_val = max(1, int(n * val_fraction))
        rng = np.random.default_rng(42)
        idx = rng.permutation(n)
        train_idx, val_idx = idx[n_val:], idx[:n_val]

        print(
            f"[Microscope] Training projection: layer {target_layer}, "
            f"alpha={alpha}, train={len(train_idx)}, val={len(val_idx)}"
        )

        proj = LinearProjection(alpha=alpha)
        proj.fit(X[train_idx], y[train_idx])
        print(f"[Microscope]   → {proj}")

        # Validate
        val_metrics = proj.validate(X[val_idx], y[val_idx])
        print(f"[Microscope] Validation:")
        print(f"[Microscope]   R²             = {val_metrics['projection_r2']:.4f}")
        print(f"[Microscope]   cosine dist    = {val_metrics['cosine_dist_mean']:.4f} "
              f"± {val_metrics['cosine_dist_std']:.4f}")
        for k, v in val_metrics.items():
            if "recall" in k:
                print(f"[Microscope]   {k} = {v:.4f}")

        return {
            "projection": proj,
            "validation": val_metrics,
            "layer": target_layer,
        }

    # ------------------------------------------------------------------
    # Phase 4: Image generation
    # ------------------------------------------------------------------

    def generate_layer_sweep(
        self,
        text: str,
        projection,
        data: dict,
        *,
        layers: Optional[list[int]] = None,
        cfg_scales: Sequence[float] = (3.0, 7.0, 12.0),
        seed: int = 42,
        num_inference_steps: int = 30,
        save_dir: Optional[str] = None,
    ) -> dict:
        """
        Generate a grid of images: rows = layers, columns = CFG scales.

        Parameters
        ----------
        text       : the text to visualise across layers
        projection : trained LinearProjection
        data       : output of prepare_training_data() (for the LLM activations)
        layers     : subset of layers to visualise (default: evenly spaced)
        """
        from .generator import DiffusionMicroscope

        llm_acts = data["llm_activations"]
        available = sorted(llm_acts.keys())

        if layers is None:
            # Pick ~6 evenly spaced layers including first and last
            n = len(available)
            step = max(1, n // 6)
            layers = available[::step]
            if available[-1] not in layers:
                layers.append(available[-1])

        # We need per-text activations — extract just for this text
        # (data has all texts; we need the specific one)
        # Re-extract for the single text
        print(f"[Microscope] Extracting single-text activations for layer sweep …")
        from .clip_bridge import extract_llm_last_token_activations
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(self.llm_model_name)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        model = AutoModelForCausalLM.from_pretrained(
            self.llm_model_name, torch_dtype=torch.float32,
        )

        single_acts = extract_llm_last_token_activations(
            [text], model, tokenizer,
            layers=layers, device=self.device,
        )
        del model
        if self.device.startswith("cuda"):
            torch.cuda.empty_cache()

        # Squeeze batch dim: {layer: (1, dim)} → {layer: (dim,)}
        single_acts = {k: v[0] for k, v in single_acts.items()}

        # Generate images
        print(f"[Microscope] Generating {len(layers)} layers × "
              f"{len(cfg_scales)} CFG scales = {len(layers)*len(cfg_scales)} images …")
        scope = DiffusionMicroscope(
            sd_model_id=self.sd_model_id,
            device=self.device,
            dtype="float16" if "cuda" in self.device else "float32",
        )
        images = scope.layer_sweep(
            single_acts, projection,
            cfg_scales=cfg_scales, seed=seed,
            num_inference_steps=num_inference_steps,
        )

        # Save grid
        if save_dir is None:
            ts = time.strftime("%Y%m%d_%H%M%S")
            save_dir = str(self.output_dir / f"sweep_{ts}")
        save_path = Path(save_dir)
        save_path.mkdir(parents=True, exist_ok=True)

        for (layer_idx, cfg), img in images.items():
            fname = f"layer{layer_idx:03d}_cfg{cfg:.1f}.png"
            img.save(save_path / fname)

        # Composite grid image
        grid_path = self._make_grid(images, layers, cfg_scales, text, save_path)

        print(f"[Microscope] Saved {len(images)} images → {save_path}")
        if grid_path:
            print(f"[Microscope] Grid  → {grid_path}")

        return {
            "images": images,
            "save_dir": str(save_path),
            "grid_path": grid_path,
        }

    @staticmethod
    def _make_grid(
        images: dict,
        layers: list[int],
        cfg_scales: Sequence[float],
        title: str,
        save_dir: Path,
    ) -> Optional[str]:
        """Assemble individual images into a labelled grid."""
        try:
            from PIL import Image, ImageDraw, ImageFont
        except ImportError:
            return None

        cell_w, cell_h = 256, 256
        label_h = 30
        header_h = 40
        n_rows = len(layers)
        n_cols = len(cfg_scales)
        row_label_w = 80

        grid_w = row_label_w + n_cols * cell_w
        grid_h = header_h + n_rows * (cell_h + label_h)

        grid = Image.new("RGB", (grid_w, grid_h), "white")
        draw = ImageDraw.Draw(grid)

        # Column headers (CFG values)
        for j, cfg in enumerate(cfg_scales):
            x = row_label_w + j * cell_w + cell_w // 2
            draw.text((x, 10), f"CFG {cfg:.1f}", fill="black", anchor="mt")

        # Rows
        for i, layer in enumerate(layers):
            y_top = header_h + i * (cell_h + label_h)

            # Row label
            draw.text(
                (row_label_w // 2, y_top + cell_h // 2),
                f"L{layer}", fill="black", anchor="mm",
            )

            for j, cfg in enumerate(cfg_scales):
                key = (layer, cfg)
                if key not in images:
                    continue
                img = images[key].resize((cell_w, cell_h))
                x = row_label_w + j * cell_w
                grid.paste(img, (x, y_top))

        # Title
        safe_title = title[:80] + ("…" if len(title) > 80 else "")
        path = save_dir / "grid.png"
        grid.save(path)
        return str(path)

    # ------------------------------------------------------------------
    # Full run
    # ------------------------------------------------------------------

    def run(
        self,
        training_texts: list[str],
        probe_texts: list[str],
        *,
        alpha: str = "auto",
        cfg_scales: Sequence[float] = (3.0, 7.0, 12.0),
        cache_dir: Optional[str] = None,
        seed: int = 42,
    ) -> dict:
        """
        Full pipeline: extract → train per-layer projections → validate → generate.

        Parameters
        ----------
        training_texts : texts for training (~5k recommended)
        probe_texts    : texts to visualise via layer sweeps
        alpha          : Ridge regularisation per layer, or ``"auto"`` to select
                         via cross-validation (recommended)
        cfg_scales     : SD guidance scales to sweep
        cache_dir      : cache extracted activations here
        """
        ts = time.strftime("%Y%m%d_%H%M%S")
        run_dir = self.output_dir / f"run_{ts}"
        run_dir.mkdir(parents=True, exist_ok=True)

        # Phase 1: data
        data = self.prepare_training_data(training_texts, cache_dir=cache_dir)

        # Phase 2+3: train one projection per layer
        train_result = self.train_layer_projections(data, alpha=alpha)
        proj_set = train_result["projection_set"]
        proj_set.save(str(run_dir / "projection_set"))

        # Phase 4: generate for each probe text
        sweep_results = []
        for i, text in enumerate(probe_texts):
            print(
                f"\n[Microscope] === Probe {i+1}/{len(probe_texts)}: "
                f"{text[:60]}{'…' if len(text) > 60 else ''} ===",
                flush=True,
            )
            sweep_dir = str(run_dir / f"probe_{i:03d}")
            result = self.generate_layer_sweep(
                text, proj_set, data,
                cfg_scales=cfg_scales, seed=seed, save_dir=sweep_dir,
            )
            sweep_results.append(result)

        # Save summary — per-layer validation condensed to avoid huge JSON
        layer_summary = {
            str(l): {
                "projection_r2": round(m["projection_r2"], 5),
                "cosine_dist_mean": round(m["cosine_dist_mean"], 5),
                "alpha": m["alpha"],
            }
            for l, m in train_result["validation"].items()
        }
        summary = {
            "llm_model": self.llm_model_name,
            "sd_model": self.sd_model_id,
            "clip_model": f"{self.clip_model_name}/{self.clip_pretrained}",
            "alpha": alpha,
            "n_training_texts": len(training_texts),
            "n_probe_texts": len(probe_texts),
            "per_layer_validation": layer_summary,
            "probe_texts": probe_texts,
        }
        with open(run_dir / "summary.json", "w") as f:
            json.dump(summary, f, indent=2, default=str)

        print(f"\n[Microscope] Run complete → {run_dir}", flush=True)
        return {
            "run_dir": str(run_dir),
            "projection_set": proj_set,
            "validation": train_result["validation"],
            "sweeps": sweep_results,
        }
