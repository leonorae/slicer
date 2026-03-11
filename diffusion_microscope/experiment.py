"""
Batch experiment runner for the Diffusion Microscope.

Phases (run in order or selectively):
  train     – train Ridge maps (per_layer, single_layer, mixed_layer),
              store SVD spectra, populate manifest
  generate  – generate probe images for all (proj × probe × layer × cfg × seed)
  grids     – compose layer × CFG grid PNGs per (proj, text, seed)
  analyze   – load SVD data, produce erank-vs-layer plots and spectrum heatmaps
  dashboard – write self-contained dashboard.html

Every phase is idempotent: re-running skips completed work via manifest.json.

Config keys (experiment_config.json):
  models.llm / models.sd / models.clip_model / models.clip_pretrained
  projections.types          list of "per_layer" | "single_final_layer" | "mixed_layer"
  projections.alpha_values   list of floats or "auto"
  projections.training_data_size
  probe_texts                dict of {group: [text, ...]} or flat list
  layers                     null (= all) or [int, ...]
  cfg_values                 list of CFG floats
  seeds                      list of ints
  output.base_dir
"""

from __future__ import annotations

import json
import re
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


# ---------------------------------------------------------------------------
# Module-level helpers
# ---------------------------------------------------------------------------

_PROJ_MODE_MAP = {
    "single_final_layer": "single_layer",
    "mixed_layer": "mixed_layer",
    "per_layer": "per_layer",
}


def _slug(text: str, maxlen: int = 40) -> str:
    """Convert arbitrary text to a filesystem-safe slug."""
    s = re.sub(r"[^\w\s-]", "", text.lower())
    s = re.sub(r"[\s_]+", "_", s).strip("_")
    return s[:maxlen] or "text"


def _proj_key(proj_type: str, alpha: Any) -> str:
    if isinstance(alpha, float) and alpha == int(alpha):
        return f"{proj_type}_alpha{int(alpha)}"
    return f"{proj_type}_alpha{alpha}"


def _flatten_probe_texts(probe_cfg: Any) -> List[str]:
    """Return a deduplicated, order-preserving flat list of probe texts."""
    if isinstance(probe_cfg, list):
        texts = probe_cfg
    elif isinstance(probe_cfg, dict):
        texts = []
        for key, val in probe_cfg.items():
            if key == "interpolations":
                continue
            if isinstance(val, list):
                texts.extend(str(t) for t in val)
    else:
        texts = []
    # Deduplicate while preserving order
    seen: set = set()
    out: List[str] = []
    for t in texts:
        if t not in seen:
            seen.add(t)
            out.append(t)
    return out


# ---------------------------------------------------------------------------
# ExperimentRunner
# ---------------------------------------------------------------------------

class ExperimentRunner:
    """
    Orchestrates all experiment phases.

    Parameters
    ----------
    config        : dict loaded from experiment_config.json
    llm_model     : HuggingFace model ID for the LLM
    sd_model      : HuggingFace model ID for Stable Diffusion
    clip_model    : OpenCLIP architecture name
    clip_pretrained : OpenCLIP weights tag
    device        : "cpu" or "cuda"
    cache_dir     : directory for caching LLM training-data activations
    """

    PHASE_ORDER = ["train", "generate", "grids", "analyze", "dashboard"]

    def __init__(
        self,
        config: Dict,
        llm_model: str = "gpt2",
        sd_model: str = "sd-legacy/stable-diffusion-v1-5",
        clip_model: str = "ViT-L-14",
        clip_pretrained: str = "openai",
        device: str = "cpu",
        cache_dir: Optional[str] = None,
    ):
        self.config = config
        self.llm_model = llm_model
        self.sd_model = sd_model
        self.clip_model = clip_model
        self.clip_pretrained = clip_pretrained
        self.device = device

        out = config.get("output", {})
        self.base_dir = Path(out.get("base_dir", "./experiment_results"))
        self.proj_dir = self.base_dir / "projections"
        self.grids_dir = self.base_dir / "grids" / "by_projection"
        self.probe_cache_dir = self.base_dir / ".probe_cache"
        self.analysis_dir = self.base_dir / "analysis"
        self.manifest_path = self.base_dir / "manifest.json"
        self.cache_dir = Path(cache_dir) if cache_dir else self.base_dir / ".train_cache"

        self.base_dir.mkdir(parents=True, exist_ok=True)
        self.manifest = self._load_manifest()

        # Lazy model handles
        self._llm = None
        self._tokenizer = None
        self._sd = None

    # ------------------------------------------------------------------
    # Manifest
    # ------------------------------------------------------------------

    def _load_manifest(self) -> Dict:
        if self.manifest_path.exists():
            with open(self.manifest_path) as f:
                return json.load(f)
        return {
            "config": self.config,
            "layers": [],
            "projections": {},
            "images": {},
            "grids": {},
            "probe_corpus_distances": {},
        }

    def _save_manifest(self) -> None:
        with open(self.manifest_path, "w") as f:
            json.dump(self.manifest, f, indent=2)

    # ------------------------------------------------------------------
    # Lazy model loaders
    # ------------------------------------------------------------------

    def _load_llm(self):
        if self._llm is None:
            from transformers import AutoTokenizer, AutoModelForCausalLM
            print(f"Loading LLM: {self.llm_model}")
            self._tokenizer = AutoTokenizer.from_pretrained(self.llm_model)
            # GPT-2 and similar models have no pad token by default
            if self._tokenizer.pad_token is None:
                self._tokenizer.pad_token = self._tokenizer.eos_token
            self._llm = (
                AutoModelForCausalLM.from_pretrained(
                    self.llm_model, output_hidden_states=True
                )
                .to(self.device)
                .eval()
            )
        return self._llm, self._tokenizer

    def _load_sd(self):
        if self._sd is None:
            from .generator import DiffusionMicroscope
            print(f"Loading SD: {self.sd_model}")
            self._sd = DiffusionMicroscope(sd_model_id=self.sd_model, device=self.device)
        return self._sd

    # ------------------------------------------------------------------
    # Phase: train
    # ------------------------------------------------------------------

    def run_train(self, training_texts: List[str]) -> None:
        """Train all (projection_type × alpha) combinations; store SVD in manifest."""
        from .clip_bridge import (
            extract_clip_text_embeddings,
            extract_llm_last_token_activations,
        )
        from .projection import LinearProjection, LayerProjectionSet

        proj_cfg = self.config.get("projections", {})
        types = proj_cfg.get("types", ["per_layer"])
        alphas = proj_cfg.get("alpha_values", [1.0])
        n_train = proj_cfg.get("training_data_size", len(training_texts))
        texts = training_texts[:n_train]

        # Resolve which layers to train
        cfg_layers: Optional[List[int]] = self.config.get("layers")

        # --- Extract / load CLIP embeddings (training targets) ---
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        clip_cache = self.cache_dir / "clip_embeddings.npy"
        if clip_cache.exists():
            print("  Loading cached CLIP training embeddings…")
            clip_embeddings = np.load(clip_cache)
        else:
            print(f"  Extracting CLIP embeddings for {len(texts)} texts…")
            clip_embeddings = extract_clip_text_embeddings(
                texts,
                clip_model_name=self.clip_model,
                pretrained=self.clip_pretrained,
                device=self.device,
            )
            np.save(clip_cache, clip_embeddings)

        # --- Extract / load LLM activations (training inputs) ---
        # Run LLM once to discover n_layers, then cache per layer.
        llm, tok = self._load_llm()

        # Discover number of layers from the model config
        n_layers = llm.config.num_hidden_layers
        all_layers = list(range(n_layers))
        layers_to_use = cfg_layers if cfg_layers else all_layers

        # Load or extract activations per layer
        layer_act: Dict[int, np.ndarray] = {}
        missing = [l for l in layers_to_use if not (self.cache_dir / f"llm_layer_{l:04d}.npy").exists()]
        if missing:
            print(f"  Extracting LLM activations for {len(texts)} texts ({n_layers} layers)…")
            all_acts = extract_llm_last_token_activations(
                texts, llm, tok,
                layers=layers_to_use,
                device=self.device,
            )
            for layer_idx, acts in all_acts.items():
                np.save(self.cache_dir / f"llm_layer_{layer_idx:04d}.npy", acts)

        for layer_idx in layers_to_use:
            layer_act[layer_idx] = np.load(self.cache_dir / f"llm_layer_{layer_idx:04d}.npy")

        # Store layer list in manifest
        self.manifest["layers"] = layers_to_use

        # --- Train each (type, alpha) combination ---
        for proj_type in types:
            for alpha in alphas:
                key = _proj_key(proj_type, alpha)
                if key in self.manifest["projections"]:
                    print(f"  [skip] {key} already in manifest")
                    continue

                print(f"\n  Training {key}…")
                save_path = self.proj_dir / key
                mode = _PROJ_MODE_MAP.get(proj_type, "per_layer")
                alpha_val: Any = "auto" if str(alpha).lower() == "auto" else float(alpha)

                val_metrics: Dict = {}
                svd_data: Dict = {}

                if mode == "per_layer":
                    lps = LayerProjectionSet(alpha=alpha_val)
                    val_metrics_raw = lps.fit(layer_act, clip_embeddings)
                    lps.save(str(save_path))
                    # Collect SVD data per layer from fitted projections
                    for layer_idx, proj in lps.projections.items():
                        val_metrics[str(layer_idx)] = val_metrics_raw[layer_idx]
                        svd_data[str(layer_idx)] = {
                            "erank": proj.erank,
                            "condition_number": proj.condition_number,
                            "n_visible": proj._meta.get("n_visible"),
                        }

                elif mode == "single_layer":
                    last_layer = max(layers_to_use)
                    X = layer_act[last_layer]
                    a = alpha_val if alpha_val != "auto" else "auto"
                    if a == "auto":
                        from .projection import _auto_alpha
                        a = _auto_alpha(X, clip_embeddings)
                    proj = LinearProjection(alpha=float(a))
                    n = len(clip_embeddings)
                    n_val = max(1, int(n * 0.15))
                    rng = np.random.default_rng(42)
                    idx = rng.permutation(n)
                    proj.fit(X[idx[n_val:]], clip_embeddings[idx[n_val:]])
                    metrics = proj.validate(X[idx[:n_val]], clip_embeddings[idx[:n_val]])
                    metrics["alpha"] = float(a)
                    val_metrics["single"] = metrics
                    proj.save(str(save_path))
                    svd_data["single"] = {
                        "erank": proj.erank,
                        "condition_number": proj.condition_number,
                        "n_visible": proj._meta.get("n_visible"),
                    }

                elif mode == "mixed_layer":
                    # Pool activations from all layers
                    X_mixed = np.concatenate(
                        [layer_act[l] for l in sorted(layers_to_use)], axis=0
                    )
                    y_mixed = np.tile(clip_embeddings, (len(layers_to_use), 1))
                    a = alpha_val if alpha_val != "auto" else "auto"
                    if a == "auto":
                        from .projection import _auto_alpha
                        a = _auto_alpha(X_mixed, y_mixed)
                    proj = LinearProjection(alpha=float(a))
                    n = len(y_mixed)
                    n_val = max(1, int(n * 0.15))
                    rng = np.random.default_rng(42)
                    idx = rng.permutation(n)
                    proj.fit(X_mixed[idx[n_val:]], y_mixed[idx[n_val:]])
                    metrics = proj.validate(X_mixed[idx[:n_val]], y_mixed[idx[:n_val]])
                    metrics["alpha"] = float(a)
                    val_metrics["mixed"] = metrics
                    proj.save(str(save_path))
                    svd_data["mixed"] = {
                        "erank": proj.erank,
                        "condition_number": proj.condition_number,
                        "n_visible": proj._meta.get("n_visible"),
                    }

                self.manifest["projections"][key] = {
                    "type": proj_type,
                    "mode": mode,
                    "alpha": alpha_val if alpha_val != "auto" else "auto",
                    "save_path": str(save_path.relative_to(self.base_dir)),
                    "validation": val_metrics,
                    "svd": svd_data,
                }
                self._save_manifest()
                print(f"  Saved: {save_path}")

        print(f"\nTrain phase complete. {len(self.manifest['projections'])} projection(s) ready.")

    # ------------------------------------------------------------------
    # Phase: generate
    # ------------------------------------------------------------------

    def run_generate(self) -> None:
        """Generate probe images for all (proj × probe × layer × cfg × seed)."""
        from .clip_bridge import extract_llm_last_token_activations
        from .projection import LinearProjection, LayerProjectionSet

        probe_texts = _flatten_probe_texts(self.config.get("probe_texts", []))
        cfg_scales = self.config.get("cfg_values", [7.5])
        seeds = self.config.get("seeds", [42])
        layers: List[int] = self.manifest.get("layers", [])

        if not probe_texts:
            print("No probe texts configured — skipping generate phase.")
            return
        if not layers:
            print("No layers in manifest (run train first) — skipping generate phase.")
            return
        if not self.manifest["projections"]:
            print("No projections in manifest (run train first) — skipping generate phase.")
            return

        llm, tok = self._load_llm()
        sd = self._load_sd()

        # Extract and cache probe activations
        probe_acts: Dict[str, Dict[int, np.ndarray]] = {}
        for text in probe_texts:
            slug = _slug(text)
            text_cache = self.probe_cache_dir / slug
            text_cache.mkdir(parents=True, exist_ok=True)
            missing = [l for l in layers if not (text_cache / f"layer_{l:04d}.npy").exists()]
            if missing:
                print(f'  Extracting probe activations: "{text[:50]}"')
                acts = extract_llm_last_token_activations(
                    [text], llm, tok, layers=layers, device=self.device
                )
                for layer_idx, arr in acts.items():
                    np.save(text_cache / f"layer_{layer_idx:04d}.npy", arr[0])
            probe_acts[text] = {
                l: np.load(text_cache / f"layer_{l:04d}.npy") for l in layers
            }

        # Cache loaded projections in memory (one per key)
        loaded_projs: Dict[str, Any] = {}
        for key, rec in self.manifest["projections"].items():
            proj_path = self.base_dir / rec["save_path"]
            mode = rec["mode"]
            if mode == "per_layer":
                loaded_projs[key] = LayerProjectionSet.load(str(proj_path))
            else:
                loaded_projs[key] = LinearProjection.load(str(proj_path))

        # Compute corpus-distance metrics for each (probe, projection, layer).
        # d_act: normalised L2 distance from corpus LLM-activation centroid (z-scored).
        # d_clip: cosine distance from corpus CLIP centroid in CLIP space.
        # These are stored in manifest["probe_corpus_distances"] so downstream
        # analysis can partial-out corpus distance when interpreting LPIPS differences.
        self._compute_probe_corpus_distances(probe_texts, probe_acts, layers, loaded_projs)

        # Record token count and LLM perplexity per probe text.  Both are
        # cheap (single forward pass per text) and address two confounders:
        #   n_tokens   – prompt length co-varies with unusualness tier in Exp 3
        #   perplexity – proxy for distance from LLM pretraining corpus; addresses
        #                the cross-model confound for GPT-2 vs Pythia comparisons
        self._compute_probe_text_stats(probe_texts, llm, tok)

        n_total = (
            len(self.manifest["projections"])
            * len(probe_texts)
            * len(layers)
            * len(cfg_scales)
            * len(seeds)
        )
        n_done = 0
        n_skipped = 0

        for key, proj in loaded_projs.items():
            mode = self.manifest["projections"][key]["mode"]
            for text in probe_texts:
                slug = _slug(text)
                img_dir = self.grids_dir / key / slug / "per_layer"
                img_dir.mkdir(parents=True, exist_ok=True)
                for layer_idx in layers:
                    activation = probe_acts[text][layer_idx]
                    if mode == "per_layer":
                        if layer_idx not in proj:
                            continue
                        clip_vec = proj.transform(layer_idx, activation)
                    else:
                        clip_vec = proj.transform(activation)
                    for cfg in cfg_scales:
                        seed_rel_paths: List[str] = []
                        for seed in seeds:
                            rel = (
                                img_dir.relative_to(self.base_dir)
                                / f"L{layer_idx:04d}_CFG{cfg:g}_seed{seed}.png"
                            )
                            rel_str = str(rel)
                            seed_rel_paths.append(rel_str)
                            if rel_str in self.manifest["images"]:
                                n_skipped += 1
                                continue
                            img = sd.generate_from_vector(
                                clip_vec,
                                guidance_scale=cfg,
                                seed=seed,
                            )
                            out_path = self.base_dir / rel
                            img.save(str(out_path))
                            self.manifest["images"][rel_str] = {
                                "projection": key,
                                "probe_text": text,
                                "layer": layer_idx,
                                "cfg": cfg,
                                "seed": seed,
                            }
                            n_done += 1
                            if n_done % 50 == 0:
                                self._save_manifest()
                                print(f"  Generated {n_done}/{n_total - n_skipped} images…")

                        # Compute per-pixel variance across seeds for this
                        # (proj, probe, layer, CFG) group.  Requires ≥ 4 seed
                        # images to exist; skipped silently otherwise.
                        # Stored in manifest["seed_variance"] for Phase 4
                        # diffusion-prior quantification.
                        self._compute_group_seed_variance(
                            key, slug, layer_idx, cfg, seed_rel_paths
                        )

        self._save_manifest()
        print(
            f"\nGenerate phase complete. "
            f"{n_done} new images, {n_skipped} skipped (already done)."
        )

    def _compute_probe_corpus_distances(
        self,
        probe_texts: List[str],
        probe_acts: Dict[str, Dict[int, np.ndarray]],
        layers: List[int],
        loaded_projs: Dict[str, Any],
    ) -> None:
        """
        Compute and store corpus-distance metrics per (probe, projection, layer).

        Two metrics:
          d_act  – normalised L2 distance of the probe LLM activation from the corpus
                   centroid in activation space.  Uses each layer projection's z-score
                   parameters (scaler_mean / scaler_scale) as corpus statistics.
                   Equivalent to a spherical Mahalanobis distance.

          d_clip – cosine distance of the probe's projected CLIP vector from the corpus
                   CLIP centroid.  Requires corpus_clip_centroid.npy to have been saved
                   alongside the projection (available for projections trained after this
                   change).  None when centroid file is absent.

        Results are written to manifest["probe_corpus_distances"][slug][proj_key][layer].
        Already-computed entries are skipped so the phase remains idempotent.
        """
        from sklearn.metrics.pairwise import cosine_distances
        from .projection import LayerProjectionSet

        distances = self.manifest.setdefault("probe_corpus_distances", {})

        for key, proj in loaded_projs.items():
            mode = self.manifest["projections"][key]["mode"]
            if mode != "per_layer":
                continue
            assert isinstance(proj, LayerProjectionSet)

            for text in probe_texts:
                slug = _slug(text)
                distances.setdefault(slug, {}).setdefault(key, {})

                for layer_idx in layers:
                    layer_key = str(layer_idx)
                    if layer_key in distances[slug][key]:
                        continue  # idempotent
                    if layer_idx not in proj:
                        continue

                    layer_proj = proj.projections[layer_idx]
                    activation = probe_acts[text][layer_idx]

                    d_act = layer_proj.corpus_distance(activation)

                    d_clip: Optional[float] = None
                    if layer_proj.corpus_clip_centroid is not None:
                        clip_vec = layer_proj.transform(activation)
                        d_clip = float(
                            cosine_distances(
                                clip_vec[np.newaxis],
                                layer_proj.corpus_clip_centroid[np.newaxis],
                            )[0, 0]
                        )

                    distances[slug][key][layer_key] = {
                        "d_act": d_act,
                        "d_clip": d_clip,
                    }

        self._save_manifest()

    def _compute_probe_text_stats(
        self,
        probe_texts: List[str],
        llm: Any,
        tok: Any,
    ) -> None:
        """
        Record per-probe text statistics addressing two confounders:

          n_tokens   – token count of the probe text.  Prompt length co-varies
                       with the "unusualness" tier in Exp 3 (unusual probes are
                       systematically longer), which could confound LPIPS results.

          perplexity – LLM perplexity of the probe text.  Serves as a proxy for
                       distance from the LLM's pretraining corpus, addressing the
                       cross-model confound: GPT-2 (WebText) and Pythia (The Pile)
                       assign different perplexities to the same probe, so a gap
                       in LPIPS between models could reflect pretraining-corpus
                       coverage rather than architecture.

        Results are stored in manifest["probe_text_stats"][slug].
        Already-computed entries are skipped (idempotent).
        """
        import torch

        stats = self.manifest.setdefault("probe_text_stats", {})

        for text in probe_texts:
            slug = _slug(text)
            if slug in stats:
                continue

            inputs = tok(text, return_tensors="pt").to(self.device)
            n_tokens = int(inputs["input_ids"].shape[1])

            try:
                with torch.no_grad():
                    outputs = llm(**inputs, labels=inputs["input_ids"])
                perplexity = float(torch.exp(outputs.loss).item())
            except Exception:
                perplexity = None

            stats[slug] = {
                "text": text,
                "n_tokens": n_tokens,
                "perplexity": perplexity,
            }

        self._save_manifest()

    def _compute_group_seed_variance(
        self,
        proj_key: str,
        text_slug: str,
        layer_idx: int,
        cfg: float,
        seed_rel_paths: List[str],
    ) -> None:
        """
        Compute mean per-pixel variance across all seed images for one
        (projection, probe, layer, CFG) group.

        This is the Phase 4 diffusion-prior quantification metric: low variance
        means the CLIP conditioning vector determines the image content; high
        variance means the diffusion prior is filling in the details.  Comparing
        variance between alpha=1 and alpha=1000 images (same probe, same layer)
        tests whether the LPIPS signal reflects real conditioning differences or
        just diffusion-prior noise.

        Requires ≥ 4 seed images to be present on disk; skipped silently
        otherwise (e.g., if the experiment was interrupted mid-group).
        Results are stored in manifest["seed_variance"][var_key] and are
        idempotent across re-runs.
        """
        from PIL import Image

        var_key = f"{proj_key}/{text_slug}/L{layer_idx:04d}/CFG{cfg:g}"
        if var_key in self.manifest.get("seed_variance", {}):
            return

        imgs = []
        for rel_str in seed_rel_paths:
            p = self.base_dir / rel_str
            if p.exists():
                imgs.append(
                    np.array(Image.open(p).convert("RGB"), dtype=np.float32)
                )

        if len(imgs) < 4:
            return

        stack = np.stack(imgs, axis=0)  # (N, H, W, 3)
        mean_var = float(stack.var(axis=0).mean())

        self.manifest.setdefault("seed_variance", {})[var_key] = {
            "mean_pixel_var": mean_var,
            "n_seeds": len(imgs),
        }

    # ------------------------------------------------------------------
    # Phase: grids
    # ------------------------------------------------------------------

    def run_grids(self) -> None:
        """Compose layer × CFG grid PNGs per (projection, probe text, seed)."""
        from PIL import Image, ImageDraw, ImageFont

        cfg_scales = sorted({rec["cfg"] for rec in self.manifest["images"].values()})
        seeds = sorted({rec["seed"] for rec in self.manifest["images"].values()})

        # Group images by (proj_key, text_slug, seed)
        groups: Dict[Tuple, Dict[Tuple[int, float], str]] = {}
        for rel_str, rec in self.manifest["images"].items():
            key = (rec["projection"], _slug(rec["probe_text"]), rec["seed"])
            groups.setdefault(key, {})[(rec["layer"], rec["cfg"])] = rel_str

        CELL = 256
        LABEL_H = 20

        for (proj_key, text_slug, seed), cell_map in groups.items():
            grid_rel = (
                Path("grids") / "by_projection" / proj_key / text_slug
                / f"grid_seed{seed}.png"
            )
            if str(grid_rel) in self.manifest["grids"]:
                continue

            layers_present = sorted({l for l, _ in cell_map.keys()})
            cfgs_present = sorted({c for _, c in cell_map.keys()})
            n_rows = len(layers_present)
            n_cols = len(cfgs_present)

            grid_w = LABEL_H + n_cols * CELL
            grid_h = LABEL_H + n_rows * CELL
            grid = Image.new("RGB", (grid_w, grid_h), color=(30, 30, 30))
            draw = ImageDraw.Draw(grid)

            try:
                font = ImageFont.load_default(size=11)
            except TypeError:
                font = ImageFont.load_default()

            # Column headers (CFG values)
            for ci, cfg in enumerate(cfgs_present):
                x = LABEL_H + ci * CELL + CELL // 2
                draw.text((x, 2), f"CFG {cfg:g}", fill=(200, 200, 200), anchor="mt", font=font)

            # Rows
            for ri, layer in enumerate(layers_present):
                y_top = LABEL_H + ri * CELL
                draw.text(
                    (2, y_top + CELL // 2),
                    f"L{layer}",
                    fill=(200, 200, 200),
                    anchor="lm",
                    font=font,
                )
                for ci, cfg in enumerate(cfgs_present):
                    x_left = LABEL_H + ci * CELL
                    img_path = self.base_dir / cell_map.get((layer, cfg), "")
                    if img_path.exists():
                        cell_img = Image.open(img_path).resize((CELL, CELL))
                        grid.paste(cell_img, (x_left, y_top))

            out_path = self.base_dir / grid_rel
            out_path.parent.mkdir(parents=True, exist_ok=True)
            grid.save(str(out_path))

            self.manifest["grids"][str(grid_rel)] = {
                "projection": proj_key,
                "text_slug": text_slug,
                "seed": seed,
                "path": str(grid_rel),
            }

        self._save_manifest()
        print(f"Grids phase complete. {len(self.manifest['grids'])} grid(s) produced.")

    # ------------------------------------------------------------------
    # Phase: analyze  (Phase 0)
    # ------------------------------------------------------------------

    def run_analyze(self) -> None:
        """
        Read SVD data from the manifest and stored S.npy files; produce:
          analysis/erank_vs_layer.png         – erank per layer, one line per projection
          analysis/spectrum_{key}.png         – singular value heatmap per projection
          analysis/erank_summary.json         – machine-readable erank table
        """
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            import matplotlib.colors as mcolors
        except ImportError:
            print("matplotlib not installed — skipping analyze phase.")
            return

        per_layer_projs = {
            key: rec
            for key, rec in self.manifest["projections"].items()
            if rec.get("mode") == "per_layer"
        }
        if not per_layer_projs:
            print("No per_layer projections found — analyze phase requires per_layer training.")
            return

        self.analysis_dir.mkdir(parents=True, exist_ok=True)

        # Collect erank-per-layer data
        erank_table: Dict[str, Dict[int, float]] = {}  # {proj_key: {layer: erank}}
        for key, rec in per_layer_projs.items():
            svd = rec.get("svd", {})
            erank_table[key] = {}
            for layer_str, svd_rec in svd.items():
                er = svd_rec.get("erank")
                if er is not None:
                    erank_table[key][int(layer_str)] = er

        # --- Plot: erank vs. layer ---
        fig, ax = plt.subplots(figsize=(10, 5))
        for key, layer_eranks in sorted(erank_table.items()):
            if not layer_eranks:
                continue
            xs = sorted(layer_eranks.keys())
            ys = [layer_eranks[x] for x in xs]
            ax.plot(xs, ys, marker="o", markersize=3, linewidth=1.5, label=key)

        ax.set_xlabel("Layer index")
        ax.set_ylabel("Effective rank")
        ax.set_title(f"Effective Rank vs. Layer — {self.llm_model}")
        ax.legend(fontsize=8, loc="upper left")
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        erank_plot_path = self.analysis_dir / "erank_vs_layer.png"
        fig.savefig(str(erank_plot_path), dpi=150)
        plt.close(fig)
        print(f"  Saved: {erank_plot_path}")

        # --- Plot: singular value spectrum heatmap per projection ---
        for key, rec in per_layer_projs.items():
            save_path = self.base_dir / rec["save_path"]
            layers_sorted = sorted(
                int(l) for l in rec.get("svd", {}).keys()
            )
            # Load S.npy for each layer
            spectra = []
            layers_loaded = []
            for layer_idx in layers_sorted:
                s_path = save_path / f"layer_{layer_idx:04d}" / "S.npy"
                if s_path.exists():
                    spectra.append(np.load(s_path).astype(np.float32))
                    layers_loaded.append(layer_idx)

            if not spectra:
                print(f"  [skip heatmap] no S.npy found for {key}")
                continue

            # Stack spectra — may differ in length if clip_dim != llm_dim; pad to max
            max_k = max(s.shape[0] for s in spectra)
            mat = np.zeros((len(spectra), max_k), dtype=np.float32)
            for i, s in enumerate(spectra):
                mat[i, : s.shape[0]] = s

            fig, ax = plt.subplots(figsize=(12, max(3, len(layers_loaded) * 0.3 + 1)))
            # log1p for readability across the wide dynamic range
            im = ax.imshow(
                np.log1p(mat),
                aspect="auto",
                interpolation="nearest",
                cmap="viridis",
                origin="upper",
            )
            ax.set_yticks(range(len(layers_loaded)))
            ax.set_yticklabels(layers_loaded, fontsize=7)
            ax.set_xlabel("Singular value index")
            ax.set_ylabel("Layer")
            ax.set_title(f"Singular Value Spectrum — {key}")
            fig.colorbar(im, ax=ax, label="log(1 + σ)")
            fig.tight_layout()
            heat_path = self.analysis_dir / f"spectrum_{key}.png"
            fig.savefig(str(heat_path), dpi=150)
            plt.close(fig)
            print(f"  Saved: {heat_path}")

            # --- Plot: singular value curves for representative layers ---
            # Pick ~8 evenly spaced layers so the plot doesn't get too crowded.
            n_layers = len(layers_loaded)
            if n_layers <= 8:
                selected_indices = list(range(n_layers))
            else:
                step = (n_layers - 1) / 7
                selected_indices = sorted(set(round(i * step) for i in range(8)))

            cmap_lines = plt.get_cmap("plasma")
            fig, ax = plt.subplots(figsize=(10, 5))
            for rank, i in enumerate(selected_indices):
                s = spectra[i]
                color = cmap_lines(rank / max(len(selected_indices) - 1, 1))
                ax.plot(
                    np.arange(len(s)),
                    s,
                    linewidth=1.2,
                    color=color,
                    label=f"layer {layers_loaded[i]}",
                )

            ax.set_yscale("log")
            ax.set_xlabel("Singular value index $k$")
            ax.set_ylabel("$\\sigma_k$ (log scale)")
            ax.set_title(f"Singular Value Spectra — {key}")
            ax.legend(fontsize=7, loc="upper right", ncol=2)
            ax.grid(True, alpha=0.3, which="both")
            fig.tight_layout()
            lines_path = self.analysis_dir / f"spectrum_lines_{key}.png"
            fig.savefig(str(lines_path), dpi=150)
            plt.close(fig)
            print(f"  Saved: {lines_path}")

        # --- erank_summary.json ---
        summary_path = self.analysis_dir / "erank_summary.json"
        # Convert int keys to str for JSON serialisation
        summary = {
            k: {str(l): er for l, er in v.items()}
            for k, v in erank_table.items()
        }
        with open(summary_path, "w") as f:
            json.dump(summary, f, indent=2)
        print(f"  Saved: {summary_path}")
        print("Analyze phase complete.")

    # ------------------------------------------------------------------
    # Phase: dashboard
    # ------------------------------------------------------------------

    def run_dashboard(self) -> None:
        """Write a self-contained HTML dashboard."""
        import base64

        def _b64(path: Path) -> str:
            if path.exists():
                return base64.b64encode(path.read_bytes()).decode()
            return ""

        # Collect analysis plots to embed
        erank_plot = self.analysis_dir / "erank_vs_layer.png"
        spectrum_plots = sorted(self.analysis_dir.glob("spectrum_*.png")) if self.analysis_dir.exists() else []

        # Build image grid section (link to grid PNGs)
        grid_rows = ""
        for rel_str, rec in self.manifest["grids"].items():
            grid_rows += (
                f'<div class="grid-entry">'
                f'<p><b>{rec["projection"]}</b> | {rec["text_slug"]} | seed {rec["seed"]}</p>'
                f'<img src="{rel_str}" loading="lazy" style="max-width:100%">'
                f"</div>\n"
            )

        # Embed analysis plots as base64
        erank_img_tag = ""
        if erank_plot.exists():
            b64 = _b64(erank_plot)
            erank_img_tag = f'<img src="data:image/png;base64,{b64}" style="max-width:100%">'

        spectrum_tags = ""
        for p in spectrum_plots:
            b64 = _b64(p)
            spectrum_tags += (
                f'<h3>{p.stem}</h3>'
                f'<img src="data:image/png;base64,{b64}" style="max-width:100%">\n'
            )

        # Projection summary table
        proj_rows = ""
        for key, rec in self.manifest["projections"].items():
            mode = rec.get("mode", "?")
            alpha = rec.get("alpha", "?")
            n_layers = len(rec.get("svd", {}))
            # Average erank across layers for per_layer projections
            eranks = [v["erank"] for v in rec.get("svd", {}).values() if v.get("erank") is not None]
            avg_erank = f"{sum(eranks)/len(eranks):.1f}" if eranks else "—"
            proj_rows += (
                f"<tr><td>{key}</td><td>{mode}</td><td>{alpha}</td>"
                f"<td>{n_layers}</td><td>{avg_erank}</td></tr>\n"
            )

        manifest_json = json.dumps(self.manifest, indent=2)
        n_images = len(self.manifest["images"])
        n_grids = len(self.manifest["grids"])
        n_projs = len(self.manifest["projections"])

        html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>Diffusion Microscope — Experiment Dashboard</title>
<style>
  body {{ font-family: system-ui, sans-serif; margin: 0; padding: 1rem 2rem;
         background: #111; color: #e0e0e0; }}
  h1 {{ color: #fff; }}
  h2 {{ color: #ccc; border-bottom: 1px solid #444; padding-bottom: 0.3rem; }}
  h3 {{ color: #aaa; }}
  a {{ color: #7af; }}
  table {{ border-collapse: collapse; width: 100%; margin-bottom: 1rem; }}
  th, td {{ border: 1px solid #444; padding: 0.4rem 0.7rem; text-align: left; }}
  th {{ background: #222; }}
  .grid-entry {{ margin-bottom: 1.5rem; }}
  details summary {{ cursor: pointer; color: #7af; }}
  pre {{ background: #1a1a1a; padding: 1rem; overflow: auto; font-size: 0.8rem;
         border: 1px solid #333; max-height: 400px; }}
</style>
</head>
<body>
<h1>Diffusion Microscope — Experiment Dashboard</h1>
<p>LLM: <b>{self.llm_model}</b> &nbsp;|&nbsp;
   SD: <b>{self.sd_model}</b> &nbsp;|&nbsp;
   Projections: <b>{n_projs}</b> &nbsp;|&nbsp;
   Images: <b>{n_images}</b> &nbsp;|&nbsp;
   Grids: <b>{n_grids}</b></p>

<h2>Phase 0 — Effective Rank Analysis</h2>
{erank_img_tag if erank_img_tag else '<p><i>Run the <code>analyze</code> phase to generate plots.</i></p>'}
{spectrum_tags}

<h2>Projections</h2>
<table>
<tr><th>Key</th><th>Mode</th><th>Alpha</th><th>Layers</th><th>Avg Erank</th></tr>
{proj_rows if proj_rows else '<tr><td colspan="5"><i>No projections yet — run the <code>train</code> phase.</i></td></tr>'}
</table>

<h2>Image Grids</h2>
{grid_rows if grid_rows else '<p><i>No grids yet — run the <code>generate</code> and <code>grids</code> phases.</i></p>'}

<h2>Manifest</h2>
<details>
<summary>Show full manifest JSON</summary>
<pre>{manifest_json}</pre>
</details>

</body>
</html>
"""
        out = self.base_dir / "dashboard.html"
        out.write_text(html, encoding="utf-8")
        print(f"Dashboard written: {out}")

    # ------------------------------------------------------------------
    # Orchestration
    # ------------------------------------------------------------------

    def run(
        self,
        training_texts: List[str],
        phases: List[str],
    ) -> Dict:
        """Run the requested phases in order and return the final manifest."""
        if "all" in phases:
            phases = list(self.PHASE_ORDER)

        # Snapshot config
        cfg_dir = self.base_dir / "configs"
        cfg_dir.mkdir(parents=True, exist_ok=True)
        ts = int(time.time())
        with open(cfg_dir / f"config_{ts}.json", "w") as f:
            json.dump(self.config, f, indent=2)

        if "train" in phases:
            print("\n=== Phase: train ===")
            self.run_train(training_texts)

        if "generate" in phases:
            print("\n=== Phase: generate ===")
            self.run_generate()

        if "grids" in phases:
            print("\n=== Phase: grids ===")
            self.run_grids()

        if "analyze" in phases:
            print("\n=== Phase: analyze ===")
            self.run_analyze()

        if "dashboard" in phases:
            print("\n=== Phase: dashboard ===")
            self.run_dashboard()

        return {
            "manifest": self.manifest,
            "base_dir": str(self.base_dir),
        }
