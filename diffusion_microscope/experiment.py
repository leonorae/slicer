"""
Batch experiment runner for the Diffusion Microscope.

Reads a JSON config file and systematically executes five phases:
  1. train     – train one projection per (type, alpha) combination
  2. generate  – generate images for every (projection × probe × layer × CFG × seed)
  3. grids     – compose per-(projection, probe, seed) grid images
  4. metrics   – LPIPS between consecutive layers, image variance
  5. (manifest always updated after each phase)

Checkpointing: the manifest.json records every trained projection and generated
image.  Re-running any phase skips already-completed work.

Config projection type names → pipeline modes:
  "single_final_layer"  →  single_layer   (one map, trained on the last layer)
  "mixed_layer"         →  mixed_layer    (one map, all layers pooled)
  "per_layer"           →  per_layer      (one map per layer)
"""

from __future__ import annotations

import json
import re
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np


# ---------------------------------------------------------------------------
# Helpers
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


def _proj_key(proj_type: str, alpha) -> str:
    return f"{proj_type}_alpha{alpha:g}"


def _to_tensor(arr: np.ndarray, device: str = "cpu"):
    """Convert (H, W, 3) float32 [0,1] array to LPIPS-compatible tensor."""
    import torch
    t = torch.tensor(arr).permute(2, 0, 1).unsqueeze(0) * 2 - 1  # [-1, 1]
    return t.to(device)


# ---------------------------------------------------------------------------
# ExperimentRunner
# ---------------------------------------------------------------------------

class ExperimentRunner:
    """
    Reads a config dict and runs the full parameter-grid experiment.

    Parameters
    ----------
    config          : experiment configuration dict (see experiment_config.example.json)
    llm_model       : HuggingFace LLM model name
    sd_model        : HuggingFace SD model ID
    clip_model      : OpenCLIP architecture
    clip_pretrained : OpenCLIP weights tag
    device          : 'cpu' or 'cuda'
    cache_dir       : directory for caching LLM training-data activations
    """

    def __init__(
        self,
        config: dict,
        llm_model: str = "gpt2",
        sd_model: str = "sd-legacy/stable-diffusion-v1-5",
        clip_model: str = "ViT-L-14",
        clip_pretrained: str = "openai",
        device: str = "cpu",
        cache_dir: Optional[str] = None,
        write_sidecars: bool = False,
        track_lpips: bool = True,
    ):
        self.config = config
        self.llm_model = llm_model
        self.sd_model = sd_model
        self.clip_model = clip_model
        self.clip_pretrained = clip_pretrained
        self.device = device
        self.cache_dir = cache_dir
        self.write_sidecars = write_sidecars
        self.track_lpips = track_lpips

        out = config.get("output", {})
        self.base_dir = Path(out.get("base_dir", "./experiment_results"))
        self.image_fmt = out.get("image_format", "png")

        self.proj_dir = self.base_dir / "projections"
        self.grids_dir = self.base_dir / "grids" / "by_projection"
        self.by_text_dir = self.base_dir / "grids" / "by_text"
        self.configs_dir = self.base_dir / "configs"
        self.probe_cache_dir = self.base_dir / ".probe_cache"
        self.manifest_path = self.base_dir / "manifest.json"

        self._manifest: dict = self._load_manifest()

    # ------------------------------------------------------------------
    # Manifest
    # ------------------------------------------------------------------

    def _load_manifest(self) -> dict:
        if self.manifest_path.exists():
            with open(self.manifest_path) as f:
                return json.load(f)
        return {
            "projections": {},
            "images": {},
            "grids": {},
            "metrics_computed": False,
            "layers": None,
        }

    def _save_manifest(self):
        self.manifest_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.manifest_path, "w") as f:
            json.dump(self._manifest, f, indent=2, default=str)

    # ------------------------------------------------------------------
    # Probe activation cache
    # ------------------------------------------------------------------

    def _probe_cache_key(self, text: str, layer_idx: int) -> Path:
        return self.probe_cache_dir / _slug(text) / f"layer_{layer_idx:04d}.npy"

    def _save_probe_cache(
        self, probe_acts: Dict[str, Dict[int, np.ndarray]]
    ):
        for text, layer_map in probe_acts.items():
            for layer_idx, arr in layer_map.items():
                p = self._probe_cache_key(text, layer_idx)
                p.parent.mkdir(parents=True, exist_ok=True)
                np.save(p, arr)

    def _load_probe_cache(
        self, texts: List[str], layers: List[int]
    ) -> Optional[Dict[str, Dict[int, np.ndarray]]]:
        """Return cached probe activations if all (text, layer) pairs exist."""
        result: Dict[str, Dict[int, np.ndarray]] = {}
        for text in texts:
            result[text] = {}
            for layer_idx in layers:
                p = self._probe_cache_key(text, layer_idx)
                if not p.exists():
                    return None
                result[text][layer_idx] = np.load(p)
        return result

    # ------------------------------------------------------------------
    # Phase 1: Train projections
    # ------------------------------------------------------------------

    def train_projections(self, data: dict) -> dict:
        """
        Train all (projection_type × alpha) combinations.
        Already-saved projections are loaded from disk (checkpointing).

        Returns
        -------
        dict mapping projection_key → projection object
        """
        from .pipeline import MicroscopePipeline
        from .projection import LinearProjection, LayerProjectionSet

        proj_config = self.config.get("projections", {})
        types = proj_config.get("types", ["single_final_layer"])
        alphas = proj_config.get("alpha_values", [1.0])

        # Reuse a single pipeline instance for all training calls.
        pipeline = MicroscopePipeline(
            llm_model_name=self.llm_model,
            sd_model_id=self.sd_model,
            clip_model_name=self.clip_model,
            clip_pretrained=self.clip_pretrained,
            device=self.device,
        )

        projections: dict = {}

        for proj_type in types:
            if proj_type not in _PROJ_MODE_MAP:
                print(
                    f"[Experiment] Unknown projection type {proj_type!r}, skipping",
                    flush=True,
                )
                continue
            mode = _PROJ_MODE_MAP[proj_type]

            for alpha in alphas:
                key = _proj_key(proj_type, alpha)
                save_path = self.proj_dir / key

                # --- Load from cache ---
                if key in self._manifest["projections"] and save_path.exists():
                    print(
                        f"[Experiment] Loading cached projection: {key}", flush=True
                    )
                    if LayerProjectionSet.is_saved_at(str(save_path)):
                        proj = LayerProjectionSet.load(str(save_path))
                    else:
                        proj = LinearProjection.load(str(save_path))
                    projections[key] = proj
                    continue

                # --- Train ---
                print(f"\n[Experiment] Training: {key}", flush=True)
                alpha_f = float(alpha)

                if mode == "single_layer":
                    result = pipeline.train_projection(data, alpha=alpha_f)
                    proj = result["projection"]
                    val = result["validation"]
                elif mode == "mixed_layer":
                    result = pipeline.train_mixed_projection(data, alpha=alpha_f)
                    proj = result["projection"]
                    val = result["validation"]
                else:  # per_layer
                    result = pipeline.train_layer_projections(data, alpha=alpha_f)
                    proj = result["projection_set"]
                    val = result["validation"]

                proj.save(str(save_path))

                # Condense validation for manifest
                if isinstance(val, dict) and val and isinstance(
                    next(iter(val.values())), dict
                ):
                    # per_layer: {layer_idx: metrics_dict}
                    r2s = [m["projection_r2"] for m in val.values()]
                    val_summary = {
                        "r2_min": round(min(r2s), 5),
                        "r2_max": round(max(r2s), 5),
                        "r2_mean": round(sum(r2s) / len(r2s), 5),
                    }
                else:
                    val_summary = {
                        k: round(v, 5) if isinstance(v, float) else v
                        for k, v in val.items()
                    }

                self._manifest["projections"][key] = {
                    "type": proj_type,
                    "mode": mode,
                    "alpha": alpha,
                    "save_path": str(save_path),
                    "validation": val_summary,
                }
                self._save_manifest()
                projections[key] = proj

        return projections

    # ------------------------------------------------------------------
    # Phase 2a: Extract probe activations (with cache)
    # ------------------------------------------------------------------

    def extract_probe_activations(
        self,
        texts: List[str],
        layers: List[int],
    ) -> Dict[str, Dict[int, np.ndarray]]:
        """
        Extract last-token LLM activations for each text at each layer.

        Returns {text: {layer_idx: (hidden_dim,)}} — one vector per (text, layer).
        Results are saved to the probe cache for future runs.
        """
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        from .clip_bridge import extract_llm_last_token_activations

        if not texts:
            return {}

        # Check cache first
        cached = self._load_probe_cache(texts, layers)
        if cached is not None:
            print(
                f"[Experiment] Loaded probe activations from cache "
                f"({len(texts)} texts × {len(layers)} layers)",
                flush=True,
            )
            return cached

        print(
            f"[Experiment] Extracting probe activations: "
            f"{len(texts)} texts × {len(layers)} layers …",
            flush=True,
        )

        tokenizer = AutoTokenizer.from_pretrained(self.llm_model)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        model = AutoModelForCausalLM.from_pretrained(
            self.llm_model, torch_dtype=torch.float32
        )

        # raw: {layer_idx: (n_texts, hidden_dim)}
        raw = extract_llm_last_token_activations(
            texts, model, tokenizer,
            layers=layers, device=self.device,
        )

        del model
        if self.device.startswith("cuda"):
            torch.cuda.empty_cache()

        # Restructure: {text: {layer_idx: (hidden_dim,)}}
        result: Dict[str, Dict[int, np.ndarray]] = {t: {} for t in texts}
        for layer_idx, arr in raw.items():
            for i, text in enumerate(texts):
                result[text][layer_idx] = arr[i]

        self._save_probe_cache(result)
        return result

    # ------------------------------------------------------------------
    # Phase 2b: Generate images
    # ------------------------------------------------------------------

    def generate_images(
        self,
        projections: dict,
        probe_activations: Dict[str, Dict[int, np.ndarray]],
        interp_activations: Optional[Dict[str, Tuple]] = None,
    ):
        """
        Generate all (projection × probe × layer × CFG × seed) images.
        Already-generated images (tracked in manifest) are skipped.

        Inline LPIPS (if self.track_lpips): compares each layer image to
        the previous layer image with the same (proj, text, cfg, seed).
        """
        from .generator import DiffusionMicroscope

        cfg_values = [float(c) for c in self.config.get("cfg_values", [7.0])]
        seeds = [int(s) for s in self.config.get("seeds", [42])]

        scope = DiffusionMicroscope(
            sd_model_id=self.sd_model,
            device=self.device,
            dtype="float16" if "cuda" in self.device else "float32",
        )

        # Set up inline LPIPS if requested
        lpips_fn = None
        if self.track_lpips:
            try:
                import torch
                import lpips as lpips_lib
                lpips_fn = lpips_lib.LPIPS(net="alex").eval()
                if "cuda" in self.device:
                    lpips_fn = lpips_fn.to(self.device)
            except ImportError:
                print(
                    "[Experiment] lpips not available; skipping LPIPS tracking",
                    flush=True,
                )

        total_todo = (
            len(projections)
            * sum(len(la) for la in probe_activations.values())
            * len(cfg_values)
            * len(seeds)
        )
        done = 0

        for proj_key, proj in projections.items():
            is_per_layer = hasattr(proj, "projections")
            proj_grids_dir = self.grids_dir / proj_key

            for text, layer_acts in probe_activations.items():
                text_slug = _slug(text)
                text_dir = proj_grids_dir / text_slug / "per_layer"
                text_dir.mkdir(parents=True, exist_ok=True)

                # prev_img_arr[(cfg, seed)] → float32 np array of prev layer img
                prev_img_arr: dict = {}

                for layer_idx, act in sorted(layer_acts.items()):
                    if is_per_layer and layer_idx not in proj:
                        continue

                    clip_vec = (
                        proj.transform(layer_idx, act)
                        if is_per_layer
                        else proj.transform(act)
                    )

                    for cfg in cfg_values:
                        for seed in seeds:
                            done += 1
                            fname = (
                                f"L{layer_idx:03d}_CFG{cfg:.1f}"
                                f"_seed{seed}.{self.image_fmt}"
                            )
                            img_path = text_dir / fname
                            mkey = str(img_path.relative_to(self.base_dir))

                            if mkey in self._manifest["images"]:
                                # Still update prev_img_arr for LPIPS continuity
                                if lpips_fn is not None and img_path.exists():
                                    from PIL import Image as _PILImg
                                    _a = np.array(
                                        _PILImg.open(img_path).convert("RGB")
                                    ).astype(np.float32) / 255.0
                                    prev_img_arr[(cfg, seed)] = _a
                                continue

                            print(
                                f"[SD] ({done}/{total_todo}) {proj_key} / "
                                f"{text_slug} / L{layer_idx} CFG{cfg:.1f} "
                                f"seed{seed}",
                                flush=True,
                            )

                            img = scope.generate_from_vector(
                                clip_vec, guidance_scale=cfg, seed=seed,
                            )
                            img.save(img_path)

                            meta = self._image_meta(
                                mkey, proj_key, text, layer_idx, cfg, seed,
                            )

                            # Inline LPIPS
                            if lpips_fn is not None:
                                import torch
                                cur_arr = (
                                    np.array(img.convert("RGB")).astype(np.float32)
                                    / 255.0
                                )
                                prev_arr = prev_img_arr.get((cfg, seed))
                                if prev_arr is not None:
                                    with torch.no_grad():
                                        d = lpips_fn(
                                            _to_tensor(prev_arr, self.device),
                                            _to_tensor(cur_arr, self.device),
                                        )
                                    meta["metrics"]["lpips_to_prev_layer"] = round(
                                        float(d.item()), 6
                                    )
                                prev_img_arr[(cfg, seed)] = cur_arr

                            if self.write_sidecars:
                                with open(img_path.with_suffix(".json"), "w") as f:
                                    json.dump(meta, f, indent=2)

                            self._manifest["images"][mkey] = meta
                            if len(self._manifest["images"]) % 25 == 0:
                                self._save_manifest()

        # Interpolations
        if interp_activations:
            self._generate_interpolations(
                scope, projections, interp_activations, cfg_values, seeds
            )

        self._save_manifest()

    def _image_meta(
        self,
        mkey: str,
        proj_key: str,
        text: str,
        layer_idx: int,
        cfg: float,
        seed: int,
    ) -> dict:
        proj_info = self._manifest["projections"].get(proj_key, {})
        return {
            "image_path": mkey,
            "experiment_config": {
                "projection_type": proj_info.get("type", proj_key),
                "alpha": proj_info.get("alpha"),
            },
            "generation_params": {
                "probe_text": text,
                "layer": layer_idx,
                "cfg": cfg,
                "seed": seed,
                "sd_model": self.sd_model,
                "llm_model": self.llm_model,
            },
            "metrics": {},
        }

    def _generate_interpolations(
        self,
        scope,
        projections: dict,
        interp_activations: Dict[str, Tuple],
        cfg_values: List[float],
        seeds: List[int],
    ):
        for _, (from_acts, to_acts, n_steps, from_text, to_text) in (
            interp_activations.items()
        ):
            alphas = [i / max(n_steps - 1, 1) for i in range(n_steps)]
            from_slug = _slug(from_text)
            to_slug = _slug(to_text)

            for proj_key, proj in projections.items():
                is_per_layer = hasattr(proj, "projections")
                interp_dir = (
                    self.grids_dir
                    / proj_key
                    / f"interp_{from_slug}_to_{to_slug}"
                    / "per_layer"
                )
                interp_dir.mkdir(parents=True, exist_ok=True)

                for layer_idx in sorted(from_acts.keys()):
                    if is_per_layer and layer_idx not in proj:
                        continue

                    for step_i, alpha in enumerate(alphas):
                        interp_act = (
                            (1 - alpha) * from_acts[layer_idx]
                            + alpha * to_acts[layer_idx]
                        )
                        clip_vec = (
                            proj.transform(layer_idx, interp_act)
                            if is_per_layer
                            else proj.transform(interp_act)
                        )

                        for cfg in cfg_values:
                            for seed in seeds:
                                fname = (
                                    f"step{step_i:02d}_a{alpha:.2f}"
                                    f"_L{layer_idx:03d}_CFG{cfg:.1f}"
                                    f"_seed{seed}.{self.image_fmt}"
                                )
                                img_path = interp_dir / fname
                                mkey = str(img_path.relative_to(self.base_dir))

                                if mkey in self._manifest["images"]:
                                    continue

                                print(
                                    f"[SD] interp {from_slug}→{to_slug} "
                                    f"step={step_i} L{layer_idx} "
                                    f"CFG{cfg:.1f} seed{seed}",
                                    flush=True,
                                )

                                img = scope.generate_from_vector(
                                    clip_vec, guidance_scale=cfg, seed=seed,
                                )
                                img.save(img_path)

                                meta = {
                                    "image_path": mkey,
                                    "experiment_config": {
                                        "projection_type": self._manifest[
                                            "projections"
                                        ]
                                        .get(proj_key, {})
                                        .get("type", proj_key),
                                        "alpha": self._manifest["projections"]
                                        .get(proj_key, {})
                                        .get("alpha"),
                                    },
                                    "generation_params": {
                                        "interp_from": from_text,
                                        "interp_to": to_text,
                                        "interp_step": step_i,
                                        "alpha": round(alpha, 4),
                                        "layer": layer_idx,
                                        "cfg": cfg,
                                        "seed": seed,
                                        "sd_model": self.sd_model,
                                        "llm_model": self.llm_model,
                                    },
                                    "metrics": {},
                                }
                                if self.write_sidecars:
                                    with open(
                                        img_path.with_suffix(".json"), "w"
                                    ) as f:
                                        json.dump(meta, f, indent=2)
                                self._manifest["images"][mkey] = meta

    # ------------------------------------------------------------------
    # Phase 3: Compose grids
    # ------------------------------------------------------------------

    def compose_grids(self, projections: dict, probe_texts: List[str]):
        """
        For each (projection, probe_text, seed), compose a grid:
          rows    = layers (ascending)
          columns = CFG values (ascending)
        Saves: grids/by_projection/{proj_key}/{text_slug}/grid_seed{N}.png
        Also creates: grids/by_text/{text_slug}/{proj_key}_seed{N}.png (symlinks)
        """
        try:
            from PIL import Image, ImageDraw
        except ImportError:
            print(
                "[Experiment] Pillow not available, skipping grid composition",
                flush=True,
            )
            return

        cfg_values = [float(c) for c in self.config.get("cfg_values", [7.0])]
        seeds = [int(s) for s in self.config.get("seeds", [42])]

        for proj_key in projections:
            for text in probe_texts:
                text_slug = _slug(text)
                per_layer_dir = (
                    self.grids_dir / proj_key / text_slug / "per_layer"
                )
                if not per_layer_dir.exists():
                    continue

                for seed in seeds:
                    grid_path = (
                        self.grids_dir
                        / proj_key
                        / text_slug
                        / f"grid_seed{seed}.png"
                    )
                    gkey = str(grid_path.relative_to(self.base_dir))
                    if gkey in self._manifest["grids"]:
                        continue

                    # Discover which layers have images for this seed
                    pattern = f"L*_CFG{cfg_values[0]:.1f}_seed{seed}.{self.image_fmt}"
                    layers = sorted(
                        int(p.stem.split("_")[0][1:])
                        for p in per_layer_dir.glob(pattern)
                    )
                    if not layers:
                        continue

                    cell_w, cell_h = 256, 256
                    header_h, row_label_w = 30, 60
                    n_rows = len(layers)
                    n_cols = len(cfg_values)
                    grid_w = row_label_w + n_cols * cell_w
                    grid_h = header_h + n_rows * cell_h

                    grid = Image.new("RGB", (grid_w, grid_h), "white")
                    draw = ImageDraw.Draw(grid)

                    for j, cfg in enumerate(cfg_values):
                        x = row_label_w + j * cell_w + cell_w // 2
                        draw.text((x, 5), f"CFG {cfg:.1f}", fill="black", anchor="mt")

                    for i, layer in enumerate(layers):
                        y_top = header_h + i * cell_h
                        draw.text(
                            (row_label_w // 2, y_top + cell_h // 2),
                            f"L{layer}",
                            fill="black",
                            anchor="mm",
                        )
                        for j, cfg in enumerate(cfg_values):
                            p = (
                                per_layer_dir
                                / f"L{layer:03d}_CFG{cfg:.1f}"
                                f"_seed{seed}.{self.image_fmt}"
                            )
                            if not p.exists():
                                continue
                            img = Image.open(p).resize((cell_w, cell_h))
                            grid.paste(img, (row_label_w + j * cell_w, y_top))

                    grid.save(grid_path)
                    print(f"[Experiment] Grid → {grid_path}", flush=True)

                    self._manifest["grids"][gkey] = {
                        "projection": proj_key,
                        "text": text,
                        "seed": seed,
                        "path": str(grid_path),
                    }

                    # by_text symlink
                    by_text_text_dir = self.by_text_dir / text_slug
                    by_text_text_dir.mkdir(parents=True, exist_ok=True)
                    link = by_text_text_dir / f"{proj_key}_seed{seed}.png"
                    if not link.exists():
                        try:
                            link.symlink_to(grid_path.resolve())
                        except (OSError, NotImplementedError):
                            pass  # symlinks not supported on this platform

        self._save_manifest()

    # ------------------------------------------------------------------
    # Phase 4: Compute metrics
    # ------------------------------------------------------------------

    def compute_metrics(self):
        """
        For each image: compute pixel variance.
        For consecutive-layer image pairs (same proj/text/cfg/seed):
        compute LPIPS distance in both directions.
        """
        try:
            import torch
            import lpips as lpips_lib

            lpips_fn = lpips_lib.LPIPS(net="alex").eval()
            if "cuda" in self.device:
                lpips_fn = lpips_fn.to(self.device)
            has_lpips = True
        except ImportError:
            has_lpips = False
            print(
                "[Experiment] lpips not available; computing image variance only",
                flush=True,
            )

        from PIL import Image as PILImage

        # Group images by (proj_type, probe_text, cfg, seed) to find layer sequences
        groups: Dict[tuple, List[tuple]] = {}
        for mkey, meta in self._manifest["images"].items():
            gp = meta.get("generation_params", {})
            if "layer" not in gp:
                continue
            group_id = (
                meta.get("experiment_config", {}).get("projection_type", ""),
                gp.get("probe_text", ""),
                gp.get("cfg", 0),
                gp.get("seed", 0),
            )
            groups.setdefault(group_id, []).append((gp["layer"], mkey))

        updated = 0
        for group_id, items in groups.items():
            items_sorted = sorted(items)  # sort by layer
            # Load all images in this group
            imgs: List[Optional[np.ndarray]] = []
            for _, mkey in items_sorted:
                p = self.base_dir / mkey
                if not p.exists():
                    imgs.append(None)
                    continue
                im = PILImage.open(p).convert("RGB")
                imgs.append(np.array(im).astype(np.float32) / 255.0)

            for i, (layer_idx, mkey) in enumerate(items_sorted):
                if imgs[i] is None:
                    continue
                m = self._manifest["images"][mkey].setdefault("metrics", {})
                m["image_variance"] = round(float(imgs[i].var()), 6)

                if has_lpips:
                    import torch

                    if i > 0 and imgs[i - 1] is not None:
                        with torch.no_grad():
                            d = lpips_fn(
                                _to_tensor(imgs[i - 1], self.device),
                                _to_tensor(imgs[i], self.device),
                            )
                        m["lpips_to_prev_layer"] = round(float(d.item()), 6)

                    if i < len(items_sorted) - 1 and imgs[i + 1] is not None:
                        with torch.no_grad():
                            d = lpips_fn(
                                _to_tensor(imgs[i], self.device),
                                _to_tensor(imgs[i + 1], self.device),
                            )
                        m["lpips_to_next_layer"] = round(float(d.item()), 6)

                updated += 1

        self._manifest["metrics_computed"] = True
        self._save_manifest()
        print(f"[Experiment] Metrics computed for {updated} images", flush=True)

    # ------------------------------------------------------------------
    # Phase 5: Animations (layer-sweep GIFs)
    # ------------------------------------------------------------------

    def create_animations(self, projections: dict, probe_texts: List[str]):
        """
        Create layer-sweep GIF animations for each (proj, text, cfg, seed).

        Saves: grids/by_projection/{proj_key}/{text_slug}/anim_CFG{cfg}_seed{seed}.gif
        """
        try:
            from PIL import Image as PILImage
        except ImportError:
            print(
                "[Experiment] Pillow not available, skipping animations",
                flush=True,
            )
            return

        cfg_values = [float(c) for c in self.config.get("cfg_values", [7.0])]
        seeds = [int(s) for s in self.config.get("seeds", [42])]

        for proj_key in projections:
            for text in probe_texts:
                text_slug = _slug(text)
                per_layer_dir = self.grids_dir / proj_key / text_slug / "per_layer"
                if not per_layer_dir.exists():
                    continue

                for cfg in cfg_values:
                    for seed in seeds:
                        anim_path = (
                            self.grids_dir
                            / proj_key
                            / text_slug
                            / f"anim_CFG{cfg:.1f}_seed{seed}.gif"
                        )
                        akey = str(anim_path.relative_to(self.base_dir))
                        if akey in self._manifest.get("animations", {}):
                            continue

                        pattern = f"L*_CFG{cfg:.1f}_seed{seed}.{self.image_fmt}"
                        layer_files = sorted(
                            per_layer_dir.glob(pattern),
                            key=lambda p: int(p.stem.split("_")[0][1:]),
                        )
                        if len(layer_files) < 2:
                            continue

                        frames = [
                            PILImage.open(p).convert("RGB").resize((256, 256))
                            for p in layer_files
                        ]
                        frames[0].save(
                            anim_path,
                            save_all=True,
                            append_images=frames[1:],
                            duration=150,
                            loop=0,
                        )
                        print(f"[Experiment] Animation → {anim_path}", flush=True)

                        self._manifest.setdefault("animations", {})[akey] = {
                            "projection": proj_key,
                            "text": text,
                            "cfg": cfg,
                            "seed": seed,
                            "n_frames": len(frames),
                            "path": str(anim_path),
                        }

        self._save_manifest()

    # ------------------------------------------------------------------
    # Phase 6: Dashboard
    # ------------------------------------------------------------------

    def generate_dashboard(self):
        """
        Generate a self-contained HTML dashboard at base_dir/dashboard.html.

        Displays all generated images grouped by probe text, with controls
        to filter by projection type, CFG, seed, and layer range.
        """
        dashboard_path = self.base_dir / "dashboard.html"

        images_by_text: dict = {}
        for mkey, meta in self._manifest["images"].items():
            gp = meta.get("generation_params", {})
            text = gp.get("probe_text", "")
            if not text:
                continue
            images_by_text.setdefault(text, []).append(
                {
                    "path": mkey,
                    "proj": meta.get("experiment_config", {}).get(
                        "projection_type", ""
                    ),
                    "alpha": meta.get("experiment_config", {}).get("alpha"),
                    "layer": gp.get("layer"),
                    "cfg": gp.get("cfg"),
                    "seed": gp.get("seed"),
                    "lpips": meta.get("metrics", {}).get("lpips_to_prev_layer"),
                }
            )

        manifest_json = json.dumps(
            {
                "images_by_text": images_by_text,
                "projections": self._manifest.get("projections", {}),
                "animations": self._manifest.get("animations", {}),
            },
            indent=None,
        )

        html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>Diffusion Microscope – Results Dashboard</title>
<style>
  body {{ font-family: system-ui, sans-serif; margin: 0; padding: 12px; background: #111; color: #eee; }}
  h1 {{ font-size: 1.2rem; margin-bottom: 8px; }}
  #controls {{ display: flex; flex-wrap: wrap; gap: 10px; background: #1e1e1e; padding: 10px; border-radius: 6px; margin-bottom: 12px; }}
  #controls label {{ font-size: 0.85rem; display: flex; flex-direction: column; gap: 3px; }}
  select, input[type=range] {{ background: #333; color: #eee; border: 1px solid #555; border-radius: 4px; padding: 3px 6px; }}
  .section-title {{ font-size: 1rem; font-weight: bold; margin: 14px 0 6px; color: #adf; border-bottom: 1px solid #333; padding-bottom: 4px; }}
  .grid {{ display: flex; flex-wrap: wrap; gap: 6px; }}
  .card {{ background: #1e1e1e; border-radius: 5px; overflow: hidden; cursor: pointer; transition: transform .15s; width: 160px; }}
  .card:hover {{ transform: scale(1.04); }}
  .card img {{ width: 160px; height: 160px; object-fit: cover; display: block; }}
  .card .meta {{ font-size: 0.65rem; padding: 4px 6px; color: #aaa; }}
  .hidden {{ display: none !important; }}
  #lightbox {{ display: none; position: fixed; inset: 0; background: rgba(0,0,0,.85); z-index: 100; align-items: center; justify-content: center; }}
  #lightbox.open {{ display: flex; }}
  #lightbox img {{ max-width: 90vw; max-height: 90vh; border-radius: 6px; }}
  #lightbox .close {{ position: absolute; top: 18px; right: 24px; font-size: 2rem; cursor: pointer; color: #eee; }}
</style>
</head>
<body>
<h1>Diffusion Microscope — Results Dashboard</h1>
<div id="controls">
  <label>Projection<select id="f-proj"><option value="">All</option></select></label>
  <label>CFG<select id="f-cfg"><option value="">All</option></select></label>
  <label>Seed<select id="f-seed"><option value="">All</option></select></label>
  <label>Layer min<input type="range" id="f-lmin" min="0" max="100" value="0"> <span id="lmin-val">0</span></label>
  <label>Layer max<input type="range" id="f-lmax" min="0" max="100" value="100"> <span id="lmax-val">100</span></label>
</div>
<div id="gallery"></div>
<div id="lightbox"><span class="close" onclick="closeLB()">✕</span><img id="lb-img" src=""></div>
<script>
const DATA = {manifest_json};
const gallery = document.getElementById('gallery');

// Populate filter dropdowns
const projs = new Set(), cfgs = new Set(), seeds = new Set();
let layerMin = Infinity, layerMax = -Infinity;
Object.values(DATA.images_by_text).flat().forEach(im => {{
  if (im.proj) projs.add(im.proj);
  if (im.cfg != null) cfgs.add(im.cfg);
  if (im.seed != null) seeds.add(im.seed);
  if (im.layer != null) {{ layerMin = Math.min(layerMin, im.layer); layerMax = Math.max(layerMax, im.layer); }}
}});
[...projs].sort().forEach(v => document.getElementById('f-proj').add(new Option(v, v)));
[...cfgs].sort((a,b)=>a-b).forEach(v => document.getElementById('f-cfg').add(new Option(v, v)));
[...seeds].sort((a,b)=>a-b).forEach(v => document.getElementById('f-seed').add(new Option(v, v)));
if (layerMin < Infinity) {{
  ['f-lmin','f-lmax'].forEach(id => {{
    const el = document.getElementById(id);
    el.min = layerMin; el.max = layerMax;
    el.value = id === 'f-lmin' ? layerMin : layerMax;
  }});
  document.getElementById('lmin-val').textContent = layerMin;
  document.getElementById('lmax-val').textContent = layerMax;
}}

function getFilters() {{
  return {{
    proj: document.getElementById('f-proj').value,
    cfg: document.getElementById('f-cfg').value,
    seed: document.getElementById('f-seed').value,
    lmin: +document.getElementById('f-lmin').value,
    lmax: +document.getElementById('f-lmax').value,
  }};
}}

function render() {{
  const f = getFilters();
  gallery.innerHTML = '';
  Object.entries(DATA.images_by_text).forEach(([text, imgs]) => {{
    const filtered = imgs.filter(im =>
      (!f.proj || im.proj === f.proj) &&
      (!f.cfg  || String(im.cfg) === f.cfg) &&
      (!f.seed || String(im.seed) === f.seed) &&
      (im.layer == null || (im.layer >= f.lmin && im.layer <= f.lmax))
    );
    if (!filtered.length) return;
    const sec = document.createElement('div');
    sec.innerHTML = `<div class="section-title">${{text}}</div>`;
    const grid = document.createElement('div'); grid.className = 'grid';
    filtered.sort((a,b) => (a.layer??0)-(b.layer??0) || (a.cfg??0)-(b.cfg??0)).forEach(im => {{
      const card = document.createElement('div'); card.className = 'card';
      const lpipsStr = im.lpips != null ? ` | LPIPS ${{im.lpips.toFixed(3)}}` : '';
      card.innerHTML = `<img src="${{im.path}}" loading="lazy" onclick="openLB(this.src)">
        <div class="meta">L${{im.layer}} CFG${{im.cfg}} seed${{im.seed}}${{lpipsStr}}<br>${{im.proj}}</div>`;
      grid.appendChild(card);
    }});
    sec.appendChild(grid);
    gallery.appendChild(sec);
  }});
}}

['f-proj','f-cfg','f-seed'].forEach(id => document.getElementById(id).addEventListener('change', render));
document.getElementById('f-lmin').addEventListener('input', function() {{
  document.getElementById('lmin-val').textContent = this.value; render();
}});
document.getElementById('f-lmax').addEventListener('input', function() {{
  document.getElementById('lmax-val').textContent = this.value; render();
}});

function openLB(src) {{ document.getElementById('lb-img').src = src; document.getElementById('lightbox').classList.add('open'); }}
function closeLB() {{ document.getElementById('lightbox').classList.remove('open'); }}
document.getElementById('lightbox').addEventListener('click', e => {{ if (e.target.id === 'lightbox') closeLB(); }});

render();
</script>
</body>
</html>"""

        with open(dashboard_path, "w", encoding="utf-8") as f:
            f.write(html)

        print(f"[Experiment] Dashboard → {dashboard_path}", flush=True)
        return str(dashboard_path)

    # ------------------------------------------------------------------
    # Top-level run
    # ------------------------------------------------------------------

    def run(
        self,
        training_texts: List[str],
        phases: List[str] = ("train", "generate", "grids", "metrics"),
    ) -> dict:
        """
        Execute the requested phases in order.

        Parameters
        ----------
        training_texts : texts used to train projections
        phases         : ordered list of phases to execute; subset of
                         'train', 'generate', 'grids', 'metrics',
                         'animations', 'dashboard'
        """
        self.base_dir.mkdir(parents=True, exist_ok=True)

        # Snapshot the config
        ts = time.strftime("%Y%m%d_%H%M%S")
        self.configs_dir.mkdir(parents=True, exist_ok=True)
        with open(self.configs_dir / f"config_{ts}.json", "w") as f:
            json.dump(self.config, f, indent=2)

        # Collect probe texts by category
        probe_cfg = self.config.get("probe_texts", {})
        flat_probes: List[str] = []
        for cat in ("concrete", "abstract", "function", "novel"):
            flat_probes.extend(probe_cfg.get(cat, []))
        flat_probes = list(dict.fromkeys(flat_probes))  # deduplicate, keep order

        interp_pairs = probe_cfg.get("interpolations", [])
        interp_endpoint_texts = list(
            dict.fromkeys(
                t
                for pair in interp_pairs
                for t in (pair.get("from", ""), pair.get("to", ""))
                if t
            )
        )

        # All texts we need LLM activations for
        all_probe_texts = list(
            dict.fromkeys(flat_probes + interp_endpoint_texts)
        )

        # Layer list
        layers_cfg: Optional[List[int]] = self.config.get("layers")

        # ---------------------------------------------------------------
        # Phase: train
        # ---------------------------------------------------------------
        projections: dict = {}
        data: Optional[dict] = None

        if "train" in phases:
            from .pipeline import MicroscopePipeline
            import torch
            from transformers import AutoModelForCausalLM

            pipeline = MicroscopePipeline(
                llm_model_name=self.llm_model,
                sd_model_id=self.sd_model,
                clip_model_name=self.clip_model,
                clip_pretrained=self.clip_pretrained,
                device=self.device,
            )

            n_train = self.config.get("projections", {}).get(
                "training_data_size", 5000
            )
            training_texts = training_texts[:n_train]
            print(
                f"\n[Experiment] Phase: train — {len(training_texts)} training texts",
                flush=True,
            )
            data = pipeline.prepare_training_data(
                training_texts, cache_dir=self.cache_dir
            )

            # Resolve layers now that we know the model
            if layers_cfg is None:
                layers_cfg = sorted(data["llm_activations"].keys())

            self._manifest["layers"] = layers_cfg
            self._save_manifest()

            projections = self.train_projections(data)

        else:
            # Load projections from manifest
            from .projection import LinearProjection, LayerProjectionSet

            for key, info in self._manifest.get("projections", {}).items():
                sp = info.get("save_path", "")
                if not sp:
                    continue
                if LayerProjectionSet.is_saved_at(sp):
                    projections[key] = LayerProjectionSet.load(sp)
                elif Path(sp).exists():
                    projections[key] = LinearProjection.load(sp)

            if layers_cfg is None:
                layers_cfg = self._manifest.get("layers") or list(range(13))

        # ---------------------------------------------------------------
        # Phase: generate
        # ---------------------------------------------------------------
        if "generate" in phases:
            if not projections:
                print(
                    "[Experiment] No projections available — run 'train' first",
                    flush=True,
                )
            else:
                print(
                    f"\n[Experiment] Phase: generate — "
                    f"{len(projections)} projections × {len(flat_probes)} probes",
                    flush=True,
                )

                probe_acts = self.extract_probe_activations(
                    all_probe_texts, layers_cfg
                )

                flat_probe_acts = {
                    t: probe_acts[t] for t in flat_probes if t in probe_acts
                }

                # Build interpolation activation dicts
                interp_acts: Dict[str, Tuple] = {}
                for pair in interp_pairs:
                    from_t = pair.get("from", "")
                    to_t = pair.get("to", "")
                    n_steps = int(pair.get("steps", 5))
                    if from_t in probe_acts and to_t in probe_acts:
                        ikey = f"{from_t}____{to_t}"
                        interp_acts[ikey] = (
                            probe_acts[from_t],
                            probe_acts[to_t],
                            n_steps,
                            from_t,
                            to_t,
                        )

                self.generate_images(
                    projections,
                    flat_probe_acts,
                    interp_acts or None,
                )

        # ---------------------------------------------------------------
        # Phase: grids
        # ---------------------------------------------------------------
        if "grids" in phases:
            if not projections:
                print(
                    "[Experiment] No projections available for grid composition",
                    flush=True,
                )
            else:
                print(f"\n[Experiment] Phase: grids", flush=True)
                self.compose_grids(projections, flat_probes)

        # ---------------------------------------------------------------
        # Phase: metrics
        # ---------------------------------------------------------------
        if "metrics" in phases:
            print(f"\n[Experiment] Phase: metrics", flush=True)
            self.compute_metrics()

        # ---------------------------------------------------------------
        # Phase: animations
        # ---------------------------------------------------------------
        if "animations" in phases:
            if not projections:
                print(
                    "[Experiment] No projections available for animations",
                    flush=True,
                )
            else:
                print(f"\n[Experiment] Phase: animations", flush=True)
                self.create_animations(projections, flat_probes)

        # ---------------------------------------------------------------
        # Phase: dashboard
        # ---------------------------------------------------------------
        if "dashboard" in phases:
            print(f"\n[Experiment] Phase: dashboard", flush=True)
            self.generate_dashboard()

        self._save_manifest()
        print(f"\n[Experiment] Complete → {self.base_dir}", flush=True)
        return {"base_dir": str(self.base_dir), "manifest": self._manifest}
