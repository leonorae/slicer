"""
Live inference: load LLM + SD (lazily, cached), extract activations for an
arbitrary probe text, project per layer, stream images back via SSE.
"""

from __future__ import annotations

import asyncio
import base64
import io
import json
import logging
from pathlib import Path
from typing import AsyncIterator

import numpy as np

log = logging.getLogger(__name__)

# ── Module-level model cache (one set per process) ─────────────────────────
_cache: dict = {
    "llm_name": None,
    "llm": None,
    "tokenizer": None,
    "sd_id": None,
    "microscope": None,
}


# ── Helpers ────────────────────────────────────────────────────────────────

def _get_device() -> str:
    try:
        import torch
        if torch.cuda.is_available():
            return "cuda"
    except ImportError:
        pass
    return "cpu"


def _load_llm(name: str, device: str):
    from transformers import AutoModelForCausalLM, AutoTokenizer  # type: ignore
    log.info("Loading LLM: %s", name)
    tok = AutoTokenizer.from_pretrained(name)
    model = AutoModelForCausalLM.from_pretrained(name, output_hidden_states=True)
    model = model.to(device).eval()
    return model, tok


def _load_sd(sd_id: str, device: str):
    import torch
    from diffusion_microscope.generator import DiffusionMicroscope
    dtype = "float16" if device != "cpu" else "float32"
    log.info("Loading SD: %s", sd_id)
    m = DiffusionMicroscope(sd_id, device=device, dtype=dtype)
    m._load_pipeline()
    return m


def _image_to_b64(img) -> str:
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()


def _sse(event_type: str, data: dict) -> str:
    return f"event: {event_type}\ndata: {json.dumps(data)}\n\n"


# ── Main streaming generator ───────────────────────────────────────────────

async def stream_layers(
    results_dir: str,
    proj_key: str,
    text: str,
    cfg: float,
    seed: int,
    layers_spec: str = "all",
    num_inference_steps: int = 30,
) -> AsyncIterator[str]:
    """
    Async generator yielding SSE events:

        status           {"message": str}
        activations_ready{"layers": [int, ...]}
        layer_complete   {"layer": int, "done": int, "total": int,
                          "image_b64": str}
        complete         {"total": int}
        error            {"message": str}
    """
    loop = asyncio.get_event_loop()

    try:
        # ── Read manifest to discover model IDs ───────────────────────────
        mpath = Path(results_dir) / "manifest.json"
        if not mpath.exists():
            yield _sse("error", {"message": f"manifest.json not found in {results_dir}"})
            return

        manifest = json.loads(mpath.read_text())
        models_cfg = manifest.get("config", {}).get("models", {})
        llm_name = models_cfg.get("llm", "gpt2")
        sd_id    = models_cfg.get("sd", "sd-legacy/stable-diffusion-v1-5")
        device   = _get_device()

        # ── LLM ───────────────────────────────────────────────────────────
        if _cache["llm_name"] != llm_name:
            yield _sse("status", {"message": f"Loading LLM: {llm_name} …"})
            llm, tok = await loop.run_in_executor(None, _load_llm, llm_name, device)
            _cache.update({"llm": llm, "tokenizer": tok, "llm_name": llm_name})

        yield _sse("status", {"message": "Extracting activations …"})

        def _extract():
            from diffusion_microscope.clip_bridge import extract_llm_last_token_activations
            return extract_llm_last_token_activations(
                [text], _cache["llm"], _cache["tokenizer"], device=device
            )

        # acts: {layer_idx: (1, hidden_dim)}
        acts: dict[int, np.ndarray] = await loop.run_in_executor(None, _extract)
        yield _sse("activations_ready", {"layers": sorted(acts.keys())})

        # ── Projection ────────────────────────────────────────────────────
        yield _sse("status", {"message": f"Loading projection: {proj_key} …"})
        proj_path = Path(results_dir) / "projections" / proj_key

        def _load_proj():
            from diffusion_microscope.projection import LayerProjectionSet, LinearProjection
            if LayerProjectionSet.is_saved_at(str(proj_path)):
                return LayerProjectionSet.load(str(proj_path))
            return LinearProjection.load(str(proj_path))

        projection = await loop.run_in_executor(None, _load_proj)

        # ── Determine layers ──────────────────────────────────────────────
        if layers_spec == "all":
            layers = (
                projection.layers() if hasattr(projection, "layers")
                else sorted(acts.keys())
            )
        else:
            layers = [int(x) for x in layers_spec.split(",") if x.strip()]
        layers = [l for l in layers if l in acts]

        # ── SD model ──────────────────────────────────────────────────────
        if _cache["sd_id"] != sd_id:
            yield _sse("status", {"message": f"Loading SD: {sd_id} …"})
            mic = await loop.run_in_executor(None, _load_sd, sd_id, device)
            _cache.update({"microscope": mic, "sd_id": sd_id})

        microscope = _cache["microscope"]
        per_layer  = hasattr(projection, "projections")
        total      = len(layers)

        # ── Generate ──────────────────────────────────────────────────────
        for i, layer_idx in enumerate(layers):
            yield _sse("status", {"message": f"Generating layer {layer_idx} ({i+1}/{total}) …"})

            act = acts[layer_idx]  # (1, hidden_dim)

            def _gen(li=layer_idx, a=act):
                clip_vec = (
                    projection.transform(li, a) if per_layer
                    else projection.transform(a)
                )                          # (1, clip_dim) or (clip_dim,)
                if clip_vec.ndim == 2:
                    clip_vec = clip_vec[0]
                return microscope.generate_from_vector(
                    clip_vec,
                    guidance_scale=cfg,
                    seed=seed,
                    num_inference_steps=num_inference_steps,
                )

            img = await loop.run_in_executor(None, _gen)
            yield _sse("layer_complete", {
                "layer": layer_idx,
                "done":  i + 1,
                "total": total,
                "image_b64": _image_to_b64(img),
            })

        yield _sse("complete", {"total": total})

    except Exception as exc:
        log.exception("Inference error")
        yield _sse("error", {"message": f"{type(exc).__name__}: {exc}"})
