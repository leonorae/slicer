"""FastAPI application for the Diffusion Microscope live dashboard."""

from __future__ import annotations

import json
import os
from pathlib import Path

from fastapi import FastAPI, Query
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles

app = FastAPI(title="Diffusion Microscope Live")

_static = Path(__file__).parent / "static"
app.mount("/static", StaticFiles(directory=str(_static)), name="static")

_DEFAULT_RESULTS = os.environ.get("DM_RESULTS_DIR", "./experiment_results")


@app.get("/")
async def root():
    return FileResponse(_static / "index.html")


@app.get("/api/defaults")
async def get_defaults():
    """Return server-side defaults so the frontend can pre-populate fields."""
    return {"results_dir": _DEFAULT_RESULTS}


@app.get("/api/manifest")
async def get_manifest(results_dir: str = Query(...)):
    path = Path(results_dir) / "manifest.json"
    if not path.exists():
        return {"error": f"No manifest.json found in: {results_dir}"}
    try:
        return json.loads(path.read_text())
    except Exception as exc:
        return {"error": str(exc)}


@app.get("/api/image")
async def get_image(
    results_dir: str = Query(...),
    rel_path: str = Query(...),
):
    path = Path(results_dir) / rel_path
    if not path.exists():
        return {"error": f"Image not found: {rel_path}"}
    return FileResponse(str(path), media_type="image/png")


@app.get("/api/stream")
async def stream_inference(
    results_dir: str = Query(...),
    proj_key: str = Query(...),
    text: str = Query(...),
    cfg: float = Query(25.0),
    seed: int = Query(42),
    layers: str = Query("all"),
    steps: int = Query(30),
):
    """SSE endpoint — streams layer images as they are generated."""
    from .inference import stream_layers

    async def _generate():
        async for event in stream_layers(
            results_dir=results_dir,
            proj_key=proj_key,
            text=text,
            cfg=cfg,
            seed=seed,
            layers_spec=layers,
            num_inference_steps=steps,
        ):
            yield event

    return StreamingResponse(
        _generate(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )
