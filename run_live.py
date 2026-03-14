#!/usr/bin/env python3
"""
Diffusion Microscope — live interactive dashboard.

Starts a local web server with two modes:
  Browse  — explore existing experiment results interactively
            (layer slideshow, grid view, compare-across-alphas, per-layer metrics)
  Live    — enter any probe text, select a pre-trained projection, stream
            layer images as they are generated

Usage
-----
    python run_live.py [--results_dir PATH] [--port PORT] [--host HOST]

Install dependencies (if not already):
    uv sync --extra microscope --extra serve
    # or: pip install fastapi "uvicorn[standard]"
"""

import argparse
import os


def main() -> None:
    parser = argparse.ArgumentParser(description="Diffusion Microscope live dashboard")
    parser.add_argument(
        "--results_dir",
        default="./experiment_results",
        help="Default results directory to open in the UI",
    )
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument("--host", default="127.0.0.1")
    args = parser.parse_args()

    os.environ.setdefault("DM_RESULTS_DIR", args.results_dir)

    try:
        import uvicorn
    except ImportError:
        raise SystemExit(
            "uvicorn not found.  Install with:\n"
            "  uv sync --extra serve\n"
            "  # or: pip install 'uvicorn[standard]'"
        )

    print(f"Starting Diffusion Microscope at  http://{args.host}:{args.port}")
    print(f"Default results dir: {args.results_dir}")
    uvicorn.run(
        "live_dashboard.server:app",
        host=args.host,
        port=args.port,
        reload=False,
    )


if __name__ == "__main__":
    main()
