#!/usr/bin/env python3
"""
Batch experiment runner for the Diffusion Microscope.

Reads a JSON configuration file and executes the requested phases:
  train     – train Ridge maps, store SVD spectra
  generate  – generate probe images for all parameter combinations
  grids     – compose layer × CFG grid PNGs
  analyze   – produce erank-vs-layer plots and singular value heatmaps
  dashboard – write self-contained dashboard.html
  all       – run every phase in order

Examples
--------
# Full run (train + generate + grids + analyze + dashboard):
    python run_experiment.py --config experiment_config.example.json --auto_corpus

# Train only (useful when generation is done separately):
    python run_experiment.py --config cfg.json --phase train --auto_corpus

# Analyze already-trained projections (no models needed):
    python run_experiment.py --config cfg.json --phase analyze dashboard

# Override model and device:
    python run_experiment.py --config cfg.json --llm gpt2-medium --device cuda
"""

import argparse
import json
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Diffusion Microscope — batch experiment runner",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    p.add_argument(
        "--config", required=True,
        help="Path to experiment JSON config file",
    )
    p.add_argument(
        "--phase",
        nargs="+",
        choices=["train", "generate", "grids", "analyze", "dashboard", "all"],
        default=["all"],
        metavar="PHASE",
        help=(
            "Phases to run (space-separated, default: all).\n"
            "  train     – train all (projection_type × alpha) combinations\n"
            "  generate  – generate images for all parameter combinations\n"
            "  grids     – compose per-(projection, probe, seed) grid images\n"
            "  analyze   – erank-vs-layer plots + singular value heatmaps\n"
            "  dashboard – write self-contained HTML dashboard\n"
            "  all       – run every phase in order"
        ),
    )

    # Model overrides
    p.add_argument("--llm", default=None, metavar="MODEL",
                   help="HuggingFace LLM model name (overrides config)")
    p.add_argument("--sd", default=None, metavar="MODEL",
                   help="HuggingFace SD model ID (overrides config)")
    p.add_argument("--clip_model", default=None,
                   help="OpenCLIP architecture (overrides config)")
    p.add_argument("--clip_pretrained", default=None,
                   help="OpenCLIP weights tag (overrides config)")
    p.add_argument("--device", default=None,
                   help="'cpu' or 'cuda' (overrides config; default: auto-detect)")
    p.add_argument("--cache_dir", default=None,
                   help="Directory for caching LLM training-data activations")

    # Training corpus
    corpus = p.add_mutually_exclusive_group()
    corpus.add_argument(
        "--auto_corpus", action="store_true",
        help="Download training corpus from public HF datasets",
    )
    corpus.add_argument(
        "--training_texts_file", default=None, metavar="FILE",
        help="File with one training text per line",
    )
    p.add_argument(
        "--n_train", type=int, default=None,
        help="Training corpus size (overrides config projections.training_data_size)",
    )
    p.add_argument(
        "--corpus_sources",
        default="flickr30k,wikipedia,cc3m,wordnet,tinystories",
        help="Comma-separated corpus sources for --auto_corpus",
    )

    return p.parse_args()


def main() -> None:
    args = parse_args()

    config_path = Path(args.config)
    if not config_path.exists():
        print(f"Error: config file not found: {config_path}", file=sys.stderr)
        sys.exit(1)
    with open(config_path) as f:
        config = json.load(f)

    # Model settings: config["models"] overridden by CLI flags
    model_cfg = config.get("models", {})
    llm_model = args.llm or model_cfg.get("llm", "gpt2")
    sd_model = args.sd or model_cfg.get("sd", "sd-legacy/stable-diffusion-v1-5")
    clip_model = args.clip_model or model_cfg.get("clip_model", "ViT-L-14")
    clip_pretrained = args.clip_pretrained or model_cfg.get("clip_pretrained", "openai")

    # Device
    device = args.device or model_cfg.get("device")
    if device is None:
        try:
            import torch
            device = "cuda" if torch.cuda.is_available() else "cpu"
        except ImportError:
            device = "cpu"

    # Override n_train in config if provided
    if args.n_train is not None:
        config.setdefault("projections", {})["training_data_size"] = args.n_train

    # Training texts — only needed for the train phase
    need_corpus = "train" in args.phase
    if args.training_texts_file:
        texts_path = Path(args.training_texts_file)
        if not texts_path.exists():
            print(f"Error: file not found: {texts_path}", file=sys.stderr)
            sys.exit(1)
        training_texts = [
            line.strip()
            for line in texts_path.read_text().splitlines()
            if line.strip()
        ]
        print(f"Loaded {len(training_texts)} training texts from {texts_path}")

    elif need_corpus and (args.auto_corpus or config.get("use_auto_corpus", False)):
        from diffusion_microscope.training_data import load_training_corpus, ALL_SOURCES

        sources_raw = [s.strip() for s in args.corpus_sources.split(",") if s.strip()]
        bad = [s for s in sources_raw if s not in ALL_SOURCES]
        if bad:
            print(
                f"Error: unknown corpus sources: {bad}. Valid: {list(ALL_SOURCES)}",
                file=sys.stderr,
            )
            sys.exit(1)
        n_train = config.get("projections", {}).get("training_data_size", 5000)
        print(f"Building auto corpus: n={n_train}, sources={sources_raw}")
        training_texts = load_training_corpus(n_total=n_train, sources=tuple(sources_raw))

    elif need_corpus:
        # Minimal built-in fallback
        training_texts = [
            "a cat on a roof",
            "a dog in a field",
            "a red apple on a white table",
            "the ocean at sunset",
            "a mountain covered in snow",
        ]
        print(
            f"Using {len(training_texts)} built-in training texts "
            "(pass --auto_corpus for better results)"
        )
    else:
        training_texts = []

    phases = args.phase
    base_dir = config.get("output", {}).get("base_dir", "./experiment_results")

    print(f"\nConfig  : {config_path}")
    print(f"LLM     : {llm_model}")
    print(f"SD      : {sd_model}")
    print(f"Device  : {device}")
    print(f"Output  : {base_dir}")
    print(f"Phases  : {phases}\n")

    from diffusion_microscope.experiment import ExperimentRunner

    runner = ExperimentRunner(
        config=config,
        llm_model=llm_model,
        sd_model=sd_model,
        clip_model=clip_model,
        clip_pretrained=clip_pretrained,
        device=device,
        cache_dir=args.cache_dir,
    )

    results = runner.run(training_texts=training_texts, phases=phases)

    manifest = results["manifest"]
    print(f"\n{'='*60}")
    print(f"Experiment complete")
    print(f"Output       : {results['base_dir']}")
    print(f"Projections  : {len(manifest.get('projections', {}))}")
    print(f"Images       : {len(manifest.get('images', {}))}")
    print(f"Grids        : {len(manifest.get('grids', {}))}")
    if "dashboard" in phases or "all" in phases:
        print(f"Dashboard    : {results['base_dir']}/dashboard.html")


if __name__ == "__main__":
    main()
