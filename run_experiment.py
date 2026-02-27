#!/usr/bin/env python3
"""
Batch experiment runner for the Diffusion Microscope.

Reads a JSON configuration file and systematically trains projections,
generates images for all parameter combinations, composes grid images,
and computes metrics.  Fully checkpointed — re-running resumes from where
it left off.

Examples
--------
# Full run:
    python run_experiment.py --config experiment_config.example.json

# Only train projections, then generate images (resume-friendly):
    python run_experiment.py --config cfg.json --phase train
    python run_experiment.py --config cfg.json --phase generate grids

# Use HF datasets corpus for training texts:
    python run_experiment.py --config cfg.json --auto_corpus --n_train 5000

# Use a GPU:
    python run_experiment.py --config cfg.json --device cuda

# Skip training (use saved projections) and only compute metrics:
    python run_experiment.py --config cfg.json --phase metrics
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
        help="Path to experiment JSON config file (see experiment_config.example.json)",
    )
    p.add_argument(
        "--phase",
        nargs="+",
        choices=["train", "generate", "grids", "metrics", "all"],
        default=["all"],
        metavar="PHASE",
        help=(
            "Phases to run (space-separated, default: all).\n"
            "  train    – train all (projection_type × alpha) combinations\n"
            "  generate – generate images for all parameter combinations\n"
            "  grids    – compose per-(projection, probe, seed) grid images\n"
            "  metrics  – compute LPIPS + image variance\n"
            "  all      – run every phase in order"
        ),
    )

    # Model overrides — also settable under config["models"]
    p.add_argument("--llm", default=None, metavar="MODEL",
                   help="HuggingFace LLM model name (overrides config)")
    p.add_argument("--sd", default=None, metavar="MODEL",
                   help="HuggingFace SD model ID (overrides config)")
    p.add_argument("--clip_model", default=None,
                   help="OpenCLIP architecture (overrides config)")
    p.add_argument("--clip_pretrained", default=None,
                   help="OpenCLIP weights tag (overrides config)")
    p.add_argument("--device", default=None,
                   help="'cpu' or 'cuda' (overrides config, default: auto-detect)")
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
        "--corpus_sources", default="coco,wikipedia,cc3m,wordnet",
        help="Comma-separated sources for --auto_corpus",
    )

    return p.parse_args()


def main() -> None:
    args = parse_args()

    # Load config
    config_path = Path(args.config)
    if not config_path.exists():
        print(f"Error: config file not found: {config_path}", file=sys.stderr)
        sys.exit(1)
    with open(config_path) as f:
        config = json.load(f)

    # Model settings: config["models"] < CLI flags
    model_cfg = config.get("models", {})
    llm_model = args.llm or model_cfg.get("llm", "gpt2")
    sd_model = args.sd or model_cfg.get("sd", "sd-legacy/stable-diffusion-v1-5")
    clip_model = args.clip_model or model_cfg.get("clip_model", "ViT-L-14")
    clip_pretrained = args.clip_pretrained or model_cfg.get(
        "clip_pretrained", "openai"
    )

    # Device
    device = args.device or model_cfg.get("device")
    if device is None:
        try:
            import torch
            device = "cuda" if torch.cuda.is_available() else "cpu"
        except ImportError:
            device = "cpu"
    print(f"Device: {device}")

    # Training corpus size
    n_train = args.n_train or config.get("projections", {}).get(
        "training_data_size", 5000
    )

    # Training texts
    if args.training_texts_file:
        texts_path = Path(args.training_texts_file)
        if not texts_path.exists():
            print(
                f"Error: training texts file not found: {texts_path}",
                file=sys.stderr,
            )
            sys.exit(1)
        training_texts = [
            line.strip()
            for line in texts_path.read_text().splitlines()
            if line.strip()
        ]
        print(f"Loaded {len(training_texts)} training texts from {texts_path}")

    elif args.auto_corpus or config.get("use_auto_corpus", False):
        from diffusion_microscope.training_data import load_training_corpus, ALL_SOURCES

        sources_raw = [s.strip() for s in args.corpus_sources.split(",") if s.strip()]
        bad = [s for s in sources_raw if s not in ALL_SOURCES]
        if bad:
            print(
                f"Error: unknown corpus sources: {bad}. Valid: {list(ALL_SOURCES)}",
                file=sys.stderr,
            )
            sys.exit(1)
        print(f"Building auto corpus: n={n_train}, sources={sources_raw}")
        training_texts = load_training_corpus(
            n_total=n_train, sources=tuple(sources_raw)
        )

    else:
        from run_microscope import DEFAULT_TRAINING_TEXTS

        training_texts = DEFAULT_TRAINING_TEXTS
        print(
            f"Using {len(training_texts)} built-in training texts "
            f"(consider --auto_corpus for better results)"
        )

    # Resolve phases
    phases = args.phase
    if "all" in phases:
        phases = ["train", "generate", "grids", "metrics"]
    print(f"Phases : {phases}")
    print(f"Config : {config_path}")
    print(f"Output : {config.get('output', {}).get('base_dir', './experiment_results')}")
    print()

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

    n_images = len(results["manifest"].get("images", {}))
    n_grids = len(results["manifest"].get("grids", {}))
    print(f"\n{'='*60}")
    print(f"Experiment complete")
    print(f"Output : {results['base_dir']}")
    print(f"Images : {n_images}")
    print(f"Grids  : {n_grids}")


if __name__ == "__main__":
    main()
