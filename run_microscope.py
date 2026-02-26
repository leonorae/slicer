#!/usr/bin/env python3
"""
CLI for the Diffusion Microscope.

Renders LLM hidden-state activations as images by projecting them into CLIP
space and feeding the result through a frozen Stable Diffusion model.

Examples
--------
# Quick test with GPT-2 and SD 2.1 (CPU, slow but works):
    python run_microscope.py --llm gpt2 --probe "The cat sat on the mat"

# Full run with cached data and GPU:
    python run_microscope.py \\
        --llm gpt2 \\
        --sd stabilityai/stable-diffusion-2-1 \\
        --device cuda \\
        --cache_dir ./microscope_cache \\
        --probe "The cat sat on the mat" \\
        --probe "Justice is a social construct" \\
        --probe "x = torch.randn(3, 4)"

# Use a pre-trained projection (skip training):
    python run_microscope.py \\
        --llm gpt2 \\
        --load_projection ./microscope_outputs/run_20260221/projection \\
        --probe "Hello world"
"""

import argparse
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Default training texts (a diverse sample; ideally 5k+ for production)
# ---------------------------------------------------------------------------
DEFAULT_TRAINING_TEXTS = [
    # Concrete visual
    "A golden retriever running across a sunny meadow",
    "A red sports car parked in front of a glass building",
    "Snow-capped mountains reflected in a clear alpine lake",
    "A bowl of fresh strawberries on a wooden kitchen table",
    "Children playing on a sandy beach at sunset",
    "A black cat sitting on a windowsill watching rain",
    "An old stone bridge over a misty river at dawn",
    "Rows of colorful tulips in a Dutch flower garden",
    "A steaming cup of coffee next to an open book",
    "Fireworks exploding over a city skyline at night",

    # Concrete but less visual
    "The chemical formula for water is H2O",
    "DNA consists of two polynucleotide chains forming a double helix",
    "The speed of light in a vacuum is 299792458 metres per second",
    "Photosynthesis converts carbon dioxide and water into glucose and oxygen",
    "The Krebs cycle produces two molecules of ATP per glucose",

    # Abstract / conceptual
    "The concept of infinity has puzzled philosophers for millennia",
    "Democracy requires the informed consent of the governed",
    "Entropy always increases in an isolated thermodynamic system",
    "The categorical imperative demands universalisability of maxims",
    "Consciousness remains the hard problem of philosophy of mind",
    "Justice must balance individual rights against collective welfare",
    "The uncertainty principle limits simultaneous knowledge of position and momentum",
    "Truth is correspondence between propositions and states of affairs",
    "Free will may be an illusion created by deterministic neural processes",
    "Language shapes thought according to the Sapir-Whorf hypothesis",

    # Narrative / temporal
    "She opened the door and found the room completely empty",
    "After three years of research the team finally published their findings",
    "The train arrived late, and the passengers rushed to the platform",
    "He remembered the conversation they had ten years ago by the river",
    "The company grew from a small startup to a global enterprise in a decade",

    # Technical / code-like
    "import torch; model = torch.nn.Linear(768, 512)",
    "SELECT * FROM users WHERE created_at > '2024-01-01' ORDER BY id",
    "git commit -m 'fix: resolve race condition in connection pool'",
    "The transformer attention mechanism computes Q K V matrices",
    "Backpropagation computes gradients by applying the chain rule layer by layer",

    # Function words / minimal semantic content
    "the the the the the the the the the the the the",
    "however nevertheless furthermore moreover consequently",
    "if and or but not yet so for nor",
    "very quite rather somewhat extremely incredibly",

    # Mixed / unusual
    "The number 42 is the answer to life the universe and everything",
    "Colorless green ideas sleep furiously",
    "Buffalo buffalo Buffalo buffalo buffalo buffalo Buffalo buffalo",
    "Time flies like an arrow but fruit flies like a banana",
]


DEFAULT_PROBES = [
    "A beautiful sunset over the ocean",
    "The meaning of life is uncertain",
    "import numpy as np",
    "the the the the the",
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Diffusion Microscope — visualise LLM internals via Stable Diffusion",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Models
    p.add_argument("--llm", default="gpt2", help="HuggingFace LLM model name")
    p.add_argument(
        "--sd", default="sd-legacy/stable-diffusion-v1-5",
        help="HuggingFace Stable Diffusion model ID",
    )
    p.add_argument(
        "--clip_model", default="ViT-L-14",
        help="OpenCLIP architecture (must match SD's text encoder)",
    )
    p.add_argument(
        "--clip_pretrained", default="openai",
        help="OpenCLIP pretrained weights tag",
    )

    # Texts
    p.add_argument(
        "--training_texts_file", default=None,
        help="File with one training text per line (default: built-in set)",
    )
    p.add_argument(
        "--auto_corpus", action="store_true",
        help="Download a diverse training corpus from public HF datasets instead "
             "of using the small built-in set.",
    )
    p.add_argument(
        "--n_train", type=int, default=5000,
        help="Number of training texts when --auto_corpus is set (default: 5000).",
    )
    p.add_argument(
        "--corpus_sources", default="coco,wikipedia,cc3m,wordnet",
        help="Comma-separated corpus sources for --auto_corpus. "
             "Available: coco, wikipedia, cc3m, wordnet",
    )
    p.add_argument(
        "--probe", action="append", default=None,
        help="Text to visualise (repeatable). If none given, uses defaults.",
    )

    # Projection
    p.add_argument("--alpha", type=float, default=1.0, help="Ridge regularisation")
    p.add_argument(
        "--target_layer", type=int, default=None,
        help="LLM layer to project from (default: last layer)",
    )
    p.add_argument(
        "--load_projection", default=None,
        help="Path to a saved projection directory (skip training)",
    )

    # Generation
    p.add_argument(
        "--cfg_scales", default="3.0,7.0,12.0",
        help="Comma-separated CFG guidance scales",
    )
    p.add_argument("--seed", type=int, default=42, help="RNG seed")
    p.add_argument(
        "--steps", type=int, default=30, help="SD inference steps",
    )

    # Infrastructure
    p.add_argument("--device", default=None, help="'cpu' or 'cuda' (auto-detect)")
    p.add_argument("--output_dir", default="./microscope_outputs", help="Output root")
    p.add_argument("--cache_dir", default=None, help="Cache extracted activations")

    return p.parse_args()


def main() -> None:
    args = parse_args()

    # Auto-detect device
    device = args.device
    if device is None:
        try:
            import torch
            device = "cuda" if torch.cuda.is_available() else "cpu"
        except ImportError:
            device = "cpu"
    print(f"Device: {device}")

    # Load training texts
    if args.training_texts_file:
        texts_path = Path(args.training_texts_file)
        if not texts_path.exists():
            print(f"Error: training texts file not found: {texts_path}", file=sys.stderr)
            sys.exit(1)
        training_texts = [
            l.strip() for l in texts_path.read_text().splitlines() if l.strip()
        ]
        print(f"Loaded {len(training_texts)} training texts from {texts_path}")
    elif args.auto_corpus:
        from diffusion_microscope.training_data import load_training_corpus, ALL_SOURCES
        sources_raw = [s.strip() for s in args.corpus_sources.split(",") if s.strip()]
        bad = [s for s in sources_raw if s not in ALL_SOURCES]
        if bad:
            print(f"Error: unknown corpus sources: {bad}. "
                  f"Valid: {list(ALL_SOURCES)}", file=sys.stderr)
            sys.exit(1)
        print(f"Building auto corpus: n={args.n_train}, sources={sources_raw}")
        training_texts = load_training_corpus(
            n_total=args.n_train,
            sources=tuple(sources_raw),
            seed=args.seed,
        )
    else:
        training_texts = DEFAULT_TRAINING_TEXTS
        print(f"Using {len(training_texts)} built-in training texts")

    # Probe texts
    probe_texts = args.probe if args.probe else DEFAULT_PROBES
    print(f"Probe texts: {len(probe_texts)}")

    cfg_scales = [float(x) for x in args.cfg_scales.split(",")]

    from diffusion_microscope import MicroscopePipeline

    pipeline = MicroscopePipeline(
        llm_model_name=args.llm,
        sd_model_id=args.sd,
        clip_model_name=args.clip_model,
        clip_pretrained=args.clip_pretrained,
        output_dir=args.output_dir,
        device=device,
    )

    if args.load_projection:
        # Skip training — load existing projection and just generate
        from diffusion_microscope.projection import LinearProjection

        proj = LinearProjection.load(args.load_projection)
        print(f"Loaded projection: {proj}")

        data = pipeline.prepare_training_data(
            training_texts, cache_dir=args.cache_dir,
        )

        for i, text in enumerate(probe_texts):
            print(f"\n=== Probe {i+1}/{len(probe_texts)}: {text[:60]} ===")
            pipeline.generate_layer_sweep(
                text, proj, data, cfg_scales=cfg_scales, seed=args.seed,
            )
    else:
        results = pipeline.run(
            training_texts=training_texts,
            probe_texts=probe_texts,
            target_layer=args.target_layer,
            alpha=args.alpha,
            cfg_scales=cfg_scales,
            cache_dir=args.cache_dir,
            seed=args.seed,
        )

        print("\n" + "=" * 60)
        print("Diffusion Microscope — Run Complete")
        print(f"Output directory: {results['run_dir']}")
        print(f"\nValidation:")
        for k, v in results["validation"].items():
            print(f"  {k}: {v:.4f}" if isinstance(v, float) else f"  {k}: {v}")


if __name__ == "__main__":
    main()
