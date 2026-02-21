#!/usr/bin/env python3
"""
Command-line entry point for the Geometric Visualization Pipeline.

Examples
--------
# Analyse GPT-2 with built-in default prompts:
    python run_pipeline.py --model gpt2

# Analyse TinyLlama with custom prompts, skip UMAP for speed:
    python run_pipeline.py \\
        --model TinyLlama/TinyLlama-1.1B-Chat-v1.0 \\
        --prompts_file my_prompts.txt \\
        --no_umap

# Full run on Pythia-160M with custom output directory:
    python run_pipeline.py \\
        --model EleutherAI/pythia-160m \\
        --output_dir ./results/pythia \\
        --max_tokens 15000 \\
        --umap_sample_size 8000

Prompt file format: one prompt per line, blank lines ignored.
"""

import argparse
import sys
from pathlib import Path

from geometric_viz import GeometricFertilityPipeline

# ---------------------------------------------------------------------------
# Built-in default prompts for validation runs (see spec §"Validation")
# ---------------------------------------------------------------------------
DEFAULT_PROMPTS = [
    # ── Natural sentences (Wikitext-style) ───────────────────────────────
    "The history of science is the study of the development of science and "
    "scientific knowledge, including both natural and social science.",
    "In mathematics, a group is a set equipped with an operation that combines "
    "any two elements to form a third element within the same set.",
    "The economy of France is highly developed and free-market-oriented, "
    "with government intervention playing a notable but declining role.",
    "Language is a structured system of communication used by humans, "
    "involving the use of words in a structured and conventional way.",
    "Quantum mechanics is a fundamental theory in physics that describes "
    "nature at the smallest scales of energy levels of atoms and subatomic particles.",
    "The neural network was trained for ten epochs on the full dataset "
    "using stochastic gradient descent with a learning rate of 0.001.",
    "Transformer architectures process input sequences in parallel through "
    "multi-head self-attention and position-wise feed-forward layers.",
    "The mitochondria are membrane-bound organelles found in the cytoplasm "
    "of eukaryotic cells that generate most of the cell's supply of ATP.",
    "Climate change refers to long-term shifts in temperatures and weather "
    "patterns, mainly caused by human activities since the 1800s.",
    "The French Revolution was a period of radical political and societal "
    "change in France that began with the Estates General of 1789.",

    # ── Repeated-token sequences (amplifies positional signals) ──────────
    "the " * 32,
    "a " * 32,
    "is " * 32,
    "0 1 2 3 4 5 6 7 8 9 " * 6,

    # ── Structured/random-like sequences (test structure without semantics) ─
    "1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20",
    "( ) [ ] { } ; : , . ! ? @ # $ % ^ & * - + = / \\ | ~ ` ' \"",
    "alpha beta gamma delta epsilon zeta eta theta iota kappa lambda mu",
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Geometric Visualization Pipeline for transformer latent spaces.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # ── Required ──────────────────────────────────────────────────────────
    p.add_argument(
        "--model",
        required=True,
        metavar="MODEL_NAME",
        help="HuggingFace model name or local path (e.g. 'gpt2', "
             "'EleutherAI/pythia-160m', 'TinyLlama/TinyLlama-1.1B-Chat-v1.0').",
    )

    # ── Input ─────────────────────────────────────────────────────────────
    p.add_argument(
        "--prompts_file",
        default=None,
        metavar="FILE",
        help="Path to a text file with one prompt per line. "
             "If omitted, built-in default prompts are used.",
    )

    # ── Output ────────────────────────────────────────────────────────────
    p.add_argument(
        "--output_dir",
        default="./outputs",
        metavar="DIR",
        help="Root directory for outputs. A timestamped subdirectory is created per run.",
    )

    # ── Extraction ────────────────────────────────────────────────────────
    p.add_argument(
        "--max_length",
        type=int,
        default=128,
        help="Maximum token sequence length (longer sequences are truncated).",
    )
    p.add_argument(
        "--batch_size",
        type=int,
        default=4,
        help="Number of prompts per forward pass.",
    )
    p.add_argument(
        "--max_tokens",
        type=int,
        default=10_000,
        help="Maximum tokens per extraction point after global subsampling.",
    )

    # ── PCA ───────────────────────────────────────────────────────────────
    p.add_argument(
        "--n_pca_components",
        type=int,
        default=50,
        help="Number of PCA components to compute per extraction point.",
    )

    # ── UMAP ──────────────────────────────────────────────────────────────
    p.add_argument(
        "--umap_sample_size",
        type=int,
        default=5_000,
        help="Tokens sampled per extraction point when fitting the UMAP model.",
    )
    p.add_argument(
        "--umap_neighbors",
        type=int,
        default=15,
        help="UMAP n_neighbors (controls local vs. global structure trade-off).",
    )
    p.add_argument(
        "--no_umap",
        action="store_true",
        help="Skip UMAP entirely (much faster; useful for quick validation runs).",
    )

    # ── Device ────────────────────────────────────────────────────────────
    p.add_argument(
        "--device",
        default=None,
        metavar="DEVICE",
        help="PyTorch device string: 'cuda', 'cpu', or auto-detect if omitted.",
    )

    return p.parse_args()


def load_prompts(path: str) -> list:
    p = Path(path)
    if not p.exists():
        print(f"Error: prompts file not found: {path}", file=sys.stderr)
        sys.exit(1)
    lines = [l.strip() for l in p.read_text(encoding="utf-8").splitlines() if l.strip()]
    if not lines:
        print(f"Error: prompts file is empty: {path}", file=sys.stderr)
        sys.exit(1)
    return lines


def main() -> None:
    args = parse_args()

    # ── Load prompts ───────────────────────────────────────────────────────
    if args.prompts_file:
        prompts = load_prompts(args.prompts_file)
        print(f"Loaded {len(prompts)} prompts from {args.prompts_file}")
    else:
        prompts = DEFAULT_PROMPTS
        print(f"Using {len(prompts)} built-in default prompts")

    # ── Build pipeline ─────────────────────────────────────────────────────
    pipeline = GeometricFertilityPipeline(
        model_name=args.model,
        output_dir=args.output_dir,
        device=args.device,
        max_tokens=args.max_tokens,
    )

    # ── Run ────────────────────────────────────────────────────────────────
    results = pipeline.run(
        prompts=prompts,
        max_length=args.max_length,
        batch_size=args.batch_size,
        n_pca_components=args.n_pca_components,
        umap_sample_size=args.umap_sample_size,
        n_umap_neighbors=args.umap_neighbors,
        skip_umap=args.no_umap,
    )

    # ── Print summary ──────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("Pipeline complete.")
    print(f"Output directory : {results['run_dir']}")
    print()

    m = results["metrics"]

    if "intrinsic_dimensionality" in m:
        idim = m["intrinsic_dimensionality"]
        print("Intrinsic Dimensionality")
        print(f"  Components to 90% variance : {idim['n_to_90_percent']}")
        print(f"  Participation ratio         : {idim['participation_ratio']:.4f}")
        print(f"  Top-5 cumulative variance   : {idim['top_5_variance']:.4f}")
        print()

    if "attention_mlp_separability" in m:
        sep = m["attention_mlp_separability"]
        print("Attention–MLP Separability")
        if "error" in sep:
            print(f"  {sep['error']}")
        else:
            print(f"  Silhouette score   : {sep['silhouette_score']:.4f}")
            print(f"  Separability ratio : {sep['separability_ratio']:.4f}")
        print()

    if "position_dependence" in m:
        pos = m["position_dependence"]
        print("Position Dependence")
        if "error" in pos:
            print(f"  {pos['error']}")
        else:
            print(
                f"  Ridge R² (pos. pred.): "
                f"{pos['position_r2_mean']:.4f} ± {pos['position_r2_std']:.4f}"
            )
            print(
                "  Interpretation: ~0 → RoPE (position not in residual stream); "
                "~1 → learned absolute positional embeddings"
            )
        print()

    if results.get("plot_paths"):
        print("Plots saved:")
        for key, path in results["plot_paths"].items():
            print(f"  {key:<25} : {path}")
        print()

    print(f"Full report : {results['report_paths']['json']}")
    print(f"Text summary: {results['report_paths']['txt']}")


if __name__ == "__main__":
    main()
