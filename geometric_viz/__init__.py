"""
geometric_viz – Geometric Visualization Pipeline for transformer latent spaces.

This package extracts internal activations from pretrained HuggingFace transformer
models and applies PCA/UMAP to produce quantitative metrics and visualisations of
latent-space geometry.  It is part of Phase 0.5 of a research program investigating
how pretraining data and architecture shape "geometric fertility" – the richness of
off-principal subspaces that RL-fine-tuning can cheaply navigate.

Key entry points
----------------
GeometricFertilityPipeline
    High-level orchestrator; load a model and call .run(prompts).

ActivationExtractor
    Registers forward hooks and collects activations at up to 7 points per layer.

PCAAnalyzer / UMAPAnalyzer
    Dimensionality reduction.  UMAP uses a joint fitting sample for cross-point
    comparability.

compute_intrinsic_dimensionality / compute_attention_mlp_separability /
compute_position_dependence
    Stand-alone metric functions.

GeometryVisualizer
    Produces scree, UMAP scatter, and layer-evolution PNG plots.

GeometryReport
    Accumulates results and writes JSON + text report files.

Quick start
-----------
>>> from geometric_viz import GeometricFertilityPipeline
>>> pipe = GeometricFertilityPipeline("gpt2", output_dir="./outputs")
>>> results = pipe.run(["The quick brown fox jumps over the lazy dog."] * 20)
>>> print(results["metrics"]["intrinsic_dimensionality"])
"""

from .pipeline import GeometricFertilityPipeline
from .extractor import ActivationExtractor
from .analyzers import PCAAnalyzer, UMAPAnalyzer
from .metrics import (
    compute_intrinsic_dimensionality,
    compute_separability,
    compute_attention_mlp_separability,
    compute_position_dependence,
)
from .visualizer import GeometryVisualizer
from .report import GeometryReport

__all__ = [
    "GeometricFertilityPipeline",
    "ActivationExtractor",
    "PCAAnalyzer",
    "UMAPAnalyzer",
    "compute_intrinsic_dimensionality",
    "compute_separability",
    "compute_attention_mlp_separability",
    "compute_position_dependence",
    "GeometryVisualizer",
    "GeometryReport",
]
