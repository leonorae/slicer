# CLAUDE.md — Diffusion Microscope

This file guides Claude Code when working in this repository.

## Project overview

A research tool that visualises LLM hidden-state activations as images by:
1. Extracting last-token hidden states from each transformer layer
2. Projecting them into CLIP embedding space via a trained linear (Ridge) map
3. Feeding the projected vector into a frozen Stable Diffusion model as conditioning

The generated images act as a "microscope" — they reveal what the LLM's internal
representations look like to a cross-modal visual system.

```
LLM activation  ──▶  Linear projection  ──▶  CLIP space  ──▶  SD image
(per layer)           (Ridge, trained)        (frozen)         (frozen)
```

## Repository layout

```
slicer/
├── run_microscope.py        # Quick single-run CLI (train + probe one text)
├── run_experiment.py        # Batch experiment runner (full parameter grid)
├── run_pipeline.py          # Geometric visualisation pipeline (separate tool)
├── experiment_config.example.json  # Reference experiment config
├── pyproject.toml
│
├── diffusion_microscope/
│   ├── pipeline.py          # MicroscopePipeline — orchestrates phases
│   ├── projection.py        # LinearProjection, LayerProjectionSet
│   ├── clip_bridge.py       # CLIP extraction + LLM activation extraction
│   ├── generator.py         # DiffusionMicroscope — SD/SDXL image generation
│   ├── training_data.py     # Corpus loader (multiple HF dataset sources)
│   └── experiment.py        # ExperimentRunner — full parameter-grid runs
│
└── geometric_viz/           # Separate geometric analysis tool (PCA/UMAP/metrics)
```

## Development setup

```bash
uv sync --extra microscope        # install all deps
uv pip install -e .               # install package in editable mode
```

For GPU: add `--extra-index-url https://download.pytorch.org/whl/cu121` to `uv sync`.

## Common commands

```bash
# Quick test (built-in training texts, CPU)
python run_microscope.py --probe "a cat on a roof"

# Quick test with HF corpus
python run_microscope.py --auto_corpus --n_train 1000 --probe "a cat on a roof"

# Single-layer projection (original approach)
python run_microscope.py --projection_mode single_layer --probe "democracy"

# Mixed-layer projection
python run_microscope.py --projection_mode mixed_layer --probe "democracy"

# Full experiment run (reads experiment_config.example.json)
python run_experiment.py --config experiment_config.example.json --auto_corpus

# Resume only the generate phase
python run_experiment.py --config experiment_config.example.json --phase generate

# Generate dashboard after a run
python run_experiment.py --config experiment_config.example.json --phase dashboard

# Create layer-sweep animations
python run_experiment.py --config experiment_config.example.json --phase animations
```

## Projection modes

| Mode | Description | When to use |
|------|-------------|-------------|
| `per_layer` | One Ridge map per layer (default) | Best for studying layer-specific geometry |
| `single_layer` | One map from a single layer (use `--target_layer`) | Reproduces original approach |
| `mixed_layer` | One map trained on all layers pooled | Natural progression across layers |

Alpha can be a float or `"auto"` (RidgeCV cross-validation, recommended).

## Supported LLMs

Any `AutoModelForCausalLM` model should work. Tested/recommended small models:

| Model | HF ID | Params | Notes |
|-------|-------|--------|-------|
| GPT-2 | `gpt2` | 137M | Default; fast on CPU |
| GPT-2 Medium | `gpt2-medium` | 345M | More layers = richer sweep |
| GPT-2 Large | `gpt2-large` | 774M | Good balance |
| Pythia-70M | `EleutherAI/pythia-70m` | 70M | Tiny, great for testing |
| Pythia-160M | `EleutherAI/pythia-160m` | 160M | |
| Pythia-410M | `EleutherAI/pythia-410m` | 410M | |
| Pythia-1B | `EleutherAI/pythia-1b` | 1B | |
| OPT-125M | `facebook/opt-125m` | 125M | Left-padding (handled) |
| OPT-350M | `facebook/opt-350m` | 350M | |
| OPT-1.3B | `facebook/opt-1.3b` | 1.3B | |
| TinyLlama | `TinyLlama/TinyLlama-1.1B-Chat-v1.0` | 1.1B | RoPE, SwiGLU |
| SmolLM2-135M | `HuggingFaceTB/SmolLM2-135M` | 135M | Very fast |
| SmolLM2-360M | `HuggingFaceTB/SmolLM2-360M` | 360M | |
| Qwen2-0.5B | `Qwen/Qwen2-0.5B` | 500M | |

## Supported SD models

| Model | HF ID | Speed | Notes |
|-------|-------|-------|-------|
| SD 1.5 | `sd-legacy/stable-diffusion-v1-5` | ~30s/img (CPU) | Default |
| SD 2.1 | `stabilityai/stable-diffusion-2-1` | ~30s/img | Slightly higher quality |
| SDXL Turbo | `stabilityai/sdxl-turbo` | ~2s/img (GPU) | 1-4 steps, guidance=0 |

SDXL Turbo is auto-detected by model ID (`sdxl-turbo` substring). It uses
4 inference steps and `guidance_scale=0` by default.

## Training corpus sources

Controlled via `--corpus_sources` or `config.use_auto_corpus`:

| Key | Dataset | Content type |
|-----|---------|-------------|
| `flickr30k` | `nlphuji/flickr30k` | Visual image captions |
| `wikipedia` | `wikimedia/wikipedia` | Encyclopaedic sentences |
| `cc3m` | `google-research-datasets/conceptual_captions` | Web image alt-text |
| `wordnet` | NLTK WordNet | Noun definitions |
| `tinystories` | `roneneldan/TinyStories` | Short narrative stories |

Default mix uses all five equally. Weights can be customised via
`load_training_corpus(weights={"wikipedia": 2.0, "flickr30k": 1.0, ...})`.

## Adding a new corpus source

1. Add a `_load_mysource(n: int) -> list[str]` function to `training_data.py`
2. Register it in `_LOADERS` and `ALL_SOURCES`
3. Handle failures gracefully (raise an exception; the loader will catch and skip)

## Experiment output structure

```
experiment_results/
├── manifest.json            # All generated files + metadata + metrics
├── dashboard.html           # Self-contained browser viewer (generated)
├── configs/config_{ts}.json
├── projections/{proj_key}/  # Saved LinearProjection or LayerProjectionSet
├── grids/
│   ├── by_projection/{proj_key}/{text_slug}/
│   │   ├── per_layer/L{N}_CFG{v}_seed{s}.png   ← individual images
│   │   ├── grid_seed{s}.png                     ← composite grid
│   │   └── anim_CFG{v}_seed{s}.gif              ← layer animation
│   └── by_text/{text_slug}/{proj_key}_seed{s}.png  ← symlinks
└── .probe_cache/{slug}/layer_{N}.npy            ← cached LLM activations
```

## Architecture notes

- **Linearity is intentional**: Ridge regression preserves geometry. Differences
  between projected vectors reflect real differences in the LLM's representation
  space; the diffusion model is just a decoder.
- **SDXL conditioning**: Our 768-dim CLIP vector is placed in the first encoder
  slot; the second encoder (ViT-bigG, 1280-dim) is zero-padded. This is an
  approximation — full dual-encoder support would require a separate projection.
- **Probe activation cache**: `.probe_cache/` stores per-(text, layer) numpy
  files so the LLM doesn't reload for generate-only runs.
- **Checkpointing**: `manifest.json` tracks every image. All phases skip
  already-completed work on re-run.
