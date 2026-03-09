# CLAUDE.md — Diffusion Microscope

This file guides Claude Code when working in this repository.

## What this project does

Visualises LLM hidden-state activations as images:
1. Extract last-token hidden states from every transformer layer
2. Project them into CLIP embedding space via a trained Ridge regression map
3. Feed the projected vector into a frozen Stable Diffusion model as conditioning

```
LLM activation  ──▶  Ridge projection  ──▶  CLIP space  ──▶  SD image
(per layer)           (trained, frozen)      (frozen)         (frozen)
```

The diffusion model is never trained — it is purely a decoder.  All variation
in the output comes from the projected LLM activation.

## Repository layout

```
slicer/
├── run_microscope.py        # Quick single-run CLI
├── run_experiment.py        # Batch experiment runner (full parameter grid)
├── run_pipeline.py          # Geometric visualisation pipeline (separate tool)
├── experiment_config.example.json
├── pyproject.toml
│
├── diffusion_microscope/
│   ├── pipeline.py          # MicroscopePipeline — orchestrates all phases
│   ├── projection.py        # LinearProjection, LayerProjectionSet, _auto_alpha
│   ├── clip_bridge.py       # CLIP extraction, LLM activation extraction,
│   │                        #   format_sd_conditioning, format_sdxl_conditioning
│   ├── generator.py         # DiffusionMicroscope — SD 1.x / SDXL image gen
│   ├── training_data.py     # load_training_corpus — 5 HF/NLTK sources
│   └── experiment.py        # ExperimentRunner — full grid with checkpointing
│
└── geometric_viz/           # Separate PCA/UMAP analysis tool (unrelated)
```

## Development setup

```bash
uv sync --extra microscope
uv pip install -e .
# GPU: add --extra-index-url https://download.pytorch.org/whl/cu121
```

## Key commands

```bash
python run_microscope.py --probe "a cat on a roof"
python run_experiment.py --config experiment_config.example.json --auto_corpus
python run_experiment.py --config experiment_config.example.json --phase dashboard
```

## Projection modes

| Mode | Class | Description |
|------|-------|-------------|
| `per_layer` | `LayerProjectionSet` | One `LinearProjection` per layer (default) |
| `single_layer` | `LinearProjection` | One map from a chosen layer |
| `mixed_layer` | `LinearProjection` | One map trained on all layers pooled |

`alpha="auto"` → `_auto_alpha()` → `RidgeCV` with grid `[0.01, 0.1, 1, 10, 100, 1000]`.

## Supported LLMs

Any `AutoModelForCausalLM`. Left-padding models (OPT) handled in `clip_bridge.py`.

| Model | HF ID |
|-------|-------|
| GPT-2 | `gpt2` |
| GPT-2 Medium/Large | `gpt2-medium`, `gpt2-large` |
| Pythia-70M–1B | `EleutherAI/pythia-{70m,160m,410m,1b}` |
| OPT-125M–1.3B | `facebook/opt-{125m,350m,1.3b}` |
| TinyLlama | `TinyLlama/TinyLlama-1.1B-Chat-v1.0` |
| SmolLM2 | `HuggingFaceTB/SmolLM2-{135M,360M}` |
| Qwen2-0.5B | `Qwen/Qwen2-0.5B` |

## Supported SD models

| Model | HF ID | Notes |
|-------|-------|-------|
| SD 1.5 | `sd-legacy/stable-diffusion-v1-5` | Default |
| SD 2.1 | `stabilityai/stable-diffusion-2-1` | |
| SDXL Turbo | `stabilityai/sdxl-turbo` | Auto-detected by `"xl"` in ID; 4 steps, CFG=0 |

SDXL conditioning: 768-dim projection is zero-padded to 2048 for the sequence
slot; pooled embedding (1280-dim) is zeros.

## Training corpus sources

| Key | Dataset |
|-----|---------|
| `flickr30k` | `phiyodr/flickr30k` |
| `wikipedia` | `wikimedia/wikipedia` |
| `cc3m` | `google-research-datasets/conceptual_captions` |
| `wordnet` | NLTK WordNet |
| `tinystories` | `roneneldan/TinyStories` |

To add a source: add `_load_mysource(n) -> list[str]` to `training_data.py`,
register in `_LOADERS` and `ALL_SOURCES`.

## Experiment output structure

```
experiment_results/
├── manifest.json                        # Checkpointing key
├── dashboard.html
├── configs/config_{ts}.json
├── projections/{proj_key}/
├── grids/by_projection/{proj_key}/{text_slug}/
│   ├── per_layer/L{N}_CFG{v}_seed{s}.png
│   ├── grid_seed{s}.png
│   └── anim_CFG{v}_seed{s}.gif
├── grids/by_text/{text_slug}/           # Symlinks
└── .probe_cache/{slug}/layer_{N}.npy
```

## Experiment phases

| Phase | What it does |
|-------|-------------|
| `train` | Fit all (projection_type × alpha) combinations |
| `generate` | Render all (proj × probe × layer × CFG × seed) images |
| `grids` | Compose layer × CFG grid PNGs |
| `metrics` | Post-hoc LPIPS + image variance |
| `animations` | Layer-sweep GIFs per (proj, text, CFG, seed) |
| `dashboard` | Write `dashboard.html` |

All phases are idempotent — re-running skips completed work via `manifest.json`.
Inline LPIPS (`track_lpips=True`) runs during `generate` and is cheaper than
the post-hoc `metrics` phase.

## Architecture notes

- **Linearity is intentional** — Ridge preserves geometry; it's a linear readout,
  not a feature extractor.
- **No fine-tuning** — SD and CLIP are completely frozen.  Only the Ridge map
  is trained (~hidden_dim × 768 parameters).
- **Probe cache** — `.probe_cache/` stores per-(text, layer) numpy arrays so
  the LLM only runs once per probe set.
- **Manifest** — `manifest.json` is the single source of truth for completed work.

## Observed results (GPT-2, 2 experiments)

> **Caveat:** All findings below come from only two experiment runs with GPT-2.
> Generalisation to other models is untested.

### Geometry (from `analyze` phase)

- **All 12 layers are full-rank** relative to the 768-dim CLIP target.  No dead
  zones; every layer in principle carries projectable signal.
- **Effective rank (erank) peaks at mid layers (~6–8)** and is lowest at layer 0
  and layer 11.  Early layers are spiky (signal concentrated in ~50 directions);
  late layers broaden but shrink in overall variance.
- **Singular value spectra** are smooth and decay gradually — no cliff edges.
  This means the projection is well-conditioned and alpha choice affects magnitude
  uniformly rather than selectively.

### Alpha behaviour (observed)

- `alpha=1` and `alpha=10` produce the most layer-discriminative outputs.
- `alpha=1000` and `auto` collapse per-layer differences — layer sweeps look
  visually homogeneous.
- `auto` (RidgeCV) selected high alpha values, favouring stability over
  expressiveness in this corpus size.

### Known limitations

- Only 2 experiment runs completed.  Effect sizes are not yet confirmed as stable.
- Training corpus was small (< 5000 samples in both runs).  Larger corpora may
  shift the alpha → quality relationship.
- GPT-2 only.  Mid-layer erank peak and early-layer spikiness may not generalise
  to deeper or instruction-tuned models.
- `generate` phase with non-GPT-2 models was blocked by a `model_id` kwarg bug
  (now fixed in `experiment.py:_load_sd`).
