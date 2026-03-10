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

## Observed results (GPT-2, per-layer experiment, 5000-sample corpus)

> **Caveat:** Findings below come from GPT-2 only. Generalisation to other models
> is untested. No images have been generated from the 5000-sample run yet — visual
> claims below are inferences from projection metrics, not confirmed visually.
>
> **Disregard** any renders from earlier runs using ~43 training samples and
> single-layer Ridge regression. Those showed dark middle layers as an artifact of
> severe underfitting at that corpus size, not a property of the model.

### Geometry (from per-layer SVD analysis, manifest.json)

- **Layers 1–11 are effectively full-rank.** n_visible ≈ 765/768 for all
  layers ≥ 1. The projection covers nearly the entire CLIP space — no dead zones.
- **Layer 0 is numerically singular at low alpha** (condition_number = ∞ at
  alpha = 1, 10, and auto). At high alpha (100, 1000) it becomes well-conditioned.
  Treat L0 projections with caution; they are rank-deficient under low regularisation.
- **Erank increases monotonically from L0 to L11** — it does not peak at mid-layers.
  At alpha=1000: L0=349, L6=432, L11=448. At alpha=1: L0=200, L6=396, L11=422.
  Early layers are genuinely more concentrated (lower erank); late layers spread
  variance more evenly across directions.
- **Singular value spectra are smooth and decay gradually** — no cliff edges.
  Alpha scales magnitude uniformly across the spectrum rather than selectively
  zeroing directions.

### Geometric implications

- **Probing is unconstrained for layers 1–11.** The projection spans ~765 of 768
  CLIP dimensions. Any linear combination of LLM activations maps to a reachable
  point; there is no null region that SD cannot decode.
- **Discrimination comes for free from the projection geometry.** Because the
  spectrum is full-rank and smooth, two probes that differ anywhere in activation
  space produce distinguishable CLIP vectors. The projection does not introduce a
  bottleneck that erases resolution.
- **The limiting factor is the LLM's geometry, not the projection's.** If two
  concepts activate the same directions in the LLM, their images will look similar
  regardless of alpha or layer. The Ridge map faithfully transmits whatever structure
  is present in the activations.
- **Layer choice determines which kind of structure is transmitted.** Lower erank
  at L0 means early-layer variance is concentrated in fewer directions — likely
  token-identity and position. Higher erank at L11 means late-layer variance is
  spread more evenly, encoding more compositional and semantic content. Semantic
  probing will be most interpretable at mid-to-late layers.

### Alpha behaviour (observed, 5000-sample corpus)

- **`auto` (RidgeCV) selected alpha=1000 for every layer** — hitting the top of
  the grid `[0.01, 0.1, 1, 10, 100, 1000]`. This is a ceiling effect; the true
  CV optimum may be even higher. RidgeCV optimises R², which prefers high
  regularisation.
- **R² and nn_recall@5 pull in opposite directions.** High alpha maximises R² but
  collapses neighbourhood structure. Low alpha gives negative R² on early layers
  (overfitting) but preserves topology better at mid-to-late layers:

  | alpha | L11 R²  | L11 nn_recall@5 |
  |-------|---------|-----------------|
  | 1     | 0.170   | **0.686**       |
  | 10    | 0.188   | 0.684           |
  | 100   | 0.266   | 0.667           |
  | 1000  | 0.339   | 0.517           |

- **Alpha acts as a compression ratio** on the dynamic range of projected vectors.
  High alpha brings all probe projections toward the mean, reducing the amplitude
  difference between "cat" and "democracy" at any given layer. Low alpha preserves
  relative distances but amplifies noise, especially where the projection is
  ill-conditioned (L0 at low alpha).
- **RidgeCV is not the right optimiser** if the goal is visually discriminative
  images. nn_recall@5 better predicts whether semantically distinct probes produce
  distinct images. Consider alpha=1 or alpha=10 for generation; use alpha=1000
  only if stability across layers matters more than discriminability.

### Pythia-410m results (24 layers, 5000-sample corpus)

A second run with `EleutherAI/pythia-410m` produced meaningfully different geometry.

**Key differences from GPT-2:**

- **No singular layers.** All 24 layers are well-conditioned at every alpha
  (condition numbers ~48M–195M throughout). The L0 singularity observed in GPT-2
  does not appear in Pythia-410m.
- **n_visible = 767/768 for every layer and every alpha** — full coverage of CLIP
  space without exception.
- **Higher absolute erank across the board.** At alpha=1, GPT-2 L0 erank=200;
  Pythia-410m L0 erank=354. The whole distribution is shifted toward higher rank.
- **Erank profile is non-monotonic at high alpha.** At alpha=1000, L0 has the
  *highest* erank (494), L1 dips to 478, then rises gradually to ~497 at L20+.
  This L0-peak / L1-dip pattern is absent at low alpha and likely reflects
  alpha's differential effect on the embedding layer vs. transformer layers.
- **At alpha=1, erank increases rapidly through early layers** (L0=354 → L6=455)
  then plateaus and slightly drops at L22-23 (~480). Not a clean monotone.
- **R² vs nn_recall tension holds:** auto again selected alpha=1000 for all layers;
  alpha=1 gives substantially better nn_recall@5 (L23: 0.688 vs 0.570).
- **Pythia outperforms GPT-2 on nn_recall at high alpha**: Pythia L23 α=1000:
  nn_recall=0.570 vs GPT-2 L11 α=1000: nn_recall=0.517.

**Cross-model summary:**

| Property | GPT-2 (12L) | Pythia-410m (24L) |
|---|---|---|
| L0 singularity at low α | Yes (∞ cond.) | No (~150M cond.) |
| Erank at L0, α=1 | 200 | 354 |
| Erank at last layer, α=1 | 422 | 480 |
| Erank profile (α=1) | Monotone increasing | Rise → plateau, slight drop |
| Erank profile (α=1000) | Monotone increasing | L0 peak, L1 dip, then rise |
| Best nn_recall (α=1) | L11: 0.686 | L23: 0.688 |
| RidgeCV selection | α=1000, all layers | α=1000, all layers |

### Known limitations

- No images generated yet from either 5000-sample run. Visual claims are
  projections from metric data only.
- `generate` phase with non-GPT-2 models was blocked by a `model_id` kwarg bug
  (now fixed in `experiment.py:_load_sd`).
