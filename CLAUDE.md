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

---

## Confounder analysis

### Prompt-corpus membership as a confound

Alpha compression at high regularisation shrinks every projected vector toward the corpus
mean in CLIP space (Ridge drives W → 0 as α → ∞, so the output converges to the bias
term b, which encodes the training corpus centroid).  Probes whose LLM activations are
far from the corpus centroid are moved more by this shrinkage and therefore produce higher
LPIPS(alpha=1, alpha=1000).

The key risk: **corpus distance and semantic unusualness co-vary** by construction in Exp 3.
Common-concrete probes ("a cat", "a dog") are heavily represented in Flickr30k and CC3M.
Unusual probes ("the color of Tuesday") appear nowhere in the corpus.  We cannot separate
the semantic signal from the corpus-coverage signal without an independent measure of each.

#### Which results are specifically confounded

| Result | Confound severity | Notes |
|--------|-------------------|-------|
| **Exp 3** — LPIPS tier ordering (unusual > abstract > concrete) | **High** | Tier labels co-vary with corpus distance by construction |
| **Exp 1** — LPIPS(concrete, abstract) | **Moderate** | Concrete probes over-represented in image-caption sources; abstract probes in definitional sources |
| **nn_recall@5 as proxy for probe discriminability** | **Proxy validity** | nn_recall is computed on corpus validation data; it does not predict behaviour on out-of-distribution probes |
| **Cross-model LPIPS comparison** | **Moderate** | GPT-2 and Pythia-410m have different pretraining corpora; the same probe text may sit at different distances from each model's internal distribution |
| **SVD metrics (erank, condition_number, n_visible)** | **None** | Computed on the projection matrix W alone; probe texts are not involved |
| **RidgeCV ceiling effect** | **None** | Property of RidgeCV optimising R² on corpus data |
| **R² vs nn_recall tension** | **None** | Both are corpus-level training metrics |

#### Corpus-distance metrics (implemented)

Two metrics are now computed per (probe, projection, layer) during the `generate` phase
and stored in `manifest["probe_corpus_distances"][slug][proj_key][layer]`:

- **`d_act`** — normalised L2 distance of the probe LLM activation from the corpus
  centroid in activation space.  Uses each layer's z-score parameters (`scaler_mean`,
  `scaler_scale`) as corpus statistics.  Equivalent to a spherical Mahalanobis distance.
  A value > 3 indicates a likely out-of-distribution probe.

- **`d_clip`** — cosine distance of the probe's projected CLIP vector from the corpus
  CLIP centroid.  Available only for projections trained after this metric was added
  (requires `corpus_clip_centroid.npy` alongside the projection weights).

These metrics let you:
- Test whether LPIPS differences within Exp 3 survive after controlling for `d_act`
  (partial correlation: LPIPS ~ tier | d_act)
- Flag probes with `d_act` > 3 as out-of-distribution and interpret their results with a caveat
- Check whether the tier ordering in Exp 3 simply recapitulates corpus distance

**Sanity check:** For "a cat", `d_act` should be lower than for "the color of Tuesday"
across all layers and both models.  If it is not, the corpus centroid distance is not
capturing expected structure and warrants investigation.

#### What the confound does NOT invalidate

Even if Exp 3 LPIPS differences are entirely explained by corpus distance, that finding
is itself scientifically meaningful: it shows that alpha compression acts as a
**corpus-coverage filter** — probes in densely covered regions are stable under alpha
changes, while probes in sparse regions are compressed toward the mean.  The semantic
framing ("common" vs "unusual") would need to be re-labelled as "corpus-central" vs
"corpus-peripheral", but the visual discriminability pattern would still hold.

#### Architecture differences as confounders (not controlled)

Four cross-model architecture effects are not independently controlled:

1. **L0 singularity is GPT-2-specific** — Pythia-410m has finite, stable condition
   numbers at L0 for every alpha.  Architectural cause unknown; could be embedding
   initialisation, weight tying, or pretraining corpus differences.

2. **Non-monotone erank at alpha=1000 in Pythia** (L0 peak, L1 dip) — only appears
   at high regularisation.  May reflect alpha differentially compressing the embedding
   layer vs. transformer layers due to different activation magnitudes, rather than a
   semantic bottleneck.

3. **RidgeCV ceiling effect is cross-model** — auto → alpha=1000 for every layer in
   both models.  Confirms the effect is a property of RidgeCV + R² scoring, not of a
   specific model.

4. **R² vs nn_recall tension is cross-model** — confirms the conflict is structural,
   not a GPT-2 artifact.
- `generate` phase with non-GPT-2 models was blocked by a `model_id` kwarg bug
  (now fixed in `experiment.py:_load_sd`).

---

## Planned experiments

Three experiments in priority order, each with a config file ready to run.
All use the existing 5000-sample trained projections — only the `generate` and
`grids` phases need running (train phase is already done).

### Exp 1 — Alpha compression visibility

**Goal:** Validate that nn_recall@5 predicts visual discriminability.
**Hypothesis:** alpha=1 images are more distinct per probe than alpha=1000 images.
LPIPS between alpha=1 and alpha=1000 for the same prompt quantifies how much
information the compressor destroys.

**Design:**
- Models: GPT-2 (last layer L11) and Pythia-410m (last layer L23) — run separately
- Alpha: 1 and 1000 (extremes only)
- Probe set: concrete + abstract (see configs)
- CFG: 7.5 only (one value to keep output manageable)
- Seeds: 42, 123, 777 (3 seeds for stability)
- Layers: last layer only (`layers: [11]` for GPT-2, `[23]` for Pythia)

**Analysis:** For each probe, compute LPIPS(alpha=1 image, alpha=1000 image).
If nn_recall@5 is a good proxy for visual discriminability, LPIPS should be
higher for abstract prompts (lower nn_recall at high alpha) than concrete ones.

**Configs:** `experiment_config_exp1_gpt2.json`, `experiment_config_exp1_pythia.json`
**Commands:**
```bash
python run_experiment.py --config experiment_config_exp1_gpt2.json --phase generate
python run_experiment.py --config experiment_config_exp1_gpt2.json --phase grids
python run_experiment.py --config experiment_config_exp1_pythia.json --phase generate
python run_experiment.py --config experiment_config_exp1_pythia.json --phase grids
```

---

### Exp 2 — L0→L1 bottleneck in Pythia

**Goal:** Visually characterise what the first Pythia transformer layer discards.
**Background:** At alpha=1000, Pythia L0 has the highest erank (494) and L1 dips
to 478. Opus interpreted this as a bottleneck-then-expansion architecture. **Caveat:**
this pattern only holds at high alpha; at alpha=1, L0 is the *lowest* erank point.
The experiment tests whether the erank dip is visually significant.

**Design:**
- Model: Pythia-410m (GPT-2 L0 is rank-deficient at alpha=1 — unusable)
- Alpha: 1 only (preserves topology; L0 at alpha=1 is well-conditioned for Pythia)
- Layers: 0, 1, 2, 3 (first four, to see trajectory not just endpoints)
- Same probe set as Exp 1

**Analysis:** Layer-sweep images L0→L3 for each probe. What changes between L0
and L1? Is L0 more diffuse/abstract and L1 more structured? Does the visual
transition match the erank drop?

**Config:** `experiment_config_exp2_pythia.json`
**Commands:**
```bash
python run_experiment.py --config experiment_config_exp2_pythia.json --phase generate
python run_experiment.py --config experiment_config_exp2_pythia.json --phase grids
```

---

### Exp 3 — Alpha sensitivity as novelty detector

**Goal:** Test whether alpha sensitivity correlates with prompt unusualness.
**Hypothesis:** Unusual prompts have distinguishing information concentrated in
low-variance (high-noise) directions that high alpha kills. Common prompts live
near the corpus mean — their projection is insensitive to alpha compression.

**Design:**
- Probe set: three tiers explicitly labelled by expected unusualness:
  - *Common-concrete* (should be alpha-insensitive): "a cat", "a dog", "a house"
  - *Common-abstract* (moderate sensitivity): "democracy", "justice", "beauty"
  - *Unusual* (should be alpha-sensitive): "the feeling of almost remembering",
    "the color of Tuesday", "entropy at midnight"
- Alpha: 1 and 1000
- Models: both (cross-model check is the key result)
- Layers: last layer of each model
- Seeds: 42, 123, 777

**Analysis:** LPIPS(alpha=1, alpha=1000) per probe. Test whether LPIPS ranks
probes by tier: unusual > common-abstract > common-concrete.

**Confound to watch:** low-variance directions in the *projection* ≠ semantically
unusual prompts — they could be directions with sparse corpus coverage. The
experiment tests the surface correlation; a follow-up would compare against
activation-space distance from the corpus centroid as an independent novelty measure.

**Configs:** `experiment_config_exp3_gpt2.json`, `experiment_config_exp3_pythia.json`
**Commands:**
```bash
python run_experiment.py --config experiment_config_exp3_gpt2.json --phase generate
python run_experiment.py --config experiment_config_exp3_gpt2.json --phase grids
python run_experiment.py --config experiment_config_exp3_pythia.json --phase generate
python run_experiment.py --config experiment_config_exp3_pythia.json --phase grids
```
