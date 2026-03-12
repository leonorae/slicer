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

| Mode | Class | Description | Status |
|------|-------|-------------|--------|
| `per_layer` | `LayerProjectionSet` | One `LinearProjection` per layer (default) | **Active — all experiments use this** |
| `single_layer` | `LinearProjection` | One map from a chosen layer | Redundant: `per_layer` already trains every layer; looking at the last-layer row gives identical information |
| `mixed_layer` | `LinearProjection` | One map trained on all layers pooled | Excluded from planned experiments: pooling all layers creates a map with no specific computational referent, making SVD/erank analysis geometrically ambiguous |

`single_layer` and `mixed_layer` remain supported in code but are not used in the current experiment suite.

`alpha="auto"` → `_auto_alpha()` → `RidgeCV` with grid `[0.01, 0.1, 1, 10, 100, 1000, 10000]`.

**Alpha ceiling note:** RidgeCV previously always selected alpha=1000, hitting the top of the old grid `[…, 1000]`.  The grid was extended to 10000 so that `auto` can find its true optimum.  At alpha=1000 GPT-2 L11 R²=0.339 — W is still non-trivial, not yet at collapse.  Experiments 1 and 3 now include alpha=10000 explicitly to show the full trajectory toward projection collapse (W→0, all probes converge to corpus CLIP centroid).

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
│     ├── probe_corpus_distances         # d_act, d_clip per (probe, proj, layer)
│     ├── probe_text_stats               # n_tokens, perplexity per probe slug
│     └── seed_variance                  # mean_pixel_var per (proj, probe, layer, cfg)
├── dashboard.html
├── configs/config_{ts}.json
├── projections/{proj_key}/
├── grids/by_projection/{proj_key}/{text_slug}/
│   ├── per_layer/L{N}_CFG{v}_seed{s}.png
│   └── grid_seed{s}.png
└── .probe_cache/{slug}/layer_{N}.npy
```

## Experiment phases

| Phase | What it does |
|-------|-------------|
| `train` | Fit all (projection_type × alpha) combinations |
| `generate` | Render all (proj × probe × layer × CFG × seed) images; also computes corpus-distance metrics (`d_act`, `d_clip`), probe text stats (`n_tokens`, `perplexity`), and per-group seed variance |
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
| **Prompt length** | **Moderate (Exp 3)** | Unusual tier probes are systematically longer ("the feeling of almost remembering" = 7 tokens vs "a cat" = 3). Last-token activation depends on context length. Controlled by `n_tokens` in `probe_text_stats`. |
| **LLM pretraining corpus distance** | **Moderate (cross-model)** | GPT-2 and Pythia assign different perplexities to the same probe reflecting WebText vs Pile coverage differences. Controlled by `perplexity` in `probe_text_stats`. |
| **Diffusion prior variance** | **Moderate (all LPIPS)** | Some LPIPS signal between alpha=1 and alpha=1000 images could reflect different amounts of diffusion-prior noise rather than genuine CLIP-vector differences. Controlled by `seed_variance` in manifest: compare variance at alpha=1 vs alpha=1000 for the same probe. |
| **CFG scale** | **Fixed parameter** | CFG=7.5 is held constant in Exp 1–3. **Updated standard: CFG=25 for Exp 4+.** Rationale: at CFG=7.5 Pythia's highest probes (cosine_dist ≈ 0.07) sit only 0.09 LPIPS above the noise floor (~0.43); amplification math [claude, 2026-03-12] estimates CFG≈14 as the minimum to reach LPIPS=0.60, CFG=25 gives comfortable margin. Hypothesised ceiling around CFG=30–40 where SD images become fully saturated and LPIPS measures saturation artefacts rather than conditioning content [claude, 2026-03-12]. CFG acts as a gain on the conditioning signal — high CFG amplifies CLIP vector differences, low CFG adds prior noise. Not a confound for within-experiment comparisons (same CFG for all alpha values), but limits generalisability of absolute LPIPS values to other CFG settings. |
| **nn_recall@5 as proxy for probe discriminability** | **Proxy validity** | nn_recall is computed on corpus validation data; it does not predict behaviour on out-of-distribution probes |
| **Cross-model LPIPS comparison** | **Moderate** | GPT-2 and Pythia-410m have different pretraining corpora; the same probe text may sit at different distances from each model's internal distribution |
| **SVD metrics (erank, condition_number, n_visible)** | **None** | Computed on the projection matrix W alone; probe texts are not involved |
| **RidgeCV ceiling effect** | **None** | Property of RidgeCV optimising R² on corpus data |
| **R² vs nn_recall tension** | **None** | Both are corpus-level training metrics |

#### Corpus-distance metrics (implemented)

All four metric groups below are computed automatically during the `generate` phase.

**`manifest["probe_corpus_distances"][slug][proj_key][layer]`** — per (probe, proj, layer):

- **`d_act`** — normalised L2 distance of the probe LLM activation from the corpus
  centroid in activation space.  Uses each layer's z-score parameters (`scaler_mean`,
  `scaler_scale`) as corpus statistics.  Equivalent to a spherical Mahalanobis distance.
  A value > 3 indicates a likely out-of-distribution probe.

- **`d_clip`** — cosine distance of the probe's projected CLIP vector from the corpus
  CLIP centroid.  Available only for projections trained after this metric was added
  (requires `corpus_clip_centroid.npy` alongside the projection weights).

**`manifest["probe_text_stats"][slug]`** — per probe text:

- **`n_tokens`** — number of tokens the probe text produces under the LLM's tokenizer.
  Controls for the prompt-length confound in Exp 3 (unusual probes are systematically longer).

- **`perplexity`** — LLM perplexity of the probe text (exp of mean negative log-likelihood).
  Proxy for distance from the LLM's *pretraining* corpus.  Addresses the cross-model confound:
  GPT-2 and Pythia assign different perplexities to the same probe reflecting WebText vs Pile
  coverage differences.

**`manifest["seed_variance"]["{proj_key}/{slug}/L{N}/CFG{v}"]`** — per (proj, probe, layer, CFG):

- **`mean_pixel_var`** — mean per-pixel variance across all seed images.  Low variance = the
  CLIP conditioning vector determines image content; high variance = diffusion prior is filling
  in detail.  Compare `mean_pixel_var` at alpha=1 vs alpha=1000 for the same probe to determine
  whether LPIPS differences reflect genuine conditioning differences or prior-noise differences.
  Computed when ≥ 4 seed images exist for the group.

- **`n_seeds`** — number of seed images that contributed to the variance estimate.

These metrics let you:
- Test whether LPIPS differences within Exp 3 survive after controlling for `d_act`
  (partial correlation: LPIPS ~ tier | d_act)
- Flag probes with `d_act` > 3 as out-of-distribution and interpret their results with a caveat
- Check whether the tier ordering in Exp 3 simply recapitulates corpus distance
- Distinguish "LPIPS is high because CLIP vectors differ" from "LPIPS is high because SD prior noise differs" using `mean_pixel_var`
- Control for prompt length (`n_tokens`) and pretraining-corpus membership (`perplexity`) in cross-tier or cross-model comparisons

**`manifest["probe_clip_vectors"][proj_key][slug][layer_idx]`** — per (proj, probe, layer):

- The projected CLIP vector as a Python list of floats.  Stored during the `generate`
  phase (free, since the vector is already computed for SD conditioning).  Enables
  cosine distance computation between projections at different alpha values without
  reloading the LLM or SD.  This is the **primary discriminability metric** for Exp 1
  and Exp 3: `cosine_distance(proj_α1(act), proj_α1000(act))` is CFG-independent and
  directly answers whether compression moved the CLIP vector.

**Sanity check:** For "a cat", `d_act` should be lower than for "the color of Tuesday"
across all layers and both models.  If it is not, the corpus centroid distance is not
capturing expected structure and warrants investigation.

#### Why LPIPS is the wrong primary metric, and the CFG sensitivity floor

LPIPS adds three uncontrolled steps between the quantity of interest (CLIP vector
differences) and the measurement: SD rendering, diffusion prior noise, and AlexNet's
perceptual model.  The actual question — does alpha compression destroy information in
the CLIP projection? — is answerable directly in CLIP space.

**CFG sensitivity floor mechanics:**
CFG applies as `ε_out = ε_uncond + s·(ε_cond − ε_uncond)`.  The effective deviation
from unconditional generation scales linearly with CFG.  This means LPIPS is roughly a
function of `(CLIP cosine distance × CFG)`, not CLIP distance alone.  Below some
threshold CLIP distance δ, even CFG=7.5 cannot amplify the difference above the
seed-variance noise floor — LPIPS measures noise, not signal.  δ is unknown without
sweeping both CFG and CLIP distance, which is a separate experiment.

**Consequence for Exp 1 and Exp 3:**  it is not known in advance whether the probe
pairs of interest (e.g. alpha=1 vs alpha=1000 for "a cat") produce CLIP distances above
or below δ.  If all probes are below δ, LPIPS tier ordering is entirely noise.  The
scatter of LPIPS vs. CLIP cosine distance (implemented in the notebook) characterises
δ empirically: if the scatter is flat, the images are not a reliable diagnostic at
CFG=7.5.

**Metric hierarchy:**
1. `cosine_distance(proj_α1(act), proj_α1000(act))` — primary.  CFG-independent,
   SD-independent.  Stored in `manifest["probe_clip_vectors"]`.
2. LPIPS — secondary confirmation.  Valid only if it correlates with (1).  Seed variance
   is its noise floor.
3. Images — illustrative only.  Cannot be used as primary evidence.

#### Multicollinearity between d_act and compression displacement

The partial correlation `cosine_distance(α1, α1000) ~ tier | d_act` assumes that
`d_act` and cosine distance have residual variance not accounted for by corpus distance
alone.  This may not hold.

**Geometric argument:** the compression displacement is
`(W_α1 − W_α1000) · (act − μ)`, which scales proportionally with `‖act − μ‖`
(i.e. d_act) whenever `(W_α1 − W_α1000)` has approximately uniform singular values.
Given n_visible ≈ 767/768 and a smooth spectrum for both models, this is likely.
If `r(d_act, cosine_dist) > 0.9`, the partial correlation has very little residual
variance to work with — low statistical power even if the tier effect is real.

**The ratio alternative:** `cosine_distance(α1, α1000) / d_act` controls for corpus
distance by construction without requiring residual variance.  Under the null (corpus
distance is the complete explanation), this ratio is constant across tiers.  A tier
ordering in the ratio is genuine — it cannot be explained by d_act alone.

**When to prefer the ratio over partial correlation:**
- Check the scatter of d_act vs. cosine_dist (implemented in the notebook).
- If `|r| > 0.9`: use the ratio; partial correlation is uninformative.
- If the scatter passes through the origin: proportional normalisation (ratio) is
  correct.  If there is a meaningful non-zero intercept: use regression residuals instead.
- The coefficient of variation of the ratio across tier means quantifies the effect
  size: CoV < 0.15 ≈ flat (corpus distance explains everything); CoV > 0.15 suggests
  a tier effect beyond corpus coverage.

**Null result framing:** a flat ratio is not a failure — it means alpha compression is a
pure corpus-coverage filter, operating proportionally on every probe regardless of
semantic content.  This is itself informative about the geometry of the Ridge projection.

#### What the confound does NOT invalidate

Even if Exp 3 LPIPS differences are entirely explained by corpus distance, that finding
is itself scientifically meaningful: it shows that alpha compression acts as a
**corpus-coverage filter** — probes in densely covered regions are stable under alpha
changes, while probes in sparse regions are compressed toward the mean.  The semantic
framing ("common" vs "unusual") would need to be re-labelled as "corpus-central" vs
"corpus-peripheral", but the visual discriminability pattern would still hold.

#### Architecture differences as confounders (not controlled)

The table below maps every key empirical finding to its model scope.  Any result
marked "GPT-2 only" or "Pythia only" should not be generalised until replicated on
at least one additional model.

| Observation | Scope | Likely architectural cause | Generalises? |
|---|---|---|---|
| L0 rank-deficiency at low alpha | **GPT-2 only** | Weight tying between embedding and unembedding matrices; GPT-2 ties input and output embeddings, which may constrain the activation manifold at L0 | Unknown |
| Finite, stable L0 condition numbers | **Pythia only** | Pythia does not tie embeddings; separate initialisation gives L0 activations a full-rank covariance | Unknown |
| Erank increases monotonically (α=1) | **GPT-2 only** | Smooth accumulation of representational capacity layer by layer | Not confirmed for Pythia |
| Non-monotone erank at α=1000 (L0 peak, L1 dip) | **Pythia only** | Alpha differentially shrinks the embedding layer (different activation scale) vs. transformer layers; artefact of high regularisation, not semantic bottleneck | Unknown |
| Erank plateau + slight drop at L22–23 (α=1) | **Pythia only** | Possible late-layer representational compression; only observed at α=1 | Unknown |
| RidgeCV selects α=1000 for every layer | **Both models** | RidgeCV optimises R², which prefers high regularisation; this is a property of the scoring function, not the LLM | Likely universal |
| R² vs nn_recall@5 pull in opposite directions | **Both models** | Structural conflict between reconstruction fidelity and topology preservation; not model-specific | Likely universal |
| n_visible ≈ 767/768 at layers ≥ 1 | **Both models** | CLIP space is well-covered by both models' activations once any transformer processing has occurred | Plausibly general |
| nn_recall@5 increases with layer depth (α=1) | **Both models** | Later layers encode more compositional/semantic content that aligns better with CLIP's training objective | Plausibly general |

**Architectural candidates for the L0 differences:**

- *Weight tying* (GPT-2) forces the output of the embedding layer to live in the same
  subspace as the vocabulary embedding matrix.  If that matrix is low-rank relative to
  the hidden dimension, the L0 activation covariance inherits that rank deficiency.
  Pythia uses separate matrices, so no such constraint applies.

- *Initialisation scale* differs between architectures.  If GPT-2's L0 activations
  have a near-degenerate covariance structure (many very small eigenvalues), then at
  low alpha the Ridge estimator overfits those directions, producing apparent singularity
  in the condition number.

- *Pretraining corpus distribution* cannot be ruled out.  GPT-2 was trained on WebText;
  Pythia on The Pile.  Different corpus statistics may produce different activation
  covariance structures at L0 independent of weight tying.

**What is NOT controlled:**

- All SVD metrics compare models trained on different corpora with different
  tokenizers.  Observed differences in erank or condition number could reflect corpus
  statistics rather than architecture.
- L0 is the embedding layer in both models, but its role differs: GPT-2's tied
  embeddings mean L0 directly reflects vocabulary statistics; Pythia's untied L0 does
  not.  Comparing L0 across models is comparing qualitatively different computational
  stages.
- Depth is not matched: GPT-2 has 12 layers, Pythia-410m has 24.  "Last layer"
  comparisons conflate layer depth with depth as a fraction of total network depth.
- `generate` phase with non-GPT-2 models was blocked by a `model_id` kwarg bug
  (now fixed in `experiment.py:_load_sd`).

---

## Observed results — Exp 1 & 3 (GPT-2 L11, Pythia-410m L23, 5000-sample corpus)

> **Status:** Primary metric (CLIP cosine distance) computed. LPIPS now run (Exp 3
> bar charts). Both confirm the CFG sensitivity floor failure mode — see LPIPS section below.
> Corpus quality is acknowledged as a limitation throughout — see caveat at end of section.

### Alpha compression plateau (both models)

The α=1000 vs α=10000 cosine distances are roughly 1/5th the α=1 vs α=10000 distances
for every probe and both models.  Nearly all compression happens in the α=1 → α=1000 step;
α=10000 adds little beyond α=1000.  The α=10000 condition is largely redundant for image
generation purposes.  This is consistent with the erank plateau observed in the SVD analysis.

### Collinearity results

| Model | r(d_act, cosine_dist) | Interpretation |
|---|---|---|
| GPT-2 L11 | 0.998 | Driven entirely by one outlier (democracy, d_act ≈ 1100). Partial correlation is uninformative. |
| Pythia-410m L23 | 0.700 | Below the 0.9 threshold. Partial correlation has some residual power. |

### GPT-2 L11 — key findings

- **Democracy is a pathological outlier.** d_act ≈ 1100 vs all other probes < 100 —
  a >10× separation.  Its absolute cosine displacement (≈ 0.45) dominates the abstract tier
  mean entirely.  This is not semantic unusualness; it is corpus-coverage absence.
  "Democracy" does not appear in Flickr30k/CC3M (image-caption sources); the Ridge map
  has no training signal for it and places its activation far from the corpus centroid.
- **The abstract tier is incoherent for GPT-2.** With democracy as an outlier and only
  ~3 probes per tier, the abstract mean and variance are unreliable.  The tier cannot be
  interpreted as a category.
- **After normalisation (ratio), the ordering is unusual > abstract > concrete** (CoV = 0.37,
  above the 0.15 threshold).  Democracy's ratio (≈ 0.00041) is lower than the unusual tier
  mean — meaning democracy is *less* compression-sensitive per unit corpus distance than the
  unusual probes, despite its extreme absolute displacement.  The unusual tier being highest
  in the ratio is the result that survives the confound.
- **The r=0.998 line is not a genuine proportionality relationship.** Remove democracy and the
  correlation would be near zero.  Do not interpret this as evidence that corpus distance
  mechanistically explains compression sensitivity across the probe set.

### Pythia-410m L23 — key findings

- **d_act is nearly constant across all probes** (range 27.5–31.5, ≈ 4-unit spread across 11
  probes).  For Pythia, the corpus-distance confound barely exists — all probes sit at
  roughly equal distance from the corpus centroid in activation space.  Controlling for d_act
  changes almost nothing.
- **Ordering: unusual > abstract > concrete** in raw cosine distance and in ratio (CoV = 0.16,
  marginally above the 0.15 threshold).
- **The narrow d_act range makes Pythia the cleaner test of the hypothesis.** The variation in
  cosine distance across probes is not explained by variation in corpus distance (because there
  is almost no variation in corpus distance).  The unusual tier having higher cosine displacement
  than concrete is therefore attributable to something other than corpus peripherality.
- **"entropy" (abstract) and "the color of Tuesday" (unusual) are both near the top of the
  cosine distance distribution**, with "a tree" and "a house" at the bottom.  This is
  qualitatively consistent with the tier hypothesis but the within-tier variance is large
  relative to the between-tier differences.

### Cross-model summary

| Finding | GPT-2 L11 | Pythia-410m L23 |
|---|---|---|
| Compression plateau at α ≈ 1000 | Yes | Yes |
| r(d_act, cosine_dist) | 0.998 (outlier-driven) | 0.700 |
| d_act range across probes | 20–1100 (heterogeneous) | 27.5–31.5 (homogeneous) |
| Ratio ordering | unusual > abstract > concrete | unusual > abstract > concrete |
| Ratio CoV | 0.37 | 0.16 |
| Dominant confound | Democracy outlier (corpus absence) | Weak — d_act nearly constant |
| Abstract tier reliability | Low (incoherent due to outlier) | Moderate |

**The one consistent finding:** unusual > concrete in the compression ratio holds across both
models and is not explainable by d_act alone in either case.  The abstract tier is noisy in
both models (incoherent in GPT-2; moderate variance in Pythia).

**What is NOT confirmed:** the full unusual > abstract > concrete tier ordering is not
established with sufficient reliability for either model at current probe n and corpus size.

### Corpus quality caveat

The 5000-sample corpus draws heavily from image-caption sources (Flickr30k, CC3M).  Probes
with no visual referent — "democracy", "justice", and to some extent all abstract/unusual
tier probes — are structurally underrepresented.  This means:

- d_act for abstract/unusual probes reflects **corpus composition**, not semantic properties
  of the probes.  Democracy's extreme d_act (≈ 1100) is a corpus artifact, not a finding
  about GPT-2's representation of political concepts.
- The tier classification conflates "semantically unusual" with "corpus-peripheral".  For
  Pythia this conflation is less severe (narrow d_act range), but the underlying cause is
  unclear — it may reflect Pythia's pretraining on The Pile (broader coverage) rather than
  a genuine geometric property.
- **None of these results should be considered publication-ready.** The probe set is small
  (≈ 3 per tier), the corpus is thin for non-visual concepts, and the between-tier differences
  are small relative to within-tier variance.  The current runs are exploratory and diagnostic.

### LPIPS results (Exp 3 bar charts, both models)

> **Scope:** Pythia-410m L23 and GPT-2 L11, all three alpha pairs (α=1 vs 1000, α=1 vs
> 10000, α=1000 vs 10000), tiers: concrete / abstract / unusual.

**Main finding: LPIPS is flat across all tiers and both models.**

All six panels cluster between **0.40–0.55 LPIPS** with heavily overlapping within-tier
error bars (n≈3 per tier). The predicted ordering (unusual > abstract > concrete) is not
confirmed in any panel.

**The abstract tier is the lowest, not the middle, in most panels.** This is the
democracy-outlier / projection-collapse effect: democracy is in the collapse regime
(d_act ≈ 1100, output → SD prior noise), and prior noise looks similar *across* alpha
values — producing low LPIPS, not high. The tier is incoherent.

**The α=1000 vs α=10000 column is no smaller than the α=1 vs α=1000 column.**
The compression plateau that was clear in CLIP cosine distance (~5× smaller) is
invisible in LPIPS. LPIPS is too noisy to resolve the signal.

**Interpretation: LPIPS is measuring diffusion prior noise, not conditioning signal.**
These cosine distances are below the CFG=7.5 sensitivity floor (threshold δ from the
confounder analysis).  CFG=7.5 cannot amplify the CLIP-vector differences into the
perceptual regime — the images generated at α=1 and α=1000 differ by prior noise alone.
This is exactly the failure mode predicted before the images were run.

**Consequence for the metric hierarchy:**
- CLIP cosine distance (primary metric) — valid.  Shows compression plateau, shows
  unusual > concrete in the normalised ratio for Pythia.
- LPIPS (secondary) — **not valid as evidence here**.  r(LPIPS, cosine_dist) is
  effectively zero; the scatter is flat.  Images cannot be used to confirm or deny
  the tier hypothesis at CFG=7.5.
- Images — illustrative only; do not examine as primary evidence.

**What would fix this:** sweep CFG (e.g. 3, 7.5, 15, 30) to find the threshold δ above
which CLIP cosine distances of this magnitude become perceptually visible.  Alternatively,
compute LPIPS on the extreme-alpha pairs at the layer sweep level once those images exist
— earlier layers may produce larger cosine displacements that exceed δ.

### LPIPS vs. cosine distance scatter (sensitivity floor characterisation)

**GPT-2 L11 (r=0.93): single-point artefact.**
Every probe except democracy clusters at cosine_dist < 0.1, LPIPS 0.38–0.50 — a flat
cloud.  Democracy sits at cosine_dist ≈ 0.45, LPIPS ≈ 0.61, dragging r to 0.93.  Remove
democracy and the correlation is near zero.  r=0.93 is *not* evidence that LPIPS reliably
tracks cosine distance: democracy's extreme outlier status is doing all the work.
Crucially, democracy *does* produce elevated LPIPS — the method is not broken, but the
sensitivity threshold δ is somewhere above cosine_dist ≈ 0.1, and all other probes are
far below it.

**Pythia L23 (r=0.49): weak genuine correlation, one anomaly.**
Probes spread across 0.02–0.075 cosine distance (consistent with homogeneous d_act).
r=0.49 means LPIPS does partially track cosine distance for Pythia.  The unusual probes
at the far right ("the color of Tuesday", "entropy at midnight") have both the highest
cosine distances and the highest LPIPS (~0.52–0.53).  However, "the feeling of almost
remembering" sits anomalously low — moderate cosine distance but the lowest LPIPS among
unusual probes (~0.38).  This probe's CLIP vector moves in a direction SD does not
differentiate perceptually, even though the geometric shift is real.

**Sensitivity floor estimate (both models):**
LPIPS ~0.40–0.43 is the empirical floor — below this, diffusion prior noise dominates.
Even at Pythia's maximum cosine_dist ≈ 0.07, LPIPS only reaches ~0.53.  The δ threshold
appears to require cosine_dist ≳ 0.1–0.3 at CFG=7.5.  This is achieved only by extreme
d_act probes (democracy at 0.45) or by running at earlier layers with larger displacements.
The Pythia unusual probes are approaching but not clearly above the floor.

**What the scatter resolves:**
- **r(LPIPS, cosine_dist) for GPT-2 is outlier-driven; for Pythia it is weak (0.49).**
  Neither constitutes validation of LPIPS as a reliable secondary metric for these probes.
- **Direction of CLIP movement matters, not just magnitude.**  "The feeling of almost
  remembering" has real cosine displacement but low LPIPS — its compression shift is
  orthogonal to SD's perceptually sensitive axes.  Cosine distance is necessary but not
  sufficient to predict perceptual change.
- **δ estimate constrains the CFG sweep design.**  To confirm the tier ordering visually,
  either raise CFG until Pythia's 0.07 displacements exceed δ, or augment the probe set
  with higher d_act probes in the ordinary-compression regime (not collapse like democracy).

### Open questions for Exp 1 before moving on

- **LPIPS vs. cosine distance scatter** — now run; see section above.
- **Layer sweep for compression sensitivity** — all current results are last-layer only.
  Whether compression sensitivity (ratio) increases with layer depth (consistent with the
  erank trajectory) is unexamined.  Would require running generate across all layers, which
  is not in the current Exp 1 config.
- **Democracy as a control probe** — its d_act ≈ 1100 makes it useful as a high-corpus-distance
  control, but it should not be analysed as a representative abstract probe.  Either exclude
  it from the abstract tier or add it to a separate "corpus-absent" category with d_act > 500.
- **CFG sweep** — superseded. CFG raised to 25 for Exp 4+ rather than running a full sweep. Amplification math [claude, 2026-03-12] indicates CFG=25 should push Pythia's highest probes (cosine_dist ≈ 0.07) clearly above the LPIPS floor; democracy at d_act≈1100 serves as calibration that the method is working. A sweep (3, 7.5, 15, 25, 30) could still characterise δ precisely but is deferred as lower priority than the layer sweep.

---

## Planned experiments

Three experiments in priority order, each with a config file ready to run.
All need full re-runs (train + generate + grids) due to expanded design.

### Exp 1 — Alpha compression visibility

**Goal:** Validate that nn_recall@5 predicts discriminability under alpha compression.
**Hypothesis:** alpha=1 produces more distinct CLIP projections per probe than alpha=1000.
The primary evidence is geometric; the images are secondary illustration.

**Design:**
- Models: GPT-2 (last layer L11) and Pythia-410m (last layer L23) — run separately
- Alpha: 1, 1000, and 10000 (to show full collapse trajectory)
- Probe set: concrete + abstract (see configs)
- CFG: 7.5 (fixed; acts as gain on conditioning — see confounder table)
- Seeds: 16 (for seed-variance estimation; Phase 4 metric)
- Layers: last layer only (`layers: [11]` for GPT-2, `[23]` for Pythia)

**Analysis (primary):** `cosine_distance(proj_α1(act), proj_α1000(act))` per probe,
stored in `manifest["probe_clip_vectors"]`. CFG-independent. Tests whether alpha
compression moves the CLIP vector and whether this differs by probe tier.

**Analysis (secondary):** LPIPS(alpha=1 image, alpha=1000 image) as a downstream
visibility check. Only meaningful if it correlates with CLIP cosine distance — a flat
scatter means CFG=7.5 is below the sensitivity floor for these probes. Seed variance
is the noise floor below which LPIPS cannot distinguish conditioning from prior.

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

## Observed results — Exp 2 visual analysis (Pythia-410m, 5 seeds per condition)

> **Scope note:** All visual findings in this section are from **Pythia-410m only**.
> None have been replicated on GPT-2 or any other model. GPT-2 Exp 2 was not run
> (L0 rank-deficiency at low alpha makes it unsuitable). Do not generalise.

> **Seed counts:** 5 seeds shown per condition. Full dataset is 16 seeds per
> (probe × alpha). Conclusions below are provisional pending full-set confirmation.

### L2 is an inter-seed convergence point at alpha=1

Across two probes ("a cat", "the color of Tuesday"), L2 shows tighter cross-seed
agreement than adjacent layers. For "the color of Tuesday" L2 alpha=1, all 5 seeds
produced nearly identical warm amber diagonal wave/weave textures — the strongest
single-layer convergence observed in any condition. For "a cat" L2 alpha=1, seeds
converged on a loose grey/white vertical-flow family (weaker but still the tightest
within that probe's layer sweep). Convergence at L2 is probe-specific (different
textures for different probes, not all collapsing to the same point). L1 and L3 are
more variable in both cases. This is not predicted by the erank trajectory and has no
current geometric explanation.

### The L0→L1 transition is probe-dependent

For "beauty" alpha=1 (from Exp 2 grid, single seed): L0 was near-blank; L1 introduced
dramatic high-contrast snake-scale structure. The token embedding alone contributed
almost no visual information; the first transformer layer inserted structure entirely.

For "a cat" alpha=1: L0 already had structured fragmented content. L1 changed its
character but did not amplify it from near-zero. The embedding layer for a common
concrete token has a non-trivial projection; the embedding layer for an abstract token
may not.

This difference is consistent with the hypothesis that common tokens have well-defined
embedding-space projections into CLIP space, while abstract tokens begin near the
corpus centroid and acquire visual identity only after transformer computation.
**Caveat:** single-seed observations from the Exp 2 grid; not confirmed at seed level.

### Compression stabilises unusual probes more than concrete probes at L23

"The color of Tuesday" L23 alpha=1: bimodal (yellow/purple geometric maze OR
black/sandy blobs — two coherent attractor groups across 5 seeds).
"The color of Tuesday" L23 alpha=1000: unimodal (all 5 seeds → same horizontal amber
banded pattern). Compression resolved the bimodal split.

"A cat" L23 alpha=1: five distinct visual types, no clustering — genuinely high
variance with no dominant attractor.
"A cat" L23 alpha=1000: still five distinct visual types — compression did not resolve
cat's L23 variability.

**Interpretation:** Compression moves projections toward the corpus centroid, which maps
to a well-defined SD attractor (warm amber brickwork/architecture). Unusual probes sit
far from the centroid and are pushed toward it, gaining coherence. Concrete probes
("a cat") may sit in a broad, diverse region of CLIP space where even the corpus
centroid maps to a high-entropy SD neighbourhood — many plausible image types remain
after compression. The concrete probe does not gain visual coherence from compression
because its compressed destination is itself ambiguous to SD.

**Important caveat:** this is the opposite of what the Exp 3 hypothesis predicted
(concrete probes should be alpha-*insensitive*, not alpha-incoherent). The finding
is about seed variance, not cosine distance. High seed variance at alpha=1000 is not
the same as high cosine distance between alpha=1 and alpha=1000.

### Two qualitatively different compression failure modes (GPT-2 observation)

From GPT-2 Exp 3 results: "democracy" (d_act ≈ 1100) produces images visually
indistinguishable from SD null noise (no-prompt / nonsense-prompt output). Other
probes with moderate d_act produce recognisable corpus-like textures under compression.

This identifies two failure modes:
1. **Ordinary compression** (moderate d_act): W·(act − μ) shrinks; output → bias b →
   corpus centroid → recognisable corpus-like textures (warm amber, fur, landscape).
2. **Projection collapse** (extreme d_act, e.g. democracy): the Ridge map has zero
   training signal for this activation region; even the bias term b is unreliable;
   output → near-zero CLIP conditioning → SD prior noise.

The threshold between these modes defines the effective coverage radius of the Ridge
projection. It is estimable from GPT-2 data by ordering probes by d_act and finding
where output transitions from "corpus texture" to "prior noise." Democracy at d_act ≈
1100 is in the collapse regime; all other current probes appear to be in the ordinary
compression regime.

**Consequence for Exp 3:** The GPT-2 abstract tier mixes both failure modes ("beauty"
→ ordinary compression, "democracy" → projection collapse). The tier is incoherent
not only because n is small but because it contains qualitatively different projection
behaviours. Democracy should be treated as a high-d_act diagnostic control, not an
abstract-concept probe.

### Limitations of current Exp 2 visual results — to fix in layer sweep design

The following limitations apply to all visual findings above. A planned layer sweep
experiment should address them:

| Limitation | Effect | Fix |
|---|---|---|
| Only 5 of 16 seeds shown | Convergence/divergence conclusions are provisional | Run full 16-seed analysis; compute mean_pixel_var from manifest |
| Only L0, L1, L2, L3, L23 sampled | L2 convergence peak is inferred, not traced | Sweep all 24 layers to find the actual variance minimum |
| Only two probes compared ("a cat", "the color of Tuesday") | Cross-probe conclusions have n=2 | Add remaining Exp 3 probes to the layer sweep |
| Single alpha per condition for cross-alpha comparison | Compression effect at intermediate layers uncharacterised | Run alpha=1 and alpha=1000 for all layers, not just last layer |
| No quantitative seed variance metric used | Visual judgement of coherence is subjective | Use `mean_pixel_var` from manifest to find convergence layer objectively |
| Pythia only | No cross-model confirmation | Replicate on GPT-2 once L0 rank issue is handled (use alpha ≥ 100 for L0) |
| Exp 2 grid images (single seed) used alongside seed-stack images | Single-seed observations mixed with multi-seed observations | Separate claims made from single-seed vs. multi-seed data |

**The key open question the layer sweep addresses:** Is L2 convergence a genuine local
minimum in seed variance, and does it hold across all probes? The current data shows
it for two probes but cannot determine whether it is a property of Pythia's early-layer
geometry or an artefact of the specific probes chosen.

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
- Alpha: 1, 1000, and 10000
- Models: both (cross-model check is the key result)
- Layers: last layer of each model
- Seeds: 16 (same set as Exp 1; enables seed-variance comparison across tiers)

**Analysis (primary):** `cosine_distance(proj_α1(act), proj_α1000(act))` per probe,
grouped by tier. Tests the tier hypothesis in CLIP space directly without SD or CFG.
Partial correlation against `d_act` (corpus-distance confound) and `n_tokens`
(prompt-length confound) determines whether the tier ordering is genuine.

**Analysis (secondary):** LPIPS(alpha=1, alpha=1000) per probe, with scatter against
CLIP cosine distance to verify it tracks the geometric signal. If the scatter is flat,
LPIPS is below the CFG=7.5 sensitivity floor and the images cannot be used as evidence.

**Confound to watch:** low-variance directions in the *projection* ≠ semantically
unusual prompts — they could be directions with sparse corpus coverage (`d_act`).
A tier ordering in cosine distance that survives partial correlation against `d_act`
is genuine; one that disappears is a corpus-coverage artefact.

**Configs:** `experiment_config_exp3_gpt2.json`, `experiment_config_exp3_pythia.json`
**Commands:**
```bash
python run_experiment.py --config experiment_config_exp3_gpt2.json --phase generate
python run_experiment.py --config experiment_config_exp3_gpt2.json --phase grids
python run_experiment.py --config experiment_config_exp3_pythia.json --phase generate
python run_experiment.py --config experiment_config_exp3_pythia.json --phase grids
```

---

### Exp 4 — Full layer sweep (Pythia-410m, seed variance trajectory)

**Goal:** Find where in Pythia's layer hierarchy seed variance is minimised (the
convergence layer), and whether L2 as a local minimum holds across the full probe set.
Secondarily, produce per-layer activation cache for all probes to enable direct
comparison with upcoming novel projection metrics.

**Background:** Exp 2 visual analysis (2 probes, 5 seeds, L0–L3 + L23 only) suggested
L2 as an inter-seed convergence point.  That observation is unconfirmed: only two probes,
only 5 seeds shown, only 5 of 24 layers sampled.

**Model: Pythia-410m only.**  GPT-2 L0 is rank-deficient at alpha=1 (unusable without
complicating the sweep with alpha≥100 for that layer).  Pythia has 24 layers (richer
trajectory), homogeneous d_act across probes (corpus-distance confound is weak), and
is the target model for upcoming novel projection metrics work.

**Design:**
- Model: `EleutherAI/pythia-410m`
- Layers: all 24 (L0–L23)
- Alpha: 1 only [alpha=1000 deferred; doubles compute; cross-alpha question partially
  answered at L23 by Exp 1/3; add if signal is found]
- CFG: 25 (new standard; see CFG scale row in confounder table)
- Seeds: 16 (consistent with Exp 1/3; needed for reliable `mean_pixel_var`)
- Probes: full Exp 3 set (all 9: concrete + abstract + unusual tiers)
  — democracy included as high-d_act control; treated separately, not as abstract-tier member
- Corpus: 5000-sample, same sources as Exp 1/3 (comparability with prior per-layer SVD
  results and Exp 3 compression ratios; convergence layer is more a network-structure
  property than corpus-dependent)
- track_lpips: False (seed variance is the primary output; LPIPS too noisy)

**Primary metric:** `manifest["seed_variance"][key]["mean_pixel_var"]` per (probe, layer).
Plot: layer (x) vs. mean_pixel_var (y), one line per probe.  Find the layer of minimum
variance and whether it is consistent across probes.

**Secondary:** `manifest["probe_clip_vectors"]` per (probe, layer) — enables cosine
distance between adjacent layers (layer-to-layer CLIP drift) and comparison with novel
projection metrics on the same axis.

**Coordination with novel projection metrics:** The `generate` phase populates
`.probe_cache/{slug}/layer_{N}.npy` for all 24 layers and all 9 probes.  Novel
projection methods should read from this cache (same slugs, same layer indices) to
avoid re-running the LLM.  The `probe_clip_vectors` stored in `manifest` serve as
a Ridge-projection baseline for direct comparison against whatever new metric produces.

**Config:** `experiment_config_exp4_pythia_layersweep.json`
**Commands:**
```bash
python run_experiment.py --config experiment_config_exp4_pythia_layersweep.json --phase train
python run_experiment.py --config experiment_config_exp4_pythia_layersweep.json --phase generate
python run_experiment.py --config experiment_config_exp4_pythia_layersweep.json --phase grids
```

**Key question:** Is L2 a genuine local minimum in mean_pixel_var across all probes,
or was it an artefact of the two probes in Exp 2?  A consistent minimum at L2 across
concrete, abstract, and unusual tiers would be a strong signal; probe-specific minima
at different layers would suggest the convergence point is semantics-dependent.
