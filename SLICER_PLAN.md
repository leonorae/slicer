# Slicer Implementation Plan

This document maps the roadmap phases to concrete code changes, identifying
what to keep, extend, refactor, or add from scratch.

---

## Reuse Map

| Module | Action | Reason |
|---|---|---|
| `clip_bridge.py` | **Keep as-is** | LLM activation extraction and CLIP embedding extraction are directly reused across all phases |
| `generator.py` | **Extend** | DiffusionMicroscope is reused for Phase 4; needs `diffusion_confidence_map()` added |
| `training_data.py` | **Extend** | Existing sources and loader pattern are reused; Phase 1 requires ~6 new sources |
| `projection.py` | **Extend** | `LinearProjection.fit()` must also compute/store SVD; properties for erank and null space needed |
| `pipeline.py` | **Refactor** | `prepare_training_data()` needs to accept arbitrary target extractors (not just CLIP) |
| `experiment.py` | **Refactor** | Manifest schema must grow for SVD data, multi-target, corpus experiments, confidence maps |
| `geometric_viz/` | **Leave** | Separate tool, unrelated to roadmap; highlighted below for reassessment |

New modules to create:
- `diffusion_microscope/analysis.py` — all geometric math
- `diffusion_microscope/target_spaces.py` — non-CLIP target extractors

---

## Features Outside the Roadmap (For Reassessment)

These exist in the current codebase but are not discussed in the roadmap. They
are left in place but should be reviewed before any further investment.

| Feature | Where | Note |
|---|---|---|
| **SDXL / SDXL Turbo support** | `generator.py`, `clip_bridge.py`, `experiment.py` | Roadmap focuses on SD 1.x quality semantics. SDXL conditioning (zero-padded 2048-dim, pooled EOS) is significant extra surface area. |
| **Mixed-layer projection mode** | `pipeline.py`, `projection.py`, `experiment.py` | Roadmap mentions per-layer only for erank/null-space analysis. Mixed-layer has unclear geometric interpretation under the new framework. |
| **Layer-sweep animations (GIFs)** | `experiment.py` Phase 5 | Not mentioned in roadmap. Useful visualisation but not part of the analysis plan. |
| **Linear interpolation sequences** | `generator.py` | Not in roadmap. Could be valuable but is not a planned experiment. |
| **LPIPS inline/post-hoc metrics** | `experiment.py` | Roadmap replaces raw image similarity with the more principled erank, confidence maps, and principal angles. LPIPS is a redundant perceptual metric here. |
| **geometric_viz/ PCA/UMAP tool** | `geometric_viz/` | Completely separate tool. Does not interact with the roadmap pipeline. |

---

## Module-by-Module Changes

### 1. `diffusion_microscope/projection.py` — Extend

**Current state:** `LinearProjection.fit()` trains Ridge, stores W.npy / b.npy /
scaler params / meta.json. `LayerProjectionSet` wraps one per layer.

**Changes needed:**

`LinearProjection.fit()` — after computing `self.W`, immediately compute SVD
and store the results:

```python
U, S, Vt = np.linalg.svd(self.W, full_matrices=False)
self._S   = S     # (min(clip_dim, llm_dim),)
self._Vt  = Vt    # (min(...), llm_dim) — right singular vectors
```

`LinearProjection.save()` — also write `S.npy` and `Vt.npy` alongside the
existing files.

`LinearProjection.load()` — load S.npy and Vt.npy if present (backward-
compatible: skip if absent).

Add read-only properties:
- `erank` — calls `analysis.effective_rank(self._S)`
- `condition_number` — `S[0] / S[-1]` (guarded against zero)
- `singular_values` — returns `self._S`
- `visible_vt(threshold=1e-6)` — rows of Vt with S > threshold * S[0]
- `null_vt(threshold=1e-6)` — remaining rows

`meta.json` gains two new keys written at save time:
- `erank` — float
- `condition_number` — float

No breaking changes to the existing file layout; new files are additive.

---

### 2. `diffusion_microscope/analysis.py` — New module

Pure-numpy geometric tools. No model loading, no disk I/O.

```
effective_rank(S: ndarray) -> float
    Shannon-entropy-based erank (Roy & Vetterli, 2007).
    Thresholds near-zero values at 1e-10 before computing entropy.

activation_dispersion(X: ndarray, k: int | None = None) -> float
    Effective rank of the activation covariance X^T X.
    Uses randomized SVD when d_LLM > 2048.

decompose_activation(activation, visible_vt, null_vt) -> tuple[ndarray, ndarray]
    Split a single activation into CLIP-visible and null components.

null_energy_ratio(activation, null_vt) -> float
    Fraction of activation L2 norm in the null space.

principal_angles(W1, W2, k=None) -> ndarray
    Principal angles (radians) between column spaces of two matrices.
    Returns shape (min(rank1, rank2),) or (k,) if k given.

subspace_similarity(Vt1, Vt2, k) -> ndarray
    cos(principal_angles) between top-k subspaces. Values in [0,1].

cumulative_visible_erank(list_of_Vt_visible) -> float
    Erank of the span of multiple visible subspaces stacked together.

diffusion_confidence_map(images: list[ndarray]) -> ndarray
    Per-pixel variance (H×W) across N generations, normalized to [0,1].
    Low = signal-determined, high = prior-hallucinated.
```

---

### 3. `diffusion_microscope/target_spaces.py` — New module

One extractor per target space, all sharing the same signature:

```python
def extract_{name}_embeddings(
    texts: list[str],
    device: str = "cpu",
    batch_size: int = 64,
    **kwargs,
) -> np.ndarray:  # (n_texts, d_target)
```

Targets to implement (in roadmap order):

| Function | Model | d |
|---|---|---|
| `extract_siglip_embeddings` | `google/siglip-base-patch16-224` | 768 |
| `extract_dinov2_embeddings` | `facebook/dinov2-base` | 768 |
| `extract_clap_embeddings` | `laion/larger_clap_general` | 512 |
| `extract_sentence_bert_embeddings` | `sentence-transformers/all-MiniLM-L6-v2` | 384 |
| `extract_instructor_embeddings` | `hkunlp/instructor-large` | 768 |

`clip_bridge.extract_clip_text_embeddings()` is the existing CLIP extractor and
does not move — it stays in `clip_bridge.py`.

A `TARGET_SPACES` registry dict maps short names to `(extractor_fn, d_target)`.

---

### 4. `diffusion_microscope/training_data.py` — Extend

Add six new private loaders following the exact same pattern as existing ones:

| Loader | Source | Structural axis |
|---|---|---|
| `_load_starcoder(n)` | `bigcode/starcoderdata` | Code syntax |
| `_load_dialogue(n)` | `daily_dialog` | Colloquial, interrogative |
| `_load_wikihow(n)` | `wikihow` | Imperative / procedural |
| `_load_arxiv(n)` | `togethercomputer/RedPajama-Data-1T` arxiv subset or `scientific_papers` | Dense technical |
| `_load_legal(n)` | `pile-of-law` (FreeLaw) | Nested legal clauses |
| `_load_poetry(n)` | `poem_sentiment` or `poetry_foundation` | Compressed syntax |

Register all new sources in `_LOADERS` and `ALL_SOURCES`.

Add a `CORPUS_PRESETS` dict mapping the corpus names from the Phase 1 table
(`captions_only`, `wiki_only`, `stories_only`, `mixed_current`, `max_diverse`)
to tuples of source keys and optional weights. This lets Phase 1 experiments
reference a preset by name rather than raw source lists.

---

### 5. `diffusion_microscope/generator.py` — Extend

Add `diffusion_confidence_map()` as a standalone function (not a method) after
the `DiffusionMicroscope` class:

```python
def diffusion_confidence_map(
    images: list,          # list of PIL.Image or np.ndarray
    normalize: bool = True,
) -> np.ndarray:
```

This is a thin wrapper around `analysis.diffusion_confidence_map()` that
handles PIL→ndarray conversion, so callers do not need to import analysis
directly.

---

### 6. `diffusion_microscope/pipeline.py` — Refactor

`prepare_training_data()` currently hard-codes CLIP as the target. Refactor its
signature to accept an optional `target_extractor` callable:

```python
def prepare_training_data(
    texts: list[str],
    target_extractor=None,   # defaults to extract_clip_text_embeddings
    cache_tag: str = "clip", # used in cache filenames to avoid collisions
    ...
)
```

The rest of the function is unchanged. Cache filenames become
`{cache_tag}_embeddings.npy` instead of `clip_embeddings.npy`.

`train_layer_projections()` / `train_projection()` / `train_mixed_projection()`
return the same structure as before; the SVD data is transparently added
inside `LinearProjection.fit()`.

---

### 7. `diffusion_microscope/experiment.py` — Refactor

This is the largest change. The manifest schema gains new top-level sections;
existing sections are preserved.

**Extended manifest schema:**

```json
{
  "projections": {
    "{proj_key}": {
      "type": "...",
      "mode": "...",
      "alpha": ...,
      "target_space": "clip",        ← NEW: default "clip"
      "corpus_preset": "mixed_current", ← NEW
      "save_path": "...",
      "validation": { ... },
      "svd": {                        ← NEW per-layer dict
        "{layer_idx}": {
          "erank": 42.3,
          "condition_number": 1240.1,
          "n_visible": 38,
          "spectrum_path": "projections/.../layer_NNNN/S.npy"
        }
      }
    }
  },
  "corpus_experiments": {             ← NEW (Phase 1)
    "{corpus_preset}_{n_train}": {
      "corpus_preset": "...",
      "n_train": 5000,
      "erank_per_layer": { "{layer}": 42.3 },
      "activation_dispersion_per_layer": { "{layer}": 55.1 }
    }
  },
  "confidence_maps": {                ← NEW (Phase 4)
    "{proj_key}/{text_slug}/L{N}_seed_ensemble.npy": {
      "projection": "...", "text": "...", "layer": 0,
      "n_seeds": 32, "signal_to_prior_ratio": 0.72
    }
  },
  "stability": {                      ← NEW (Phase 5)
    "{corpus_pair_key}": {
      "corpus_a": "...", "corpus_b": "...",
      "similarity_profile_per_layer": {
        "{layer}": [0.99, 0.95, 0.87, ...]   // indexed by k=1,5,10,20,...
      }
    }
  },
  "images": { ... },
  "grids": { ... },
  "animations": { ... },
  "metrics_computed": true,
  "layers": [ ... ]
}
```

**New phases to add to `ExperimentRunner` (in addition to existing 6):**

| New Phase | Method | What it does |
|---|---|---|
| `svd_analysis` | `compute_svd_stats()` | Reads stored S.npy per projection/layer; populates `manifest["projections"][key]["svd"]` |
| `corpus_sweep` | `run_corpus_sweep()` | Trains maps on each CORPUS_PRESET × n_train point; records erank and activation dispersion |
| `confidence_maps` | `generate_confidence_maps()` | Generates N-seed ensembles per (proj, text, layer); saves variance maps and SPR |
| `stability_analysis` | `run_stability_analysis()` | Computes subspace similarity across corpus experiment pairs |
| `multi_target` | `run_multi_target()` | Trains Ridge maps to non-CLIP targets; computes principal angles; renders where decoder exists |

All existing phases (train, generate, grids, metrics, animations, dashboard) are
unchanged.

The `run()` orchestration method gains a `phases` parameter (set of strings)
that determines which phases to execute, so users can run only what they need.

---

## Implementation Order

Follows the dependency graph in the roadmap:

```
Step 1 — Foundation
  projection.py: add SVD storage to LinearProjection.fit() / save() / load()
  analysis.py: effective_rank, principal_angles, decompose_activation,
               null_energy_ratio, subspace_similarity, cumulative_visible_erank,
               activation_dispersion, diffusion_confidence_map

Step 2 — Phase 0 wiring (can start as soon as Step 1 is done)
  experiment.py: svd_analysis phase + manifest["projections"][key]["svd"] section
  Visualisations: erank-vs-layer plot, singular value spectrum heatmap

Step 3 — Phase 4 (parallel with Step 2, depends only on Step 1)
  generator.py: diffusion_confidence_map wrapper
  experiment.py: confidence_maps phase + manifest["confidence_maps"] section

Step 4 — Phase 1
  training_data.py: 6 new loaders + CORPUS_PRESETS
  experiment.py: corpus_sweep phase + manifest["corpus_experiments"] section
  Visualisations: erank learning curves, activation dispersion table

Step 5 — Phase 5 (depends on Step 4)
  experiment.py: stability_analysis phase + manifest["stability"] section
  Visualisations: stability-vs-k profile, layer × k heatmap

Step 6 — Phase 2 (depends on Steps 1, 4)
  analysis.py: decompose_activation, null_energy_ratio already exist from Step 1
  experiment.py: null-space metrics integrated into existing image records
  Visualisations: null energy waterfall, visible vs null clustering

Step 7 — Phase 3 (depends on Steps 1, 4, 6)
  target_spaces.py: all 5 extractors
  pipeline.py: target_extractor refactor
  experiment.py: multi_target phase + manifest["projections"] target_space field
  Visualisations: principal angle heatmap, cumulative reconstruction curve,
                  side-by-side renderings, nearest-neighbor tables
```

---

## File Change Summary

```
diffusion_microscope/
├── analysis.py          ← NEW  (geometric math: erank, angles, null space, confidence)
├── target_spaces.py     ← NEW  (SigLIP, DINOv2, CLAP, SBERT, Instructor extractors)
├── projection.py        ← EXTEND  (SVD in fit/save/load, new properties)
├── training_data.py     ← EXTEND  (6 new loaders + CORPUS_PRESETS)
├── generator.py         ← EXTEND  (confidence_map wrapper)
├── pipeline.py          ← REFACTOR  (target_extractor param)
├── experiment.py        ← REFACTOR  (manifest schema, 5 new phases)
├── clip_bridge.py       ← KEEP AS-IS
```

New top-level runners (optional convenience entry points):
- `run_corpus_sweep.py` — CLI for Phase 1 corpus experiments
- `run_analysis.py` — CLI for Phase 0/2/5 analysis without re-generating images
