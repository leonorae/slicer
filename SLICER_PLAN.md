# Slicer Implementation Plan — Phase 0

Scope: **Phase 0 (Effective Rank Analysis)** only.
`experiment.py` will be **rewritten** with a clean architecture designed to
accommodate future phases rather than patched.

---

## Reuse Map

| Module | Action | Reason |
|---|---|---|
| `clip_bridge.py` | **Keep as-is** | LLM activation and CLIP extraction are correct and reusable |
| `training_data.py` | **Keep as-is** | Existing sources are sufficient for Phase 0 |
| `generator.py` | **Keep as-is** | Image generation unchanged for Phase 0 |
| `projection.py` | **Extend** | Add SVD computation + storage after `Ridge.fit()`; add properties |
| `pipeline.py` | **Keep as-is** | Used by `run_microscope.py` quick-run path; leave untouched |
| `experiment.py` | **Rewrite** | Design for analytical phases from the start |

New file: `diffusion_microscope/analysis.py`

---

## Features Outside Phase 0 (Flagged for Reassessment)

These exist in the current codebase but are not in scope for Phase 0. They are
left in place but not actively maintained during this work.

| Feature | Where | Status |
|---|---|---|
| SDXL / Turbo support | `generator.py`, `clip_bridge.py`, old `experiment.py` | Out of scope; new runner may not carry it over |
| Mixed-layer projection mode | `projection.py`, old `experiment.py` | Unclear geometric interpretation under SVD/erank analysis |
| Layer-sweep GIF animations | old `experiment.py` | Not in roadmap; dropped from new runner |
| Linear interpolation sequences | `generator.py` | Not in roadmap |
| LPIPS metrics | old `experiment.py` | Superseded by erank and confidence maps (future phases) |
| `geometric_viz/` PCA/UMAP tool | `geometric_viz/` | Separate; unaffected |

---

## 1. `diffusion_microscope/projection.py` — Extend

**Current:** `LinearProjection.fit()` trains Ridge and stores
`W.npy`, `b.npy`, `scaler_mean.npy`, `scaler_scale.npy`, `meta.json`.

**Changes:**

After `self.W` is assigned in `fit()`, immediately compute SVD:
```python
U, S, Vt = np.linalg.svd(self.W, full_matrices=False)
self._S  = S    # shape (min(clip_dim, llm_dim),)
self._Vt = Vt   # shape (min(...), llm_dim) — for Phase 2
```

In `save()`, write two additional files per projection directory:
- `S.npy` — singular value spectrum (needed for erank computation)
- `Vt.npy` — right singular vectors (needed for null space in Phase 2)

In `load()`, load both if present (silently skip if absent, for back-compat
with any existing saved projections).

Add three read-only properties:
```python
@property
def singular_values(self) -> np.ndarray: ...        # returns self._S

@property
def erank(self) -> float: ...                        # calls analysis.effective_rank

@property
def condition_number(self) -> float: ...             # S[0] / S[-1], guarded
```

`meta.json` gains two new keys at save time: `"erank"` and `"condition_number"`.
These are scalars so they're human-readable in the manifest without loading .npy.

No other changes to `LinearProjection` or `LayerProjectionSet`.

---

## 2. `diffusion_microscope/analysis.py` — New

Single module containing all Phase 0 mathematical functions.
No model loading. No disk I/O. Pure numpy.

```python
def effective_rank(S: np.ndarray) -> float:
    """Shannon-entropy erank (Roy & Vetterli 2007).

    Filters values below 1e-10 before computing entropy to avoid log(0).
    S should be raw singular values (not squared).
    """

def spectrum_noise_floor(S: np.ndarray, threshold: float = 1e-6) -> int:
    """Return index of first singular value below threshold * S[0].

    This is n_visible — the number of directions above the noise floor.
    Used by Phase 2 null space analysis.
    """
```

These two functions are all that Phase 0 needs. They are defined here (not
inlined in `projection.py`) so Phase 2–5 can import them without pulling in
projection machinery.

---

## 3. `diffusion_microscope/experiment.py` — Rewrite

### Why rewrite instead of extend

The existing file is 1200+ lines of CLIP-specific logic with image generation,
LPIPS, GIF creation, and HTML dashboard tightly coupled. The roadmap's
analytical phases (erank curves, spectrum heatmaps, null space ratios,
principal angle matrices) are sufficiently different in character that adding
them to the existing file produces a harder-to-reason-about result than a
clean design.

### New design

**Core principle:** phases are independent, idempotent functions that read and
write to the manifest. A minimal `ExperimentRunner` class owns the manifest and
output directories; phases are methods on it.

**Manifest schema (Phase 0):**

```json
{
  "config": { ... },
  "layers": [0, 1, 2, ...],
  "projections": {
    "{proj_key}": {
      "type": "per_layer",
      "alpha": 1.0,
      "corpus": "auto",
      "n_train": 5000,
      "save_path": "projections/per_layer_alpha1.0/",
      "validation": {
        "{layer_idx}": {
          "r2": 0.43,
          "cosine_dist_mean": 0.21,
          "nn_recall_at_1": 0.18
        }
      },
      "svd": {
        "{layer_idx}": {
          "erank": 42.3,
          "condition_number": 1240.1,
          "n_visible": 38,
          "spectrum_path": "projections/.../layer_0004/S.npy"
        }
      }
    }
  },
  "images": {
    "{rel_path}": {
      "projection": "per_layer_alpha1.0",
      "probe_text": "a cat on a roof",
      "layer": 4,
      "cfg": 7.5,
      "seed": 42
    }
  },
  "grids": {
    "{rel_path}": {
      "projection": "...", "text": "...", "seed": 0
    }
  }
}
```

**Phases (Phase 0 scope):**

| Phase key | Method | What it does |
|---|---|---|
| `train` | `run_train()` | Fit Ridge maps per layer; store SVD via extended `LinearProjection`; populate `manifest["projections"][key]["svd"]` |
| `generate` | `run_generate()` | Probe image generation (same logic as before, simplified) |
| `grids` | `run_grids()` | Layer × CFG grid PNGs (same as before) |
| `analyze` | `run_analyze()` | Read SVD data from manifest; produce erank-vs-layer plot and spectrum heatmap; save to `{out_dir}/analysis/` |
| `dashboard` | `run_dashboard()` | Write `dashboard.html`; includes erank plots alongside image grids |

**Phases NOT carried over from old `experiment.py`:**
- `metrics` (LPIPS post-hoc) — roadmap supersedes this
- `animations` (GIFs) — not in roadmap
- Symlink `by_text/` directory structure — not worth the complexity

**`run_analyze()` outputs (Phase 0):**
- `analysis/erank_vs_layer_{proj_key}.png` — line plot, one line per projection
- `analysis/spectrum_heatmap_{proj_key}.png` — layers × singular value index,
  log-scale color, with n_visible boundary marked
- `analysis/erank_summary.json` — machine-readable: `{proj_key: {layer: erank}}`

These are also embedded in the updated dashboard.

**`ExperimentRunner` interface:**
```python
runner = ExperimentRunner(config_path="experiment_config.json", out_dir="results/")
runner.run(phases=["train", "analyze"])          # subset of phases
runner.run(phases=["all"])                       # everything
```

Each phase is idempotent: checks manifest before doing work, skips completed
items.

**Config schema changes from old format:**

The existing `experiment_config.example.json` format is preserved for
`projection_types`, `alphas`, `probe_texts`, `cfg_scales`, `seeds`. Two keys
are removed since their phases are dropped: `track_lpips` and `animate`.

---

## Implementation Order

```
1. analysis.py
   └── effective_rank(), spectrum_noise_floor()

2. projection.py
   └── SVD in fit() → _S, _Vt
   └── save/load S.npy + Vt.npy
   └── erank, condition_number, singular_values properties

3. experiment.py (rewrite)
   └── ExperimentRunner skeleton + manifest load/save
   └── run_train() phase
   └── run_generate() + run_grids() phases (ported from old code)
   └── run_analyze() phase (erank plots, spectrum heatmap)
   └── run_dashboard() phase (updated HTML)

4. Tests / smoke-run
   └── python run_experiment.py --config experiment_config.example.json
         --phase train --n_train 200 --model gpt2
   └── python run_experiment.py --phase analyze
   └── Verify erank_vs_layer plot is produced and non-trivial
```

Step 3 is the largest. Steps 1 and 2 are small, self-contained, and can be
done and verified independently before the rewrite begins.

---

## File Change Summary

```
diffusion_microscope/
├── analysis.py       ← NEW  (~40 lines: effective_rank, spectrum_noise_floor)
├── projection.py     ← EXTEND  (SVD in fit/save/load, 3 new properties)
├── experiment.py     ← REWRITE  (clean phases, Phase 0 outputs)
├── clip_bridge.py    ← unchanged
├── training_data.py  ← unchanged
├── generator.py      ← unchanged
└── pipeline.py       ← unchanged
```
