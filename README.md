# Diffusion Microscope

A research tool that visualises LLM hidden-state activations as images.  It
works by projecting each layer's last-token activation into CLIP embedding space
via a trained linear (Ridge) map, then feeding that vector into a frozen Stable
Diffusion model as conditioning.

```
LLM hidden state  ──▶  Ridge projection  ──▶  CLIP space  ──▶  SD image
  (per layer)           (trained, frozen)      (frozen)         (frozen)
```

The generated images act as a cross-modal "microscope" — differences between
layers, words, and concepts become visible without any fine-tuning.

---

## Installation

```bash
uv sync --extra microscope
uv pip install -e .
```

GPU (CUDA 12.1):
```bash
uv sync --extra microscope --extra-index-url https://download.pytorch.org/whl/cu121
```

---

## Quick start

```bash
# Train a projection on built-in texts and probe one phrase
python run_microscope.py --probe "a cat on a roof"

# Use a larger corpus from HuggingFace datasets (recommended)
python run_microscope.py --auto_corpus --n_train 5000 --probe "democracy"

# Different projection modes
python run_microscope.py --projection_mode single_layer --probe "justice"
python run_microscope.py --projection_mode mixed_layer  --probe "justice"
python run_microscope.py --projection_mode per_layer    --probe "justice"

# Different LLMs
python run_microscope.py --llm_model EleutherAI/pythia-160m --probe "hello"
python run_microscope.py --llm_model facebook/opt-350m      --probe "hello"

# SDXL Turbo (much faster on GPU)
python run_microscope.py --sd_model stabilityai/sdxl-turbo --probe "fire"
```

---

## Full experiment run

Create a config (copy `experiment_config.example.json`) then:

```bash
# Run everything: train → generate → grids → metrics → animations → dashboard
python run_experiment.py --config my_config.json --auto_corpus --n_train 20000

# Run individual phases (all phases checkpoint, safe to re-run)
python run_experiment.py --config my_config.json --phase train
python run_experiment.py --config my_config.json --phase generate
python run_experiment.py --config my_config.json --phase grids
python run_experiment.py --config my_config.json --phase animations
python run_experiment.py --config my_config.json --phase dashboard

# Common overrides
python run_experiment.py --config my_config.json --device cuda --llm gpt2-medium
python run_experiment.py --config my_config.json --no_lpips          # faster generation
python run_experiment.py --config my_config.json --write_sidecars    # per-image JSON metadata
```

Open `experiment_results/dashboard.html` in a browser to browse results.

---

## Projection modes

| Mode | Description |
|------|-------------|
| `per_layer` | One Ridge map per layer — best for layer-by-layer comparison (default) |
| `single_layer` | One map from the last layer — fastest, use `--target_layer N` to pick |
| `mixed_layer` | One map trained on all layers pooled — smooth inter-layer transitions |

`--alpha auto` (default) runs RidgeCV to pick the best regularisation per map.

---

## Supported models

**LLMs** — any `AutoModelForCausalLM` on HuggingFace.  Tested:

| Model | HF ID | Params |
|-------|-------|--------|
| GPT-2 | `gpt2` | 117M — default, fast on CPU |
| GPT-2 Medium | `gpt2-medium` | 345M |
| Pythia-70M | `EleutherAI/pythia-70m` | 70M — tiny, good for testing |
| Pythia-160M … 1B | `EleutherAI/pythia-{160m,410m,1b}` | |
| OPT-125M … 1.3B | `facebook/opt-{125m,350m,1.3b}` | Left-padding handled automatically |
| TinyLlama | `TinyLlama/TinyLlama-1.1B-Chat-v1.0` | 1.1B |
| SmolLM2 | `HuggingFaceTB/SmolLM2-{135M,360M}` | |
| Qwen2-0.5B | `Qwen/Qwen2-0.5B` | |

**Stable Diffusion:**

| Model | HF ID | Notes |
|-------|-------|-------|
| SD 1.5 | `sd-legacy/stable-diffusion-v1-5` | Default |
| SD 2.1 | `stabilityai/stable-diffusion-2-1` | |
| SDXL Turbo | `stabilityai/sdxl-turbo` | ~2 s/img on GPU; auto-detected, 4 steps, CFG=0 |

---

## Training corpus sources

Pulled automatically with `--auto_corpus`.  All sources stream from HuggingFace.

| Key | Dataset | Content |
|-----|---------|---------|
| `flickr30k` | `nlphuji/flickr30k` | Visual captions |
| `wikipedia` | `wikimedia/wikipedia` | Encyclopaedic prose |
| `cc3m` | `google-research-datasets/conceptual_captions` | Web alt-text |
| `wordnet` | NLTK WordNet | Noun definitions |
| `tinystories` | `roneneldan/TinyStories` | Short stories |

---

## Output structure

```
experiment_results/
├── manifest.json                        # Index of every file + metrics
├── dashboard.html                       # Browser-based result viewer
├── configs/config_{ts}.json
├── projections/{proj_key}/              # Saved Ridge maps
├── grids/by_projection/{proj_key}/{text}/
│   ├── per_layer/L{N}_CFG{v}_seed{s}.png
│   ├── grid_seed{s}.png                 # Composite layer × CFG grid
│   └── anim_CFG{v}_seed{s}.gif         # Layer-sweep animation
└── .probe_cache/{slug}/layer_{N}.npy   # Cached LLM activations
```

---

## How it works

1. **Extract** — run the LLM on each training text; take the last real token's
   hidden state at every layer.
2. **Project** — fit a Ridge regression from LLM activation space to CLIP text
   embedding space.  Linearity is intentional: it preserves the geometry of the
   representation space rather than distorting it.
3. **Generate** — feed the projected vector into a frozen SD model as the prompt
   embedding.  The diffusion model is a decoder, not a learner.

The result is that images differ *only* because the underlying LLM representations
differ.  You can directly compare words, layers, or models by looking at images.
