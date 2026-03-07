# Slicer Roadmap: Geometric Tomography of LLM Representations

## Context

Slicer projects LLM hidden-state activations into CLIP embedding space via trained Ridge regression, then renders the projection as an image using frozen Stable Diffusion. This is geometrically equivalent to taking a single linear X-ray of the LLM's representation body through the lens of a vision-language model.

This roadmap extends the project toward a principled multi-projection tomographic approach: using multiple interpreted target spaces to reconstruct progressively more of the LLM's internal geometry. The key theoretical grounding is that each target space's training objective defines what its projection reveals and what it's orthogonal to, so adversarial (complementary) projections are chosen for maximal interpretive leverage, not random coverage.

---

## Phase 0: Effective Rank Analysis (Foundation — everything depends on this)

### What and why

The effective rank of the trained Ridge map W tells you how many dimensions of LLM activation space the projection meaningfully uses. This is the single most important diagnostic for the entire project. Without it, you don't know whether your microscope has a 5-dimensional aperture or a 50-dimensional one, and all downstream decisions (corpus design, target space selection, null space analysis) are ungrounded.

### Implementation

After training a Ridge map W (shape: d_CLIP × d_LLM), compute its SVD:

```python
U, S, Vt = np.linalg.svd(W, full_matrices=False)
```

Compute effective rank (Roy & Vetterli, 2007):

```python
def effective_rank(S):
    """Shannon-entropy-based effective rank of a matrix given its singular values."""
    # Normalize singular values to a probability distribution
    p = S / S.sum()
    # Filter zeros to avoid log(0)
    p = p[p > 0]
    # Effective rank = exp(entropy)
    H = -np.sum(p * np.log(p))
    return np.exp(H)
```

Store the full singular value spectrum S per layer alongside the existing projection files. The erank is a scalar summary, but the full spectrum shape is informative (see Visualizations below).

### What to compute

- `erank(W)` for each layer's Ridge map (in per_layer mode)
- `erank(W)` for the mixed_layer map
- The full singular value spectrum per layer
- The condition number `S[0] / S[-1]` per layer (how ill-conditioned the map is)

### Output

Add to the existing manifest/dashboard:
- A plot of erank vs. layer number
- A heatmap of singular value spectra (layers × singular value index)
- Summary statistics in the manifest JSON

### Gotchas

- **Numerical precision**: Ridge with very small α can produce near-zero singular values that are numerical noise, not signal. Use a threshold relative to S[0] (e.g., S[i] > 1e-6 * S[0]) to separate signal from noise before computing erank.
- **α affects erank directly**: Ridge regularization suppresses small singular values. The erank of the Ridge solution is *not* the same as the erank of the unregularized least-squares solution. This is fine — you want the erank of the actual map you're using — but be aware that changing α changes erank. Report α alongside erank.
- **RidgeCV may pick different α per layer**: If using `--alpha auto`, each layer's map may have different regularization. This is correct behavior but means erank differences across layers reflect both genuine geometric differences AND regularization differences. To disentangle: also compute erank at a fixed α across all layers as a control.

---

## Phase 1: Corpus Optimization via Activation Dispersion

### Depends on: Phase 0 (need erank as the optimization objective)

### What and why

The training corpus determines which directions in LLM activation space the Ridge map learns to use. A corpus of only image captions trains a map that's accurate in the "visual description" subspace but extrapolates poorly when probing abstract concepts. The goal is to find the corpus that maximizes the erank of the trained map — i.e., illuminates the largest possible CLIP-visible subspace.

### Experiment design

Train Ridge maps on several corpus configurations and compare erank:

| Corpus key | Source | Expected character |
|---|---|---|
| `captions_only` | Flickr30k + CC3M | Dense visual, narrow register |
| `wiki_only` | Wikipedia | Encyclopedic, moderate register |
| `stories_only` | TinyStories | Narrative, simple syntax |
| `mixed_current` | Current auto_corpus blend | Moderate diversity |
| `max_diverse` | See below | Maximally diverse |

For `max_diverse`, sample from sources chosen to span distinct *structural* (not topical) axes. Priority dimensions of variation:

- **Syntax**: code, legal text, poetry, dialogue, lists, nested argument
- **Register**: formal academic, colloquial, imperative (recipes/manuals), interrogative
- **Discourse structure**: narrative, expository, argumentative, descriptive, procedural
- **Length/complexity**: fragments, simple sentences, long nested clauses

Concrete sources (all streamable from HuggingFace or easily obtained):

| Source | Structural contribution |
|---|---|
| `bigcode/starcoderdata` (sample) | Code syntax, very different activation patterns |
| `pile-of-law` or FreeLaw | Legal register, long nested clauses |
| `poem_sentiment` or similar | Poetic/compressed syntax |
| OpenSubtitles or DailyDialog | Dialogue, colloquial register |
| `wikiHow` | Imperative/procedural |
| Scientific abstracts (arxiv) | Dense technical register |
| WordNet definitions (already included) | Minimal definitional fragments |

The point is NOT topical coverage. Two Wikipedia articles about different animals probably activate very similar directions. A recipe and a legal brief probably activate very different directions despite both being "text."

### How to find the ceiling

1. Start with the largest, most diverse corpus you can assemble.
2. Train the Ridge map. Record erank.
3. Plot erank vs. n_train (number of training samples) as a learning curve. If the curve flattens, you've hit the ceiling. If it's still rising at your max n_train, you haven't — get more data or more diverse data.
4. Also plot erank vs. number of distinct sources (holding total n_train fixed). If adding a new source type doesn't increase erank, that source is redundant with what you have.
5. The singular value spectrum at convergence directly shows the ceiling: it's where the spectrum drops to the noise floor. Count the singular values above the noise threshold — that's N, the dimensionality of the CLIP-visible subspace.

### Activation dispersion as a direct metric

Independent of the Ridge map, you can measure the effective rank of the activation covariance matrix X X^T itself. This tells you how many directions the corpus activates, before any projection. If erank(X X^T) is low, no Ridge map can achieve high erank — the corpus isn't spanning enough of the space.

```python
# After extracting activations X (shape: n_samples × d_LLM) for a given layer
cov = X.T @ X / len(X)
S_cov = np.linalg.svdvals(cov)
erank_activations = effective_rank(S_cov)
```

This gives you a corpus quality metric that's independent of the target space.

### Output

- Table: corpus config → erank per layer
- Learning curves: erank vs. n_train for each corpus
- Activation dispersion (erank of X X^T) per corpus per layer
- Identification of the ceiling N (if reached)

### Gotchas

- **Corpus size confound**: A larger corpus will generally produce higher erank simply because it has more samples, not because it's more diverse. Always compare at matched n_train. The learning curve is essential — you need to see saturation, not just a higher number.
- **Tokenization effects**: Different text types produce different numbers of tokens. "Last token activation" means different things for a 3-token fragment vs. a 500-token paragraph. Consider standardizing input length or using mean-pooled activations as a control.
- **OOM on large corpora**: Computing SVD of the full covariance matrix for large d_LLM can be expensive. Use truncated SVD (e.g., `scipy.sparse.linalg.svds`) if d_LLM > 2048 and you only need the top-k singular values/vectors.
- **Activation scale varies by layer**: Early and late layers may have very different activation magnitudes. Normalize activations per layer (e.g., L2 normalize) before computing covariance, or the erank will be confounded with scale.

---

## Phase 2: Null Space Analysis

### Depends on: Phase 0 (need SVD of W), Phase 1 (need a well-trained map with known erank)

### What and why

The null space of the Ridge map W is the subspace of LLM activations that maps to zero in CLIP space — everything the microscope can't see. If erank is N and d_LLM is D, the null space has dimension D - N. For GPT-2 (D=768) with an expected N of 30-80, this is the majority of the space.

### Implementation

From the SVD of W = U S V^T:

```python
# V^T rows are right singular vectors
# First N rows (large singular values) = CLIP-visible subspace
# Remaining rows = null space

threshold = 1e-6 * S[0]  # or a more principled cutoff
n_visible = np.sum(S > threshold)

V_visible = Vt[:n_visible]    # shape: n_visible × d_LLM
V_null = Vt[n_visible:]       # shape: (d_LLM - n_visible) × d_LLM

# Project activations into visible and null components
def decompose(activation, V_visible, V_null):
    """Split an activation into CLIP-visible and null components."""
    visible = V_visible.T @ (V_visible @ activation)
    null = V_null.T @ (V_null @ activation)
    return visible, null
```

### Key analyses

- **Null space energy ratio**: For each probe input, what fraction of the activation's L2 norm lives in the null space? `||null|| / ||activation||`. If this is consistently > 0.8, the microscope sees less than 20% of the signal.
- **Null space structure across layers**: Does the null space energy ratio change across layers? Hypothesis: it increases in later layers (more abstract, less visual).
- **Null space similarity**: Do semantically similar inputs (e.g., "cat" and "dog") have similar null components, or do they diverge? If they diverge, the null space encodes discriminative information that CLIP collapses. If they're similar, CLIP captures the relevant distinctions.
- **Null space vs. visible space clustering**: Run a simple clustering (k-means or similar) on the visible components and separately on the null components of a set of probe inputs. Do they produce the same clusters? Different clusters = the null space organizes information differently from the visible space.

### Output

- Per-layer null space energy ratio (scalar, plottable across layers)
- Per-probe-pair: visible similarity vs. null similarity scatter plot
- Clustering comparison: adjusted Rand index between visible-space and null-space clusterings

### Gotchas

- **The threshold for "null" is a real choice**: Too aggressive and you include noisy directions in the visible subspace. Too conservative and you leak real visible directions into the null space. Use the singular value spectrum shape to inform this — look for a gap or elbow.
- **Null space is relative to the Ridge map, not absolute**: A different Ridge map (different α, different corpus) has a different null space. Always report which map the null space is computed from.

---

## Phase 3: Multi-Target Projections (The Tomographic Core)

### Depends on: Phase 0 + 1 (need well-trained CLIP map with known erank), Phase 2 (need null space decomposition)

### What and why

Each target space provides a different X-ray of the LLM representation body. By projecting into multiple interpreted spaces, you accumulate complementary slices. The key principle: choose each successive target to maximize the erank of the *residual* — what previous targets couldn't see.

### Target spaces and their interpretive meaning

| Target | Model | Embedding dim | What it captures | What it misses |
|---|---|---|---|---|
| **CLIP** (baseline) | `openai/clip-vit-large-patch14` | 768 | Visual-semantic grounding: objects, scenes, visual attributes | Abstract reasoning, logical structure, syntax, non-visual semantics |
| **SigLIP** | `google/siglip-base-patch16-224` | 768 | Similar to CLIP but different training (sigmoid loss, no softmax). Different geometric biases. | Similar gaps to CLIP, but disagreements with CLIP are informative |
| **DINOv2** | `facebook/dinov2-base` | 768 | Visual structure without language supervision. Texture, shape, spatial layout organized by visual similarity, not textual description. | Anything linguistic. No text training at all. |
| **CLAP** | `laion/larger_clap_general` | 512 | Audio-semantic grounding: sounds, music, acoustic events | Anything non-acoustic |
| **Sentence-BERT** | `sentence-transformers/all-MiniLM-L6-v2` | 384 | Textual semantic similarity. Meaning as captured by paraphrase/entailment structure. | Perceptual grounding of any kind. Syntax (by design — it's meaning-focused). |
| **Instructor** | `hkunlp/instructor-large` | 768 | Task-conditioned text embeddings. Can be prompted for different embedding behaviors. | Perceptual grounding. |

### Implementation strategy

For each target space T:

1. Extract T-embeddings for the training corpus (same corpus used for CLIP map).
2. Train a Ridge map W_T from LLM activations to T-space.
3. Compute SVD, erank, singular value spectrum of W_T.
4. Compute principal angles between the column spaces of W_CLIP and W_T:

```python
def principal_angles(W1, W2, k=None):
    """Compute principal angles between column spaces of two matrices.
    
    Returns angles in radians. 0 = perfectly aligned, π/2 = orthogonal.
    """
    U1, _, _ = np.linalg.svd(W1, full_matrices=False)
    U2, _, _ = np.linalg.svd(W2, full_matrices=False)
    if k is not None:
        U1 = U1[:, :k]
        U2 = U2[:, :k]
    cos_angles = np.linalg.svd(U1.T @ U2, compute_uv=False)
    cos_angles = np.clip(cos_angles, -1, 1)
    return np.arccos(cos_angles)
```

5. For targets that have a natural decoder (CLIP → SD, CLAP → audio diffusion), render the projection. For targets without decoders (sentence-BERT), analyze the projected embeddings directly (nearest neighbors in the target space, clustering, etc.).

### Sequential residual protocol

This is the core tomographic procedure. Order matters.

1. Start with CLIP (highest expected mutual information with language).
2. Compute null space of W_CLIP.
3. Project training corpus activations into the CLIP null space.
4. Train W_CLAP (or next target) on the *null-projected* activations, not the originals.
5. This forces W_CLAP to only use directions that CLIP couldn't see.
6. Repeat: compute joint null space of [W_CLIP, W_CLAP], project, train next target on the residual.

```python
def project_to_null_space(X, Vt_visible):
    """Project activations into the null space of the visible subspace."""
    # Remove the visible component
    visible_component = X @ Vt_visible.T @ Vt_visible
    return X - visible_component
```

### Cumulative visible dimensionality

After k projections, the total visible subspace is the span of all k column spaces. Track:

```python
# Stack the top singular vectors from each map
V_all = np.vstack([Vt_visible_clip, Vt_visible_clap, ...])
# Compute rank of the combined subspace
S_combined = np.linalg.svdvals(V_all)
cumulative_erank = effective_rank(S_combined)
```

Plot cumulative_erank vs. number of target spaces. This is the tomographic reconstruction curve — it shows how much of the LLM's geometry you can see as you add more X-ray directions.

### Rendering for non-CLIP targets

- **DINOv2**: No natural text→image decoder. Instead, render the CLIP projection and the DINOv2 projection side by side by finding the nearest images in a reference image dataset (e.g., use the DINOv2 embedding as a query into an image retrieval index). Alternatively, train a separate Ridge map from DINOv2 space to CLIP space and chain: LLM → DINOv2 → CLIP → SD. This adds a second projection but lets you reuse the SD decoder.
- **CLAP**: Audio diffusion models exist (e.g., AudioLDM) and can be conditioned on CLAP embeddings. The pipeline would be LLM → Ridge → CLAP → AudioLDM → audio. This produces *sounds* instead of images, which is a genuinely different modality of interpretation.
- **Sentence-BERT**: No perceptual decoder. Analyze via nearest-neighbor retrieval in a sentence corpus. For a given LLM activation, the projected sentence-BERT vector's nearest neighbors tell you "what sentences this activation is most similar to" in meaning-space.

### Output

- Per-target: erank, singular value spectrum, per-layer plots
- Cross-target: principal angle matrices (which targets see similar vs. orthogonal subspaces)
- Cumulative visible dimensionality curve
- Side-by-side renderings where decoders are available
- Nearest-neighbor tables where decoders aren't available

### Gotchas

- **Embedding dimension mismatch**: Different targets have different d. The Ridge maps will have different shapes. This is fine for erank and null space analysis but means you can't directly compare singular values across targets — normalize by the target dimension or compare spectra by rank order.
- **Target space quality varies**: CLIP is very well-trained on massive data. Smaller models (CLAP, some sentence transformers) may have noisier or less structured embedding spaces. A low erank for a target might mean "the target space isn't rich enough" rather than "the LLM doesn't encode this modality." Control for this by checking the target model's own internal quality metrics.
- **Sequential residual is order-dependent**: The residual after removing CLIP then CLAP is different from removing CLAP then CLIP. Start with the highest-erank target (CLIP, almost certainly) and proceed in decreasing erank order. This is the greedy strategy — each step removes the most informative remaining subspace.
- **Chained projections accumulate error**: LLM → DINOv2 → CLIP → SD involves two Ridge maps in series. Each has its own approximation error. The output may be dominated by error rather than signal. Always compare against direct LLM → CLIP → SD to check whether the chained version adds real information or just noise.
- **GPU memory**: Running multiple large embedding models simultaneously may exceed VRAM. Process targets sequentially, caching activations to disk.

---

## Phase 4: Diffusion Prior Quantification

### Depends on: Phase 0 (can run in parallel with Phases 1-3)

### What and why

When SD generates an image from a CLIP embedding, it fills in everything the embedding underdetermines using its generative prior. Some of what you see in the output is signal from the LLM; some is hallucination from SD. Quantifying this separation is essential for trustworthy interpretation.

### Implementation

For each probe input and layer:

1. Generate N images (N ≥ 16, ideally 32-64) with different random seeds, same CLIP vector.
2. Compute per-pixel variance across the N images.
3. The variance map is your confidence mask: low variance = determined by the projection, high variance = filled in by the diffusion prior.

```python
from PIL import Image
import numpy as np

def diffusion_confidence_map(images: list[np.ndarray]) -> np.ndarray:
    """Compute per-pixel variance across multiple generations from the same CLIP vector.
    
    Returns a variance map (H × W) normalized to [0, 1].
    Low values = high confidence (signal from LLM).
    High values = low confidence (diffusion prior).
    """
    stack = np.stack(images, axis=0).astype(float)  # (N, H, W, 3)
    # Variance across seeds, averaged over color channels
    var = stack.var(axis=0).mean(axis=-1)  # (H, W)
    # Normalize
    if var.max() > 0:
        var = var / var.max()
    return var
```

### Additional analyses

- **Signal-to-prior ratio**: For each probe, report `1 - mean(variance_map)` as a scalar confidence score. Higher = more of the image is determined by the LLM projection.
- **Confidence vs. layer**: Does the signal-to-prior ratio change across layers? Hypothesis: middle layers may have higher confidence (richer, more specific representations) than early (too generic) or very late (too compressed toward next-token prediction) layers.
- **Confidence vs. erank**: Across probe inputs, does higher erank correlate with higher signal-to-prior ratio? It should — more dimensions used means the CLIP vector is more specified.

### Output

- Per-probe confidence maps (as overlays on the generated images, or as separate heatmaps)
- Signal-to-prior ratio per layer (line plot)
- Dashboard integration: option to show confidence overlay on any generated image

### Gotchas

- **SD is not deterministic even with fixed seed across different hardware/precision**: Use the same device, same dtype, same scheduler for all seeds in a comparison set.
- **SDXL Turbo with CFG=0 has lower variance by design**: The guidance-free mode produces less diverse outputs, so variance will be lower. This doesn't mean confidence is higher — it means the prior is more constrained. Compare within a single SD model, not across models.
- **N matters**: With N=4 seeds, variance estimates are very noisy. N=32 is a reasonable tradeoff. For quick iteration, N=8 with the caveat that confidence maps will be noisy.

---

## Phase 5: Singular Vector Stability Analysis

### Depends on: Phase 1 (need maps trained on multiple corpora)

### What and why

If the top singular vectors of W are the same regardless of training corpus, they represent a robust structural alignment between LLM and CLIP that the corpus choice doesn't affect. If they differ, the microscope is measuring the corpus as much as the LLM.

### Implementation

For each pair of Ridge maps trained on different corpora (same layer, same LLM, same target):

```python
def subspace_similarity(Vt1, Vt2, k):
    """Cosine of principal angle between top-k subspaces.
    
    Returns values in [0, 1]. 1 = identical subspaces, 0 = orthogonal.
    """
    angles = principal_angles_from_Vt(Vt1[:k], Vt2[:k])
    return np.cos(angles)
```

Compute this for k = 1, 5, 10, 20, ... up to erank. The profile tells you: the top-1 direction is always stable (probably), the top-5 are mostly stable, but beyond top-20 the directions are corpus-dependent.

The crossover point — where stability drops below some threshold — is the number of "structural" alignment directions vs. "corpus-dependent" directions.

### Output

- Stability profile: subspace similarity vs. k, for each corpus pair, per layer
- Heatmap: layer × k with stability values

---

## Visualization Strategies

### Singular value spectrum plot

For each layer, plot singular values on a log scale. Look for:
- **Elbow/gap**: Clear separation between signal and noise. The elbow location is N.
- **Smooth decay**: No clear gap. N is ambiguous; use erank as a soft estimate.
- **Layer comparison**: Overlay spectra from all layers on one plot (different colors). Do early/late layers have different spectral shapes?

### Erank dashboard

- Line plot: erank vs. layer (one line per corpus, or per target space)
- This is the single most informative plot in the project. It tells you where in the network the LLM→target alignment is richest.

### Principal angle heatmap

- Matrix: target spaces × target spaces, colored by mean principal angle
- Diagonal is 0 (self-alignment). Off-diagonal values near π/2 mean the targets see orthogonal subspaces. Values near 0 mean they're redundant.

### Cumulative reconstruction curve

- X-axis: number of target spaces (in order of addition)
- Y-axis: cumulative visible erank
- Shape tells you: diminishing returns (each new target adds less) vs. linear growth (each new target sees genuinely new structure)

### Layer-sweep with confidence overlay

- Extend existing layer-sweep animations to include a confidence map overlay (semi-transparent red = high variance / low confidence)
- This immediately shows which visual features are "real" per layer

### Null space energy waterfall

- Stacked bar chart per layer: fraction of activation energy in CLIP-visible, CLAP-visible, sentence-BERT-visible, and residual (no target can see) subspaces
- This is the tomographic reconstruction summary — how much of the LLM's representation each X-ray captures

---

## General Warnings

### On linear maps and their limits

Ridge regression is the right choice for now — it preserves source geometry and its properties are analytically tractable (SVD, null space, erank all have clean interpretations). Do NOT replace it with a nonlinear map (MLP, etc.) for interpretability work. A nonlinear map would produce prettier images but would make the geometric analysis meaningless — you couldn't distinguish structure in the LLM from structure introduced by the map.

Exception: if you want to explore what nonlinear structure CLIP *could* extract from LLM activations (as a ceiling on extractable information), train an MLP map alongside the Ridge map and compare. But never use the MLP map's outputs for geometric analysis.

### On the interpretation of rendered images

The images are projections, not reconstructions. Two activations that produce similar images may be very different in the null space (Busemann-Petty obstruction — cross-section comparisons are unreliable in dimensions ≥ 5, and LLM spaces are d ≥ 768). Never claim two representations are "similar" based on visual similarity of their rendered projections alone. Always check null space similarity too.

### On target space biases

Every target space has biases from its training data and objective. CLIP is biased toward web-scraped image-text pairs (objects, scenes, stock photo aesthetics). CLAP toward AudioSet-style sound events. Sentence-BERT toward NLI/paraphrase pairs. These biases determine what the projection "sees" — they're features, not bugs, but they need to be reported. The microscope has a color filter and that filter is part of the measurement.

### On activation extraction

- **Last-token activations are context-dependent**: The activation for "cat" depends on what precedes it. For controlled experiments, use consistent prompt templates (e.g., always "The concept of {word}" or always the bare word). Document the template.
- **Padding tokens**: OPT and some other models require left-padding. The current codebase handles this, but verify that the "last real token" extraction is correct for any new LLM added to the pipeline.
- **Layer indexing**: Some HuggingFace models index layers starting from 0 (embedding layer), others from 1 (first transformer block). Off-by-one errors here silently corrupt layer-sweep analyses. Always verify by checking activation dimensions — the embedding layer has d_model dimensions but hasn't been through any transformer blocks.

### On numerical stability

- SVD of large matrices can be slow and memory-intensive. For d_LLM > 2048, consider randomized SVD (`sklearn.utils.extmath.randomized_svd`) with n_components set to ~2× your expected erank.
- Erank is sensitive to very small singular values. Always threshold before computing entropy.
- When computing principal angles, clip cosines to [-1, 1] before arccos to avoid NaN from floating point drift.

---

## Suggested Implementation Order

```
Phase 0 (erank analysis)
    │
    ├──→ Phase 4 (diffusion prior quantification) — can run in parallel
    │
    ▼
Phase 1 (corpus optimization)
    │
    ├──→ Phase 5 (singular vector stability) — uses Phase 1 outputs
    │
    ▼
Phase 2 (null space analysis)
    │
    ▼
Phase 3 (multi-target projections)
```

Phase 0 is prerequisite for everything. Phase 4 is independent and can run anytime after Phase 0. Phases 1→2→3 are the critical path. Phase 5 branches off Phase 1.

### Minimum viable results before proceeding

- **Before Phase 1**: Must have erank per layer for the current default corpus. If erank < 5 everywhere, something is wrong with the map training — debug before proceeding.
- **Before Phase 2**: Must show that erank varies meaningfully across corpora (i.e., corpus choice matters). If erank is identical for captions-only and max-diverse, the ceiling is already reached and corpus optimization is moot — skip to Phase 2.
- **Before Phase 3**: Must have a well-trained map with known erank on a well-chosen corpus, and must have verified that the null space contains substantial energy (> 50% of activation norm). If the null space is tiny, multi-target projections add little — the CLIP map already sees most of the space (which would itself be a surprising and publishable finding).
