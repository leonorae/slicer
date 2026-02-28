"""
Stratified training-corpus builder for the LLM→CLIP projection.

Pulls from public datasets, mixes them proportionally, deduplicates,
and returns a shuffled list ready for prepare_training_data().

Sources
-------
flickr30k   Flickr 30k Captions  – concrete visual scene descriptions (phiyodr/flickr30k)
wikipedia   Wikipedia (English)  – encyclopaedic / abstract prose
cc3m        Conceptual Captions  – diverse web image alt-text
wordnet     WordNet (via NLTK)   – short definitions of abstract nouns
tinystories TinyStories          – short simple English stories

Each source is loaded in streaming mode so only the required number of rows are
downloaded.  Sources that fail (missing packages, auth errors, network issues)
are skipped with a warning rather than crashing the run.
"""

from __future__ import annotations

import hashlib
import random
import re
from typing import Optional


# Ordered default mix — change weights in load_training_corpus() to override.
ALL_SOURCES = ("flickr30k", "wikipedia", "cc3m", "wordnet", "tinystories")

# Sentence splitter for Wikipedia paragraphs.
_SENT_RE = re.compile(r"(?<=[.!?])\s+")


# ---------------------------------------------------------------------------
# Per-source loaders
# ---------------------------------------------------------------------------

def _load_flickr30k(n: int) -> list[str]:
    from datasets import load_dataset  # type: ignore

    # phiyodr/flickr30k uses plain parquet (no database loading script).
    # Schema: each row has a 'sentences' list of dicts with a 'raw' caption key.
    ds = load_dataset(
        "phiyodr/flickr30k",
        split="train",
        streaming=True,
    )
    texts: list[str] = []
    for row in ds:
        for sent in row.get("sentences") or []:
            cap = (sent.get("raw") if isinstance(sent, dict) else sent or "").strip()
            if cap:
                texts.append(cap)
            if len(texts) >= n:
                return texts
    return texts


def _load_wikipedia(n: int) -> list[str]:
    from datasets import load_dataset  # type: ignore

    ds = load_dataset(
        "wikimedia/wikipedia",
        "20231101.en",
        split="train",
        streaming=True,
    )
    texts: list[str] = []
    for row in ds:
        for sent in _SENT_RE.split(row.get("text", "")):
            sent = sent.strip()
            # Keep medium-length sentences; skip stubs and giant run-ons
            if 30 <= len(sent) <= 300:
                texts.append(sent)
            if len(texts) >= n:
                return texts
    return texts


def _load_cc3m(n: int) -> list[str]:
    from datasets import load_dataset  # type: ignore

    ds = load_dataset(
        "google-research-datasets/conceptual_captions",
        split="train",
        streaming=True,
    )
    texts: list[str] = []
    for row in ds:
        cap = (row.get("caption") or "").strip()
        if cap:
            texts.append(cap)
        if len(texts) >= n:
            return texts
    return texts


def _load_wordnet(n: int) -> list[str]:
    try:
        import nltk  # type: ignore
    except ImportError:
        raise ImportError(
            "nltk is required for WordNet definitions.\n"
            "Install with: pip install nltk"
        )

    # Ensure the corpus data is present.
    try:
        from nltk.corpus import wordnet as wn  # type: ignore
        wn.synsets("test")
    except LookupError:
        nltk.download("wordnet", quiet=True)
        nltk.download("omw-1.4", quiet=True)
        from nltk.corpus import wordnet as wn  # type: ignore

    synsets = list(wn.all_synsets(pos=wn.NOUN))
    random.shuffle(synsets)

    texts: list[str] = []
    for syn in synsets:
        defn = (syn.definition() or "").strip()
        if 15 <= len(defn) <= 300:
            name = syn.name().split(".")[0].replace("_", " ")
            texts.append(f"{name}: {defn}")
        if len(texts) >= n:
            break
    return texts


def _load_tinystories(n: int) -> list[str]:
    from datasets import load_dataset  # type: ignore

    ds = load_dataset(
        "roneneldan/TinyStories",
        split="train",
        streaming=True,
    )
    texts: list[str] = []
    for row in ds:
        story = (row.get("text") or "").strip()
        if not story:
            continue
        # Split into sentences and keep medium-length ones
        for sent in _SENT_RE.split(story):
            sent = sent.strip()
            if 20 <= len(sent) <= 300:
                texts.append(sent)
            if len(texts) >= n:
                return texts
    return texts


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

_LOADERS = {
    "flickr30k": _load_flickr30k,
    "wikipedia": _load_wikipedia,
    "cc3m": _load_cc3m,
    "wordnet": _load_wordnet,
    "tinystories": _load_tinystories,
}


def load_training_corpus(
    n_total: int = 5000,
    sources: tuple[str, ...] = ALL_SOURCES,
    seed: int = 42,
    weights: Optional[dict[str, float]] = None,
) -> list[str]:
    """
    Build a stratified, deduplicated training corpus from public datasets.

    Parameters
    ----------
    n_total   : target corpus size (actual count may be slightly lower after dedup)
    sources   : which sources to draw from; subset of ALL_SOURCES
    seed      : RNG seed for shuffling
    weights   : per-source sampling fractions (unnormalised); equal split if None

    Returns
    -------
    Shuffled list of unique strings, length ≤ n_total.
    """
    unknown = set(sources) - set(_LOADERS)
    if unknown:
        raise ValueError(f"Unknown sources: {unknown}. Valid: {list(_LOADERS)}")

    # Normalise weights
    if weights is None:
        w = {s: 1.0 / len(sources) for s in sources}
    else:
        total_w = sum(weights.get(s, 0.0) for s in sources)
        if total_w == 0:
            raise ValueError("All source weights are zero.")
        w = {s: weights.get(s, 0.0) / total_w for s in sources}

    rng = random.Random(seed)

    # Fetch ~30 % extra per source to absorb deduplication losses.
    oversample = 1.3
    per_source_target = {s: max(1, int(n_total * w[s] * oversample)) for s in sources}

    # --- Load ---
    buckets: dict[str, list[str]] = {}
    for source in sources:
        target = per_source_target[source]
        print(f"[Corpus] Loading up to {target} samples from '{source}' …", flush=True)
        try:
            texts = _LOADERS[source](target)
            buckets[source] = texts
            print(f"[Corpus]   → {len(texts)} loaded", flush=True)
        except Exception as exc:
            print(f"[Corpus]   ! '{source}' failed ({exc.__class__.__name__}: {exc}); skipping", flush=True)
            buckets[source] = []

    active = [s for s in sources if buckets[s]]
    if not active:
        raise RuntimeError("All corpus sources failed — cannot build training corpus.")

    # Recompute effective weights over active sources only.
    total_active_w = sum(w[s] for s in active)
    eff_w = {s: w[s] / total_active_w for s in active}

    # --- Stratified mix ---
    # Shuffle each bucket, then take each source's proportional slice.
    mixed: list[str] = []
    for source in active:
        rng.shuffle(buckets[source])
        quota = int(n_total * eff_w[source])
        mixed.extend(buckets[source][:quota])

    # --- Dedup (order-preserving, case-sensitive) ---
    seen: set[str] = set()
    deduped: list[str] = []
    for t in mixed:
        key = hashlib.md5(t.encode()).hexdigest()
        if key not in seen:
            seen.add(key)
            deduped.append(t)

    # --- Final shuffle and truncate ---
    rng.shuffle(deduped)
    result = deduped[:n_total]

    source_str = ", ".join(active)
    print(
        f"[Corpus] Final corpus: {len(result)} unique texts "
        f"(sources: {source_str})",
        flush=True,
    )
    return result
