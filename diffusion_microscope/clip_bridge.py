"""
CLIP text-embedding extraction and Stable Diffusion conditioning formatter.

Handles the interface between the linear projection output (a single CLIP-space
vector) and what Stable Diffusion actually expects as conditioning input.

SD conditioning format (diffusers):
    prompt_embeds          : (batch, seq_len, clip_dim)  — token-level embeddings
    negative_prompt_embeds : (batch, seq_len, clip_dim)  — unconditional baseline

Since we have a single projected vector (not a token sequence), we tile it
across the sequence dimension.  This is equivalent to a "constant prompt" where
every token position carries the same semantic signal — an intentional design
choice that maximises the fidelity of the geometric projection at the cost of
compositional structure (which we don't need for interpretability).
"""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np


def extract_clip_text_embeddings(
    texts: list[str],
    clip_model_name: str = "ViT-H-14",
    pretrained: str = "laion2b_s32b_b79k",
    device: str = "cpu",
    batch_size: int = 32,
) -> np.ndarray:
    """
    Extract CLIP text embeddings for a list of strings.

    Uses OpenCLIP so the model variant matches what Stable Diffusion was
    trained with.

    Parameters
    ----------
    texts           : input strings
    clip_model_name : OpenCLIP architecture name
    pretrained      : pretrained weights tag (must match SD version)
    device          : 'cpu' or 'cuda'
    batch_size      : texts per forward pass

    Returns
    -------
    (n_texts, clip_dim) float32 array of text embeddings
    """
    import open_clip
    import torch

    model, _, _ = open_clip.create_model_and_transforms(
        clip_model_name, pretrained=pretrained, device=device,
    )
    tokenizer = open_clip.get_tokenizer(clip_model_name)
    model.eval()

    all_embeds = []
    for start in range(0, len(texts), batch_size):
        batch_texts = texts[start : start + batch_size]
        tokens = tokenizer(batch_texts).to(device)
        with torch.no_grad():
            emb = model.encode_text(tokens)  # (batch, clip_dim)
        all_embeds.append(emb.cpu().float().numpy())

    return np.concatenate(all_embeds, axis=0)


def extract_llm_last_token_activations(
    texts: list[str],
    model,
    tokenizer,
    *,
    layers: Optional[list[int]] = None,
    max_length: int = 256,
    device: str = "cpu",
    batch_size: int = 4,
) -> dict[int, np.ndarray]:
    """
    Extract the last *real* token's hidden state at each requested layer.

    Parameters
    ----------
    texts      : input strings
    model      : a HuggingFace causal LM (e.g. GPT2LMHeadModel)
    tokenizer  : matching tokenizer
    layers     : layer indices to extract (default: all)
    max_length : tokeniser truncation length
    device     : 'cpu' or 'cuda'
    batch_size : texts per forward pass

    Returns
    -------
    dict mapping layer_index → (n_texts, hidden_dim) float32 array
    """
    import torch

    model.eval()
    model.to(device)
    n_layers = model.config.num_hidden_layers
    if layers is None:
        layers = list(range(n_layers + 1))  # include embedding layer (0)

    results = {l: [] for l in layers}

    for start in range(0, len(texts), batch_size):
        batch_texts = texts[start : start + batch_size]
        enc = tokenizer(
            batch_texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_length,
        ).to(device)

        with torch.no_grad():
            out = model(**enc, output_hidden_states=True)

        # hidden_states: tuple of (batch, seq_len, hidden_dim), one per layer
        # Index 0 = embedding output, 1..n_layers = transformer layers
        attention_mask = enc["attention_mask"]  # (batch, seq_len)

        for i in range(len(batch_texts)):
            # Last real token position (before padding)
            real_len = int(attention_mask[i].sum().item())
            last_pos = real_len - 1

            for layer_idx in layers:
                if layer_idx < len(out.hidden_states):
                    vec = out.hidden_states[layer_idx][i, last_pos, :]
                    results[layer_idx].append(vec.cpu().float().numpy())

    return {
        layer_idx: np.stack(vecs)
        for layer_idx, vecs in results.items()
        if vecs
    }


def format_sd_conditioning(
    projected_embedding: np.ndarray,
    seq_len: int = 77,
    clip_dim: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Format a single CLIP-space vector into SD-compatible conditioning tensors.

    Tiles the projected vector across the token sequence dimension and creates
    a zero-vector unconditional baseline for classifier-free guidance.

    Parameters
    ----------
    projected_embedding : (clip_dim,) or (batch, clip_dim)
    seq_len             : SD token sequence length (77 for SD 1.5 / 2.1)
    clip_dim            : override clip_dim (inferred from input if None)

    Returns
    -------
    (prompt_embeds, negative_prompt_embeds)
        Both shaped (batch, seq_len, clip_dim), float32
    """
    emb = np.asarray(projected_embedding, dtype=np.float32)
    if emb.ndim == 1:
        emb = emb[np.newaxis]  # (1, clip_dim)

    batch, dim = emb.shape
    if clip_dim is not None and dim != clip_dim:
        raise ValueError(f"expected clip_dim={clip_dim}, got {dim}")

    # Tile across sequence dimension: (batch, 1, dim) → (batch, seq_len, dim)
    prompt_embeds = np.tile(emb[:, np.newaxis, :], (1, seq_len, 1))
    negative_embeds = np.zeros_like(prompt_embeds)

    return prompt_embeds, negative_embeds
