"""
Activation extraction from HuggingFace transformer models.

Registers forward hooks at predefined extraction points and collects activations.
Supports GPT-2, LLaMA/Mistral, GPT-NeoX (Pythia), and OLMo-style architectures,
auto-detected from model.config.model_type.
"""

import re
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

# ---------------------------------------------------------------------------
# Architecture configurations
# Each entry maps symbolic hook targets to dotted attribute paths within a layer
# (resolved via _get_nested_attr). "final_norm_path" is resolved from the model root.
# ---------------------------------------------------------------------------
ARCH_CONFIGS: Dict[str, dict] = {
    "gpt2": {
        "layers_path": "transformer.h",
        "pre_attn_norm": "ln_1",
        "attn_out": "attn.c_proj",
        "pre_mlp_norm": "ln_2",
        "mlp_out": "mlp.c_proj",
        "final_norm_path": "transformer.ln_f",
    },
    "llama": {
        "layers_path": "model.layers",
        "pre_attn_norm": "input_layernorm",
        "attn_out": "self_attn.o_proj",
        "pre_mlp_norm": "post_attention_layernorm",
        "mlp_out": "mlp.down_proj",
        "final_norm_path": "model.norm",
    },
    "mistral": {
        "layers_path": "model.layers",
        "pre_attn_norm": "input_layernorm",
        "attn_out": "self_attn.o_proj",
        "pre_mlp_norm": "post_attention_layernorm",
        "mlp_out": "mlp.down_proj",
        "final_norm_path": "model.norm",
    },
    "gemma": {
        "layers_path": "model.layers",
        "pre_attn_norm": "input_layernorm",
        "attn_out": "self_attn.o_proj",
        "pre_mlp_norm": "post_feedforward_layernorm",
        "mlp_out": "mlp.down_proj",
        "final_norm_path": "model.norm",
    },
    "gpt_neox": {
        # Pythia uses parallel attention+MLP; we still hook both paths for symmetry.
        "layers_path": "gpt_neox.layers",
        "pre_attn_norm": "input_layernorm",
        "attn_out": "attention.dense",
        "pre_mlp_norm": "post_attention_layernorm",
        "mlp_out": "mlp.dense_4h_to_h",
        "final_norm_path": "gpt_neox.final_layer_norm",
    },
    "olmo": {
        "layers_path": "model.transformer.blocks",
        "pre_attn_norm": "attn_norm",
        "attn_out": "attn.out_proj",
        "pre_mlp_norm": "ff_norm",
        "mlp_out": "ff_out",
        "final_norm_path": "model.transformer.ln_f",
    },
}


def detect_arch_config(model) -> dict:
    """
    Auto-detect architecture configuration from model structure.
    Falls back to structural inspection if model_type is unknown.
    """
    model_type = getattr(getattr(model, "config", None), "model_type", "").lower()

    # Direct match
    if model_type in ARCH_CONFIGS:
        return ARCH_CONFIGS[model_type]

    # Fuzzy match (e.g. "llama2" → "llama")
    for key in ARCH_CONFIGS:
        if key in model_type:
            return ARCH_CONFIGS[key]

    # Structural fallback
    if hasattr(model, "transformer") and hasattr(model.transformer, "h"):
        return ARCH_CONFIGS["gpt2"]
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        return ARCH_CONFIGS["llama"]
    if hasattr(model, "gpt_neox"):
        return ARCH_CONFIGS["gpt_neox"]

    raise ValueError(
        f"Unknown model architecture '{model_type}'. "
        f"Supported: {list(ARCH_CONFIGS.keys())}. "
        "Add a custom entry to ARCH_CONFIGS if needed."
    )


def _get_nested_attr(obj, path: str):
    """Resolve a dotted attribute path, e.g. 'model.layers'."""
    for part in path.split("."):
        obj = getattr(obj, part)
    return obj


def _resolve_submodule(layer, path: str):
    """Resolve a dotted path within a layer; return None if any attribute is missing."""
    try:
        return _get_nested_attr(layer, path)
    except AttributeError:
        return None


# ---------------------------------------------------------------------------
# Extraction point names used as keys in returned dicts.
# Naming convention: layer_{idx:02d}_{point}
# ---------------------------------------------------------------------------
POINT_NAMES = {
    "pre_attn_norm": "pre_attn_norm",
    "attn_out": "attn_out",
    "post_attn_residual": "post_attn_residual",
    "pre_mlp_norm": "pre_mlp_norm",
    "mlp_out": "mlp_out",
    "post_mlp_residual": "post_mlp_residual",
    "final_norm": "final_norm",
}


class ActivationExtractor:
    """
    Registers forward hooks on a transformer model and collects activations at
    seven predefined extraction points per layer, plus the final norm.

    After all batches are processed, tokens are randomly subsampled to at most
    ``max_tokens`` per extraction point to keep memory usage bounded.

    Attributes
    ----------
    model      : the HuggingFace model
    model_name : string identifier (used for arch detection)
    max_tokens : maximum tokens to return per extraction point (default 10 000)
    """

    def __init__(self, model, model_name: str, max_tokens: int = 10_000):
        self.model = model
        self.model_name = model_name
        self.max_tokens = max_tokens
        self.arch = detect_arch_config(model)

        # Raw storage: name -> list of (batch, seq_len, hidden) tensors
        self._raw: Dict[str, List[torch.Tensor]] = defaultdict(list)
        # Real sequence lengths (excluding padding), one per sequence
        self._seq_lengths: List[int] = []
        self._hooks: List = []

    # ------------------------------------------------------------------
    # Hook factories
    # ------------------------------------------------------------------

    def _output_hook(self, name: str):
        """Hook that captures the module's output tensor."""
        def fn(module, input, output):
            out = output[0] if isinstance(output, tuple) else output
            self._raw[name].append(out.detach().float().cpu())
        return fn

    def _input_hook(self, name: str):
        """
        Pre-hook that captures the module's first input tensor.
        Used to capture the post-attention residual as the input to pre_mlp_norm.
        """
        def fn(module, input):
            inp = input[0] if isinstance(input, tuple) else input
            self._raw[name].append(inp.detach().float().cpu())
        return fn

    # ------------------------------------------------------------------
    # Hook registration
    # ------------------------------------------------------------------

    def register_hooks(self):
        """Dynamically register hooks for the detected architecture."""
        try:
            layers = _get_nested_attr(self.model, self.arch["layers_path"])
        except AttributeError as e:
            raise RuntimeError(
                f"Cannot resolve layers at '{self.arch['layers_path']}': {e}"
            )

        for idx, layer in enumerate(layers):
            pfx = f"layer_{idx:02d}"

            # 1. Pre-attention norm output
            mod = _resolve_submodule(layer, self.arch["pre_attn_norm"])
            if mod is not None:
                self._hooks.append(
                    mod.register_forward_hook(self._output_hook(f"{pfx}_pre_attn_norm"))
                )

            # 2. Attention output (projection, before residual addition)
            mod = _resolve_submodule(layer, self.arch["attn_out"])
            if mod is not None:
                self._hooks.append(
                    mod.register_forward_hook(self._output_hook(f"{pfx}_attn_out"))
                )

            # 3. Post-attention residual = input to pre_mlp_norm
            mod = _resolve_submodule(layer, self.arch["pre_mlp_norm"])
            if mod is not None:
                self._hooks.append(
                    mod.register_forward_pre_hook(
                        self._input_hook(f"{pfx}_post_attn_residual")
                    )
                )

            # 4. Pre-MLP norm output
            if mod is not None:
                self._hooks.append(
                    mod.register_forward_hook(self._output_hook(f"{pfx}_pre_mlp_norm"))
                )

            # 5. MLP output (down-projection, before residual addition)
            mod = _resolve_submodule(layer, self.arch["mlp_out"])
            if mod is not None:
                self._hooks.append(
                    mod.register_forward_hook(self._output_hook(f"{pfx}_mlp_out"))
                )

            # 6. Post-MLP residual = full layer output
            self._hooks.append(
                layer.register_forward_hook(self._output_hook(f"{pfx}_post_mlp_residual"))
            )

        # 7. Final norm
        try:
            final_norm = _get_nested_attr(self.model, self.arch["final_norm_path"])
            self._hooks.append(
                final_norm.register_forward_hook(self._output_hook("final_norm"))
            )
        except AttributeError:
            pass  # Not all models expose a final norm at the expected path

    def remove_hooks(self):
        """Remove all registered hooks."""
        for h in self._hooks:
            h.remove()
        self._hooks.clear()

    # ------------------------------------------------------------------
    # Main extraction method
    # ------------------------------------------------------------------

    def extract(
        self,
        prompts: List[str],
        tokenizer,
        max_length: int = 128,
        batch_size: int = 4,
    ) -> Tuple[Dict[str, np.ndarray], Dict[str, object]]:
        """
        Run forward passes over ``prompts`` and collect activations.

        Parameters
        ----------
        prompts    : list of text strings
        tokenizer  : HuggingFace tokenizer (must match the model)
        max_length : maximum token sequence length (longer sequences are truncated)
        batch_size : number of prompts per forward pass

        Returns
        -------
        activations : dict mapping extraction-point name -> ndarray (n_tokens, hidden)
            Tokens are randomly subsampled to at most ``max_tokens`` total.
        per_seq_data : dict with
            'post_mlp_per_seq' : list of (real_seq_len, hidden) arrays, one per sequence
            'seq_positions'     : list of (real_seq_len,) position-index arrays
        """
        self._raw.clear()
        self._seq_lengths.clear()
        self.register_hooks()

        device = next(self.model.parameters()).device

        try:
            for start in range(0, len(prompts), batch_size):
                batch = prompts[start : start + batch_size]
                inputs = tokenizer(
                    batch,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=max_length,
                )
                inputs = {k: v.to(device) for k, v in inputs.items()}

                with torch.no_grad():
                    self.model(**inputs)

                # Record real sequence lengths (sum of attention mask)
                attn_mask = inputs.get("attention_mask")
                if attn_mask is not None:
                    for i in range(attn_mask.shape[0]):
                        self._seq_lengths.append(int(attn_mask[i].sum().item()))
                else:
                    seq_len = inputs["input_ids"].shape[1]
                    self._seq_lengths.extend([seq_len] * len(batch))
        finally:
            self.remove_hooks()

        # ------------------------------------------------------------------
        # Concatenate batches and subsample tokens
        # ------------------------------------------------------------------
        activations_full: Dict[str, np.ndarray] = {}
        for key, tensors in self._raw.items():
            # Each tensor has shape (batch, seq_len, hidden).  Different batches
            # may have different seq_len (padding to the longest sequence in that
            # batch), so we cannot torch.cat across batches on dim=0.  Instead,
            # flatten each batch individually then np.vstack.
            flat_parts = []
            for t in tensors:
                B, S, H = t.shape
                flat_parts.append(t.reshape(B * S, H).numpy())
            activations_full[key] = np.concatenate(flat_parts, axis=0)

        activations = self._subsample(activations_full)

        # ------------------------------------------------------------------
        # Build per-sequence data for position-dependence analysis.
        # Use the last post_mlp_residual layer (deepest layer).
        # ------------------------------------------------------------------
        per_seq_data: Dict[str, list] = {
            "post_mlp_per_seq": [],
            "seq_positions": [],
        }

        layer_keys = sorted(
            [k for k in self._raw if "post_mlp_residual" in k],
            key=lambda x: int(re.search(r"layer_(\d+)", x).group(1)),
        )

        if layer_keys:
            last_key = layer_keys[-1]
            tensors = self._raw[last_key]
            seq_offset = 0
            for batch_tensor in tensors:
                B, S, H = batch_tensor.shape
                arr = batch_tensor.numpy()
                for i in range(B):
                    real_len = (
                        self._seq_lengths[seq_offset]
                        if seq_offset < len(self._seq_lengths)
                        else S
                    )
                    seq_offset += 1
                    # Strip padding
                    per_seq_data["post_mlp_per_seq"].append(arr[i, :real_len, :])
                    per_seq_data["seq_positions"].append(np.arange(real_len))

        return activations, per_seq_data

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _subsample(
        self, activations_full: Dict[str, np.ndarray]
    ) -> Dict[str, np.ndarray]:
        """Randomly subsample each extraction point to at most max_tokens rows."""
        result: Dict[str, np.ndarray] = {}
        rng = np.random.default_rng(seed=42)
        for key, arr in activations_full.items():
            n = len(arr)
            if n > self.max_tokens:
                idx = rng.choice(n, self.max_tokens, replace=False)
                result[key] = arr[idx]
            else:
                result[key] = arr
        return result
