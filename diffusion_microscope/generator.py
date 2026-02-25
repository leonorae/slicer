"""
Image generation from projected LLM activations via Stable Diffusion.

The diffusion model is treated as a frozen rendering engine — a "microscope
lens" that converts CLIP-space vectors into human-viewable images.  The model
is never fine-tuned; all variation in the output comes from the projected
LLM activation that conditions it.

Supports:
    - Single-image generation from one projected vector
    - Layer sweeps (same text, activations at each layer → image grid)
    - CFG sweeps (same activation, multiple guidance scales → image row)
    - Interpolation sequences between two activations
"""

from __future__ import annotations

from typing import Optional, Sequence

import numpy as np


class DiffusionMicroscope:
    """
    Generate images from CLIP-space embeddings via a frozen Stable Diffusion
    pipeline.

    Parameters
    ----------
    sd_model_id : HuggingFace model ID for the SD checkpoint
    device      : 'cpu', 'cuda', or 'cuda:0'
    dtype       : torch dtype string ('float16' for GPU, 'float32' for CPU)
    """

    def __init__(
        self,
        sd_model_id: str = "stabilityai/stable-diffusion-2-1",
        device: str = "cpu",
        dtype: str = "float16",
    ):
        self.sd_model_id = sd_model_id
        self.device = device
        self.dtype = dtype
        self._pipe = None

    def _load_pipeline(self):
        """Lazy-load the SD pipeline on first use."""
        if self._pipe is not None:
            return

        import torch
        try:
            from diffusers import StableDiffusionPipeline
        except ImportError:
            raise ImportError(
                "diffusers is required for image generation.\n"
                "Install it with:  uv pip install diffusers --system\n"
                "or:               uv sync --extra microscope"
            ) from None

        torch_dtype = getattr(torch, self.dtype)
        self._pipe = StableDiffusionPipeline.from_pretrained(
            self.sd_model_id,
            torch_dtype=torch_dtype,
            safety_checker=None,
        )
        self._pipe = self._pipe.to(self.device)
        self._pipe.set_progress_bar_config(disable=True)

    # ------------------------------------------------------------------
    # Core generation
    # ------------------------------------------------------------------

    def generate(
        self,
        prompt_embeds: np.ndarray,
        negative_prompt_embeds: Optional[np.ndarray] = None,
        *,
        guidance_scale: float = 7.0,
        num_inference_steps: int = 30,
        seed: int = 42,
        height: int = 512,
        width: int = 512,
    ):
        """
        Generate a single image from pre-formatted SD conditioning.

        Parameters
        ----------
        prompt_embeds          : (1, seq_len, clip_dim) float32
        negative_prompt_embeds : (1, seq_len, clip_dim) float32 (zeros if None)
        guidance_scale         : classifier-free guidance strength
        seed                   : RNG seed for reproducibility

        Returns
        -------
        PIL.Image
        """
        import torch

        self._load_pipeline()

        pe = torch.tensor(prompt_embeds, dtype=self._pipe.unet.dtype).to(self.device)
        if negative_prompt_embeds is None:
            ne = torch.zeros_like(pe)
        else:
            ne = torch.tensor(
                negative_prompt_embeds, dtype=self._pipe.unet.dtype
            ).to(self.device)

        gen = torch.Generator(device=self.device).manual_seed(seed)

        result = self._pipe(
            prompt_embeds=pe,
            negative_prompt_embeds=ne,
            guidance_scale=guidance_scale,
            num_inference_steps=num_inference_steps,
            generator=gen,
            height=height,
            width=width,
        )
        return result.images[0]

    # ------------------------------------------------------------------
    # Convenience: generate from a raw CLIP-space vector
    # ------------------------------------------------------------------

    def generate_from_vector(
        self,
        clip_vector: np.ndarray,
        *,
        guidance_scale: float = 7.0,
        num_inference_steps: int = 30,
        seed: int = 42,
        seq_len: int = 77,
    ):
        """
        Generate an image from a single CLIP-space vector (no pre-formatting).

        Parameters
        ----------
        clip_vector : (clip_dim,) projected embedding
        """
        from .clip_bridge import format_sd_conditioning

        pe, ne = format_sd_conditioning(clip_vector, seq_len=seq_len)
        return self.generate(
            pe, ne,
            guidance_scale=guidance_scale,
            num_inference_steps=num_inference_steps,
            seed=seed,
        )

    # ------------------------------------------------------------------
    # Layer sweep
    # ------------------------------------------------------------------

    def layer_sweep(
        self,
        layer_activations: dict[int, np.ndarray],
        projection,
        *,
        cfg_scales: Sequence[float] = (3.0, 7.0, 12.0),
        seed: int = 42,
        num_inference_steps: int = 30,
    ) -> dict[tuple[int, float], "PIL.Image"]:
        """
        Generate images for each (layer, cfg_scale) combination.

        Parameters
        ----------
        layer_activations : {layer_idx: (hidden_dim,)} single-text activations
        projection        : trained LinearProjection
        cfg_scales        : guidance scales to sweep

        Returns
        -------
        dict mapping (layer_idx, cfg_scale) → PIL.Image
        """
        images = {}
        for layer_idx, act in sorted(layer_activations.items()):
            clip_vec = projection.transform(act)
            for cfg in cfg_scales:
                img = self.generate_from_vector(
                    clip_vec,
                    guidance_scale=cfg,
                    seed=seed,
                    num_inference_steps=num_inference_steps,
                )
                images[(layer_idx, cfg)] = img
        return images

    # ------------------------------------------------------------------
    # Interpolation
    # ------------------------------------------------------------------

    def interpolation_sequence(
        self,
        clip_vec_a: np.ndarray,
        clip_vec_b: np.ndarray,
        n_steps: int = 8,
        *,
        guidance_scale: float = 7.0,
        seed: int = 42,
        num_inference_steps: int = 30,
    ) -> list:
        """
        Generate images along a linear interpolation between two CLIP vectors.

        Returns list of (alpha, PIL.Image) tuples where alpha ∈ [0, 1].
        """
        results = []
        for i in range(n_steps):
            alpha = i / max(n_steps - 1, 1)
            vec = (1 - alpha) * clip_vec_a + alpha * clip_vec_b
            img = self.generate_from_vector(
                vec,
                guidance_scale=guidance_scale,
                seed=seed,
                num_inference_steps=num_inference_steps,
            )
            results.append((alpha, img))
        return results
