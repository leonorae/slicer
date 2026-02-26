"""
Diffusion Microscope — visualise LLM internals through a frozen diffusion model.

Projects LLM hidden-state activations into CLIP embedding space via a learned
linear map, then renders the projected vectors as images through Stable
Diffusion.  The generated images act as a "visual microscope" revealing what
the LLM's representations look like to a cross-modal visual system.

Core idea (Ning-inspired, novel application):
    LLM activation  ──▶  Linear projection  ──▶  CLIP space  ──▶  SD image
    (per layer)           (Ridge, trained)        (frozen)         (frozen)

Modules
-------
projection    : train / apply / persist the linear LLM→CLIP map
clip_bridge   : extract CLIP text embeddings; format conditioning for SD
generator     : generate images from projected activations via Stable Diffusion
pipeline      : end-to-end orchestration
"""

from .projection import LinearProjection
from .pipeline import MicroscopePipeline
from .training_data import load_training_corpus, ALL_SOURCES

__all__ = ["LinearProjection", "MicroscopePipeline", "load_training_corpus", "ALL_SOURCES"]
