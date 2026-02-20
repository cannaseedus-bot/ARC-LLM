"""ARC-LLM compatibility package.

The canonical implementation now lives under `mu_arc_llm/`.
"""

from .manifesto import __manifesto__, __version__
from .model import ARCConfig, ARCLLM, MiniARCConfig, MiniARCLLM, train_step

__all__ = [
    "ARCLLM",
    "ARCConfig",
    "MiniARCLLM",
    "MiniARCConfig",
    "train_step",
    "__version__",
    "__manifesto__",
]
