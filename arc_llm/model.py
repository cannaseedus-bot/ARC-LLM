"""Compatibility wrapper around the dedicated µARC-LLM implementation."""

from mu_arc_llm.model import (  # noqa: F401
    GeodesicAttention,
    MuARCConfig,
    MuARCLLM,
    MuArcLayer,
    train_step,
)

# Backward-compatible aliases
MiniARCConfig = MuARCConfig
MiniARCLLM = MuARCLLM
ARCConfig = MuARCConfig
ARCLLM = MuARCLLM
