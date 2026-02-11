"""
Engram Memory Module

This module provides the EnhancedEngramModule for adding hash-based O(1) memory
lookup to transformer models. Ported from weagan/Engram implementation.

Usage:
    from src.memory import EnhancedEngramModule, inject_engram_into_model
"""

from .engram_module import EnhancedEngramModule

# Optional imports (require transformers)
try:
    from .model_wrapper import inject_engram_into_model, EngramModelWrapper
except ImportError:
    inject_engram_into_model = None
    EngramModelWrapper = None

__all__ = [
    "EnhancedEngramModule",
    "inject_engram_into_model",
    "EngramModelWrapper",
]
