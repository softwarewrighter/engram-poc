"""
Model Wrapper for Injecting Engram into HuggingFace Models

This module provides utilities to add EnhancedEngramModule to existing
HuggingFace transformer models without modifying their source code.

The approach:
1. Wrap each transformer layer to add Engram after attention
2. Keep the base model frozen (or trainable) as desired
3. Train the Engram memory tables along with optional LoRA adapters
"""

import torch
import torch.nn as nn
from typing import Optional, List, Tuple, Any, Dict
from transformers import PreTrainedModel, AutoModelForCausalLM, AutoTokenizer

from .engram_module import EnhancedEngramModule


class EngramLayerWrapper(nn.Module):
    """
    Wraps an existing transformer layer to add Engram memory.

    This is injected between the attention output and FFN of each layer.
    """

    def __init__(
        self,
        original_layer: nn.Module,
        d_model: int,
        memory_size: int = 50000,
        layer_idx: int = 0,
    ):
        super().__init__()
        self.original_layer = original_layer
        self.layer_idx = layer_idx

        # Add Engram module
        self.engram = EnhancedEngramModule(
            table_size=memory_size,
            d_model=d_model,
            n_heads=4,
        )

        # Layer norm after Engram injection
        self.engram_norm = nn.LayerNorm(d_model)

        # Accumulated input_ids during generation (handles autoregressive case)
        self._accumulated_input_ids: Optional[torch.Tensor] = None
        self._generation_mode: bool = False

    def set_input_ids(self, input_ids: torch.Tensor, generation_mode: bool = False):
        """Store input_ids for the current forward pass.

        Args:
            input_ids: The full accumulated input_ids sequence
            generation_mode: If True, signals that hidden_states may be shorter
                           than input_ids (due to KV caching) and we should
                           align to the last positions.
        """
        # Always just store the provided input_ids (accumulation handled by caller)
        self._accumulated_input_ids = input_ids
        self._generation_mode = generation_mode

    def reset_generation(self):
        """Reset accumulated state for new generation."""
        self._accumulated_input_ids = None
        self._generation_mode = False

    def forward(self, hidden_states: torch.Tensor, *args, **kwargs) -> Any:
        # Run original layer
        outputs = self.original_layer(hidden_states, *args, **kwargs)

        # Handle different output formats
        if isinstance(outputs, tuple):
            hidden_states_out = outputs[0]
            rest = outputs[1:]
        else:
            hidden_states_out = outputs
            rest = ()

        # Apply Engram if we have accumulated input_ids
        if self._accumulated_input_ids is not None:
            input_ids = self._accumulated_input_ids
            seq_len = hidden_states_out.shape[1]

            # In generation mode, we may have accumulated more tokens than
            # the current hidden_states (which only has the new position(s))
            if self._generation_mode and input_ids.shape[1] > seq_len:
                # Use only the last seq_len positions of input_ids
                # This aligns with the hidden states for the current step
                input_ids = input_ids[:, -seq_len:]
            elif input_ids.shape[1] < seq_len:
                # During training, pad if needed (shouldn't happen normally)
                pad_len = seq_len - input_ids.shape[1]
                input_ids = torch.cat([
                    input_ids,
                    torch.zeros(input_ids.shape[0], pad_len, dtype=input_ids.dtype, device=input_ids.device)
                ], dim=1)

            # Apply Engram memory (pure residual, no extra normalization)
            # The Engram module itself uses a residual: output = hidden + merge(gated_memory)
            # So we just apply it directly - with near-zero gates, output ≈ hidden
            hidden_states_out = self.engram(hidden_states_out, input_ids)

        if rest:
            return (hidden_states_out,) + rest
        return hidden_states_out


class EngramModelWrapper(nn.Module):
    """
    Wraps a HuggingFace model to add Engram memory to specified layers.

    This keeps the original model intact while adding trainable memory modules.
    The memory tables and gates are trained while the base model can be frozen.

    Args:
        model: HuggingFace model to wrap
        memory_size: Size of each layer's memory table (default: 50000)
        inject_layers: Which layers to inject Engram into. None = all layers.
        freeze_base: Whether to freeze the base model parameters

    Example:
        >>> from transformers import AutoModelForCausalLM
        >>> base_model = AutoModelForCausalLM.from_pretrained("HuggingFaceTB/SmolLM-135M")
        >>> engram_model = EngramModelWrapper(base_model, memory_size=50000)
        >>> # Train engram_model.engram_parameters() while base is frozen
    """

    # Known model architectures and their layer access paths
    LAYER_PATHS = {
        "LlamaForCausalLM": ("model", "layers"),
        "MistralForCausalLM": ("model", "layers"),
        "Qwen2ForCausalLM": ("model", "layers"),
        "GPT2LMHeadModel": ("transformer", "h"),
        "GPTNeoForCausalLM": ("transformer", "h"),
        "PhiForCausalLM": ("model", "layers"),
        "GemmaForCausalLM": ("model", "layers"),
        # SmolLM uses LLaMA architecture
    }

    def __init__(
        self,
        model: PreTrainedModel,
        memory_size: int = 50000,
        inject_layers: Optional[List[int]] = None,
        freeze_base: bool = True,
    ):
        super().__init__()
        self.model = model
        self.memory_size = memory_size
        self.config = model.config

        # Get hidden size
        self.d_model = getattr(
            self.config,
            "hidden_size",
            getattr(self.config, "n_embd", 768)
        )

        # Freeze base model if requested
        if freeze_base:
            for param in self.model.parameters():
                param.requires_grad = False

        # Get model dtype
        self.dtype = next(self.model.parameters()).dtype

        # Find and wrap layers
        self.wrapped_layers: List[EngramLayerWrapper] = []
        self._inject_engram_layers(inject_layers)

        # Convert Engram modules to model's dtype
        for layer in self.wrapped_layers:
            layer.engram = layer.engram.to(self.dtype)
            layer.engram_norm = layer.engram_norm.to(self.dtype)

    def _get_layers(self) -> Tuple[nn.Module, nn.ModuleList]:
        """Get the transformer layers from the model."""
        model_type = type(self.model).__name__

        if model_type in self.LAYER_PATHS:
            parent_attr, layers_attr = self.LAYER_PATHS[model_type]
            parent = getattr(self.model, parent_attr)
            layers = getattr(parent, layers_attr)
            return parent, layers

        # Try common patterns
        for parent_name in ["model", "transformer", "decoder"]:
            if hasattr(self.model, parent_name):
                parent = getattr(self.model, parent_name)
                for layers_name in ["layers", "h", "blocks"]:
                    if hasattr(parent, layers_name):
                        layers = getattr(parent, layers_name)
                        if isinstance(layers, nn.ModuleList):
                            return parent, layers

        raise ValueError(
            f"Could not find transformer layers in model type {model_type}. "
            f"Known types: {list(self.LAYER_PATHS.keys())}"
        )

    def _inject_engram_layers(self, inject_layers: Optional[List[int]]):
        """Inject Engram modules into specified layers."""
        parent, layers = self._get_layers()

        num_layers = len(layers)
        if inject_layers is None:
            inject_layers = list(range(num_layers))

        for idx in inject_layers:
            if idx >= num_layers:
                continue

            # Wrap the layer
            original_layer = layers[idx]
            wrapped = EngramLayerWrapper(
                original_layer=original_layer,
                d_model=self.d_model,
                memory_size=self.memory_size,
                layer_idx=idx,
            )

            # Replace in the model
            layers[idx] = wrapped
            self.wrapped_layers.append(wrapped)

        print(f"Injected Engram into {len(self.wrapped_layers)} layers")

    def _propagate_input_ids(self, input_ids: torch.Tensor, generation_mode: bool = False):
        """Set input_ids on all wrapped layers for memory lookup."""
        for layer in self.wrapped_layers:
            layer.set_input_ids(input_ids, generation_mode=generation_mode)

    def _reset_generation_state(self):
        """Reset accumulated state for new generation."""
        for layer in self.wrapped_layers:
            layer.reset_generation()

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        **kwargs,
    ):
        """Forward pass with Engram memory."""
        # Propagate input_ids to all wrapped layers (training mode)
        self._propagate_input_ids(input_ids, generation_mode=False)

        # Run the wrapped model
        return self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            **kwargs,
        )

    def generate(self, input_ids: torch.Tensor, **kwargs):
        """Generation with Engram memory.

        Properly handles autoregressive generation by tracking accumulated
        input_ids across generation steps.
        """
        # Reset any previous generation state
        self._reset_generation_state()

        # Store the full input sequence - we'll track it through generation
        self._full_input_ids = input_ids.clone()

        # Hook into the model's forward to intercept each step
        original_forward = self.model.forward

        def hooked_forward(*args, **fwd_kwargs):
            # Extract input_ids from either args or kwargs
            input_ids_step = fwd_kwargs.get('input_ids', args[0] if args else None)

            if input_ids_step is not None:
                if input_ids_step.shape[1] == 1:
                    # Single new token - append to our tracking
                    self._full_input_ids = torch.cat([
                        self._full_input_ids, input_ids_step
                    ], dim=1)
                else:
                    # Full sequence (first step) - update our tracking
                    self._full_input_ids = input_ids_step.clone()

                # Propagate full sequence to all Engram layers (generation_mode=True for proper truncation)
                self._propagate_input_ids(self._full_input_ids, generation_mode=True)

            return original_forward(*args, **fwd_kwargs)

        self.model.forward = hooked_forward

        try:
            result = self.model.generate(input_ids=input_ids, **kwargs)
        finally:
            # Restore original method
            self.model.forward = original_forward
            # Clean up
            self._full_input_ids = None

        return result

    def engram_parameters(self) -> List[nn.Parameter]:
        """Get only the Engram-related parameters for training."""
        params = []
        for layer in self.wrapped_layers:
            params.extend(layer.engram.parameters())
            params.extend(layer.engram_norm.parameters())
        return params

    def engram_named_parameters(self) -> List[Tuple[str, nn.Parameter]]:
        """Get named Engram parameters."""
        params = []
        for i, layer in enumerate(self.wrapped_layers):
            for name, param in layer.engram.named_parameters():
                params.append((f"engram_layer_{i}.{name}", param))
            for name, param in layer.engram_norm.named_parameters():
                params.append((f"engram_layer_{i}.norm.{name}", param))
        return params

    def get_memory_stats(self) -> Dict[str, Any]:
        """Get statistics about all memory tables."""
        stats = {
            "num_engram_layers": len(self.wrapped_layers),
            "memory_size": self.memory_size,
            "d_model": self.d_model,
            "total_engram_params": sum(p.numel() for p in self.engram_parameters()),
            "layers": [],
        }
        for i, layer in enumerate(self.wrapped_layers):
            stats["layers"].append({
                "layer_idx": layer.layer_idx,
                **layer.engram.get_memory_stats(),
            })
        return stats

    def save_engram_weights(self, path: str):
        """Save only the Engram weights."""
        state_dict = {}
        for name, param in self.engram_named_parameters():
            state_dict[name] = param.data
        torch.save(state_dict, path)
        print(f"Saved Engram weights to {path}")

    def load_engram_weights(self, path: str):
        """Load Engram weights."""
        state_dict = torch.load(path, map_location="cpu")
        for name, param in self.engram_named_parameters():
            if name in state_dict:
                param.data = state_dict[name].to(param.device)
        print(f"Loaded Engram weights from {path}")


def inject_engram_into_model(
    model_name_or_path: str,
    memory_size: int = 50000,
    inject_layers: Optional[List[int]] = None,
    freeze_base: bool = True,
    device: str = "auto",
    **model_kwargs,
) -> Tuple[EngramModelWrapper, Any]:
    """
    Convenience function to load a model and inject Engram.

    Args:
        model_name_or_path: HuggingFace model ID or path
        memory_size: Size of memory tables
        inject_layers: Which layers to inject (None = all)
        freeze_base: Whether to freeze base model
        device: Device to load model on
        **model_kwargs: Additional arguments for AutoModelForCausalLM

    Returns:
        Tuple of (EngramModelWrapper, tokenizer)

    Example:
        >>> model, tokenizer = inject_engram_into_model(
        ...     "HuggingFaceTB/SmolLM-135M-Instruct",
        ...     memory_size=50000
        ... )
    """
    # Load base model
    model = AutoModelForCausalLM.from_pretrained(
        model_name_or_path,
        device_map=device if device != "auto" else None,
        **model_kwargs,
    )

    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)

    # Wrap with Engram
    engram_model = EngramModelWrapper(
        model=model,
        memory_size=memory_size,
        inject_layers=inject_layers,
        freeze_base=freeze_base,
    )

    return engram_model, tokenizer
