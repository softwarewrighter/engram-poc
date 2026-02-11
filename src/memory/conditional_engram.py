"""
Conditional Engram: Smart routing between base model and memory.

Routes inputs to Engram memory only when they match lookup patterns,
bypassing memory for general queries where it doesn't help.

Key insight from DeepSeek: Engram works for structured lookups,
not arbitrary data. This module makes that practical.
"""

import torch
import torch.nn as nn
from typing import Optional, List, Set
from transformers import PreTrainedTokenizer


class LookupPatternDetector:
    """
    Detect when input matches structured lookup patterns.

    Returns confidence (0-1) that the input is a lookup query
    that would benefit from Engram memory.
    """

    # Patterns that indicate structured lookup queries
    LOOKUP_PREFIXES = [
        "CAPITAL:", "PORT:", "HTTP:", "ELEMENT:", "ELEMENT_NAME:", "ELEMENT_NUM:",
        "ACRONYM:", "CONVERT:", "CODE:", "DEFINE:", "LOOKUP:",
        "[Q",  # Wikidata entity IDs
    ]

    # Question patterns that suggest factual lookup
    FACTUAL_PATTERNS = [
        "what is the capital of",
        "what port does",
        "what does",
        "define ",
        "expand ",
        "the meaning of",
        "stands for",
    ]

    def __init__(self, custom_prefixes: Optional[List[str]] = None):
        """
        Args:
            custom_prefixes: Additional prefixes to recognize as lookup patterns
        """
        self.prefixes = set(self.LOOKUP_PREFIXES)
        if custom_prefixes:
            self.prefixes.update(custom_prefixes)

    def __call__(self, text: str) -> float:
        """
        Return confidence (0-1) that this is a lookup query.

        Args:
            text: Input text to analyze

        Returns:
            Confidence score: 1.0 = definitely lookup, 0.0 = definitely not
        """
        text_upper = text.upper()
        text_lower = text.lower()

        # Check for explicit lookup prefixes (highest confidence)
        for prefix in self.prefixes:
            if prefix.upper() in text_upper:
                return 1.0

        # Check for factual question patterns (medium confidence)
        for pattern in self.FACTUAL_PATTERNS:
            if pattern in text_lower:
                return 0.7

        # Check for short, structured inputs (might be lookup)
        if len(text.split()) <= 3 and ":" in text:
            return 0.5

        return 0.0

    def add_prefix(self, prefix: str):
        """Add a custom lookup prefix."""
        self.prefixes.add(prefix)


class ConditionalEngramLayer(nn.Module):
    """
    Wraps an Engram layer with conditional gating.

    Decides whether to use memory based on:
    1. Pattern detection (rule-based)
    2. Learned confidence (neural)
    """

    def __init__(
        self,
        engram_layer: nn.Module,
        d_model: int,
        use_learned_gate: bool = True,
    ):
        super().__init__()
        self.engram_layer = engram_layer
        self.d_model = d_model
        self.use_learned_gate = use_learned_gate

        # Learned confidence gate
        if use_learned_gate:
            self.confidence_net = nn.Sequential(
                nn.Linear(d_model, d_model // 4),
                nn.ReLU(),
                nn.Linear(d_model // 4, 1),
                nn.Sigmoid(),
            )
            # Initialize to low confidence (don't use memory by default)
            with torch.no_grad():
                self.confidence_net[2].bias.fill_(-2.0)

        # Pattern-based confidence (set externally)
        self._pattern_confidence: float = 0.0

    def set_pattern_confidence(self, confidence: float):
        """Set pattern-based confidence for current forward pass."""
        self._pattern_confidence = confidence

    def forward(
        self,
        hidden_states: torch.Tensor,
        input_ids: torch.Tensor,
    ) -> torch.Tensor:
        """
        Conditionally apply Engram memory.

        Args:
            hidden_states: [batch, seq_len, d_model]
            input_ids: [batch, seq_len]

        Returns:
            Output hidden states, conditionally modified by memory
        """
        # If pattern detection says this is definitely not a lookup, skip memory
        if self._pattern_confidence == 0.0 and not self.use_learned_gate:
            return hidden_states

        # Get Engram memory contribution
        memory_output = self.engram_layer(hidden_states, input_ids)

        # Compute final gate value
        if self.use_learned_gate:
            # Learned confidence from hidden states
            # Use mean pooling over sequence for global decision
            pooled = hidden_states.mean(dim=1, keepdim=True)  # [batch, 1, d_model]
            learned_confidence = self.confidence_net(pooled)  # [batch, 1, 1]

            # Combine pattern and learned confidence
            # Pattern confidence acts as a prior
            gate = self._pattern_confidence * 0.5 + learned_confidence.squeeze(-1) * 0.5
            gate = gate.unsqueeze(-1)  # [batch, 1, 1] for broadcasting
        else:
            gate = self._pattern_confidence

        # Blend base and memory outputs
        output = (1 - gate) * hidden_states + gate * memory_output

        return output


class ConditionalEngramWrapper(nn.Module):
    """
    Wraps a model with conditional Engram routing.

    Detects lookup patterns and routes to Engram only when appropriate.
    """

    def __init__(
        self,
        engram_model: nn.Module,
        tokenizer: PreTrainedTokenizer,
        custom_prefixes: Optional[List[str]] = None,
        use_learned_gate: bool = True,
    ):
        """
        Args:
            engram_model: EngramModelWrapper instance
            tokenizer: Tokenizer for decoding input_ids
            custom_prefixes: Additional lookup prefixes to recognize
            use_learned_gate: Whether to use learned gating (vs pure rule-based)
        """
        super().__init__()
        self.engram_model = engram_model
        self.tokenizer = tokenizer
        self.pattern_detector = LookupPatternDetector(custom_prefixes)
        self.use_learned_gate = use_learned_gate

        # Wrap each Engram layer with conditional gating
        self.conditional_layers: List[ConditionalEngramLayer] = []

        if hasattr(engram_model, 'wrapped_layers'):
            for layer in engram_model.wrapped_layers:
                conditional = ConditionalEngramLayer(
                    engram_layer=layer.engram,
                    d_model=engram_model.d_model,
                    use_learned_gate=use_learned_gate,
                )
                # Replace the engram module with conditional version
                layer.engram = conditional
                self.conditional_layers.append(conditional)

    def _detect_and_propagate(self, input_ids: torch.Tensor):
        """Detect lookup pattern and set confidence on all layers."""
        # Decode input to detect pattern
        text = self.tokenizer.decode(input_ids[0], skip_special_tokens=True)
        confidence = self.pattern_detector(text)

        # Propagate to all conditional layers
        for layer in self.conditional_layers:
            layer.set_pattern_confidence(confidence)

        return confidence

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        **kwargs,
    ):
        """Forward pass with conditional Engram routing."""
        # Detect pattern and set confidence
        confidence = self._detect_and_propagate(input_ids)

        # Run the wrapped model (which now has conditional layers)
        return self.engram_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            **kwargs,
        )

    def generate(self, input_ids: torch.Tensor, **kwargs):
        """Generation with conditional Engram routing."""
        # Detect pattern for initial prompt
        confidence = self._detect_and_propagate(input_ids)

        return self.engram_model.generate(input_ids=input_ids, **kwargs)

    def get_routing_stats(self) -> dict:
        """Get statistics about routing decisions."""
        return {
            "num_conditional_layers": len(self.conditional_layers),
            "use_learned_gate": self.use_learned_gate,
            "lookup_prefixes": list(self.pattern_detector.prefixes),
        }


def create_conditional_engram(
    model_name: str = "HuggingFaceTB/SmolLM-135M-Instruct",
    engram_weights_path: Optional[str] = None,
    memory_size: int = 500,
    custom_prefixes: Optional[List[str]] = None,
    use_learned_gate: bool = True,
    device: str = "auto",
):
    """
    Convenience function to create a conditional Engram model.

    Args:
        model_name: HuggingFace model name
        engram_weights_path: Path to pre-trained Engram weights
        memory_size: Memory table size
        custom_prefixes: Additional lookup prefixes
        use_learned_gate: Whether to use learned gating
        device: Device to load model on

    Returns:
        Tuple of (ConditionalEngramWrapper, tokenizer)
    """
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from .model_wrapper import EngramModelWrapper

    # Load base model and tokenizer
    base_model = AutoModelForCausalLM.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token

    # Create Engram wrapper
    engram_model = EngramModelWrapper(
        model=base_model,
        memory_size=memory_size,
        freeze_base=True,
    )

    # Load pre-trained weights if provided
    if engram_weights_path:
        engram_model.load_engram_weights(engram_weights_path)

    # Wrap with conditional routing
    conditional_model = ConditionalEngramWrapper(
        engram_model=engram_model,
        tokenizer=tokenizer,
        custom_prefixes=custom_prefixes,
        use_learned_gate=use_learned_gate,
    )

    # Move to device
    if device != "auto":
        conditional_model = conditional_model.to(device)

    return conditional_model, tokenizer
