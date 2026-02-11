"""
Enhanced Engram Module

O(1) hash-based memory lookup for transformer models.
Ported from weagan/Engram implementation.

The key innovation is using deterministic hashing to map input tokens
to memory slots, enabling constant-time retrieval regardless of sequence length.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class EnhancedEngramModule(nn.Module):
    """
    Enhanced Engram with proper initialization and content-based addressing.

    This module provides explicit, large-capacity memory storage with O(1) lookup.
    It uses multi-head hashing to reduce collisions and a gating mechanism to
    control how much retrieved memory influences the output.

    Args:
        table_size: Number of memory slots (default: 100000)
        d_model: Hidden dimension size (default: 512)
        n_heads: Number of hash heads for reduced collisions (default: 4)
        init_scale: Standard deviation for memory initialization (default: 0.02)

    Example:
        >>> module = EnhancedEngramModule(table_size=50000, d_model=768)
        >>> hidden_states = torch.randn(2, 128, 768)  # [batch, seq_len, d_model]
        >>> input_ids = torch.randint(0, 50000, (2, 128))  # [batch, seq_len]
        >>> output = module(hidden_states, input_ids)
    """

    # Prime multipliers for multi-head hashing (reduces collisions)
    HASH_PRIMES = [17, 31, 53, 79, 107, 131, 157, 181]

    def __init__(
        self,
        table_size: int = 100000,
        d_model: int = 512,
        n_heads: int = 4,
        init_scale: float = 0.02,
    ):
        super().__init__()
        self.table_size = table_size
        self.d_model = d_model
        self.n_heads = n_heads

        # Initialize memory table with small values
        # The memory table is the core of Engram - a large learnable lookup table
        self.memory_table = nn.Parameter(torch.zeros(table_size, d_model))
        nn.init.normal_(self.memory_table, mean=0.0, std=init_scale)

        # Content-based addressing (optional enhancement)
        # Allows the model to learn which retrieved memories are most relevant
        self.query_proj = nn.Linear(d_model, d_model)
        self.key_proj = nn.Linear(d_model, d_model)

        # Gating mechanism - learns when to trust memory vs hidden state
        # This is critical: gate → 0 means "ignore memory", gate → 1 means "use memory"
        self.gate = nn.Sequential(
            nn.Linear(d_model * 2, d_model),  # Combine hidden + retrieved
            nn.ReLU(),
            nn.Linear(d_model, 1),
            nn.Sigmoid(),
        )

        # Initialize gate to output near-zero by default
        # This ensures untrained memory doesn't corrupt the model's output
        # sigmoid(-5) ≈ 0.007, so memory contribution starts negligible
        with torch.no_grad():
            self.gate[2].bias.fill_(-5.0)
            self.gate[2].weight.fill_(0.01)  # Small weights for gradual learning

        # Merge projection for residual connection
        self.merge_proj = nn.Linear(d_model, d_model)

        # Initialize merge projection to small values
        nn.init.normal_(self.merge_proj.weight, std=0.01)
        nn.init.zeros_(self.merge_proj.bias)

    def multi_head_hash(self, input_ids: torch.Tensor) -> torch.Tensor:
        """
        Deterministic multi-head hashing for memory indices.

        Using multiple hash heads with different prime multipliers reduces
        the chance of collisions - different inputs mapping to the same slot.

        Args:
            input_ids: Token IDs [batch_size, seq_len]

        Returns:
            Hash indices [batch_size, seq_len, n_heads]
        """
        hashes = []
        for i in range(self.n_heads):
            # Use different prime multipliers for each head
            prime = self.HASH_PRIMES[i % len(self.HASH_PRIMES)]
            hash_val = (input_ids * prime) % self.table_size
            hashes.append(hash_val)
        return torch.stack(hashes, dim=-1)

    def forward(
        self,
        hidden_states: torch.Tensor,
        input_ids: torch.Tensor,
        use_content_addressing: bool = False,
    ) -> torch.Tensor:
        """
        Forward pass with memory retrieval and gated injection.

        Args:
            hidden_states: Current hidden states [batch_size, seq_len, d_model]
            input_ids: Token IDs for hashing [batch_size, seq_len]
            use_content_addressing: Whether to use attention over retrieved memories

        Returns:
            Updated hidden states with memory [batch_size, seq_len, d_model]
        """
        batch_size, seq_len, _ = hidden_states.shape

        # Step 1: Hash input tokens to memory indices
        indices = self.multi_head_hash(input_ids)  # [B, S, n_heads]

        # Step 2: O(1) retrieval from memory table
        retrieved_mem = F.embedding(indices, self.memory_table)  # [B, S, n_heads, d_model]

        # Step 3: Combine multi-head retrievals
        if use_content_addressing:
            # Attention-based combination (more powerful but slower)
            queries = self.query_proj(hidden_states).unsqueeze(2)  # [B, S, 1, d_model]
            keys = self.key_proj(retrieved_mem)  # [B, S, n_heads, d_model]
            attention_scores = torch.matmul(queries, keys.transpose(-1, -2))  # [B, S, 1, n_heads]
            attention_weights = F.softmax(attention_scores, dim=-1)
            retrieved_mem = torch.sum(attention_weights * retrieved_mem, dim=2)  # [B, S, d_model]
        else:
            # Simple mean pooling (faster)
            retrieved_mem = retrieved_mem.mean(dim=2)

        # Step 4: Adaptive gating - learn when to use memory
        gate_input = torch.cat([hidden_states, retrieved_mem], dim=-1)
        gate_score = self.gate(gate_input)  # [B, S, 1]
        gated_memory = retrieved_mem * gate_score

        # Step 5: Residual connection
        output = hidden_states + self.merge_proj(gated_memory)

        return output

    def get_memory_stats(self) -> dict:
        """Get statistics about the memory table for debugging."""
        with torch.no_grad():
            return {
                "table_size": self.table_size,
                "d_model": self.d_model,
                "n_heads": self.n_heads,
                "memory_mean": self.memory_table.mean().item(),
                "memory_std": self.memory_table.std().item(),
                "memory_norm": self.memory_table.norm().item(),
                "num_parameters": sum(p.numel() for p in self.parameters()),
            }


class EngramLayer(nn.Module):
    """
    A complete Engram-enhanced transformer layer.

    Combines self-attention with Engram memory retrieval.
    Can be used to build a full Engram-enhanced transformer.
    """

    def __init__(
        self,
        d_model: int = 512,
        n_heads: int = 8,
        memory_size: int = 50000,
        dropout: float = 0.1,
    ):
        super().__init__()

        # Self-attention
        self.attention = nn.MultiheadAttention(
            d_model, n_heads, batch_first=True, dropout=dropout
        )

        # Engram memory module
        self.engram = EnhancedEngramModule(
            table_size=memory_size,
            d_model=d_model,
            n_heads=4,
        )

        # Feed-forward network
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.ReLU(),
            nn.Linear(d_model * 4, d_model),
            nn.Dropout(dropout),
        )

        # Layer norms
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)

        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        hidden_states: torch.Tensor,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        # Self-attention with residual
        attn_output, _ = self.attention(hidden_states, hidden_states, hidden_states)
        hidden_states = self.norm1(hidden_states + self.dropout(attn_output))

        # Engram memory retrieval
        hidden_states = self.engram(hidden_states, input_ids)
        hidden_states = self.norm2(hidden_states)

        # FFN with residual
        ffn_output = self.ffn(hidden_states)
        hidden_states = self.norm3(hidden_states + self.dropout(ffn_output))

        return hidden_states
