"""MLX implementation of Llama model for Voxtral."""

from typing import Optional, List

import mlx.core as mx
import mlx.nn as nn

from mlx_lm.models.base import create_attention_mask, scaled_dot_product_attention
from mlx_lm.models.rope_utils import initialize_rope
from mlx_lm.models.cache import KVCache


class LlamaAttention(nn.Module):
    """Multi-headed attention with rotary embeddings."""

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = (
            config.head_dim
            if hasattr(config, "head_dim")
            else (self.hidden_size // self.num_heads)
        )
        self.num_key_value_heads = config.num_key_value_heads
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        self.max_position_embeddings = config.max_position_embeddings
        self.rope_theta = config.rope_theta
        self.is_causal = True
        self.scale = self.head_dim**-0.5

        self.q_proj = nn.Linear(
            self.hidden_size, self.num_heads * self.head_dim, bias=config.attention_bias
        )
        self.k_proj = nn.Linear(
            self.hidden_size,
            self.num_key_value_heads * self.head_dim,
            bias=config.attention_bias,
        )
        self.v_proj = nn.Linear(
            self.hidden_size,
            self.num_key_value_heads * self.head_dim,
            bias=config.attention_bias,
        )
        self.o_proj = nn.Linear(
            self.num_heads * self.head_dim, self.hidden_size, bias=config.attention_bias
        )

        self.rope = initialize_rope(
            self.head_dim,
            self.rope_theta,
            config.rope_traditional if hasattr(config, "rope_traditional") else False,
            config.rope_scaling if hasattr(config, "rope_scaling") else None,
            self.max_position_embeddings,
        )

    def __call__(
        self,
        hidden_states: mx.array,
        attention_mask: Optional[mx.array] = None,
        cache: Optional[KVCache] = None,
    ) -> mx.array:
        bsz, q_len, _ = hidden_states.shape

        queries, keys, values = (
            self.q_proj(hidden_states),
            self.k_proj(hidden_states),
            self.v_proj(hidden_states),
        )

        queries = queries.reshape(bsz, q_len, self.num_heads, -1).transpose(0, 2, 1, 3)
        keys = keys.reshape(bsz, q_len, self.num_key_value_heads, -1).transpose(
            0, 2, 1, 3
        )
        values = values.reshape(bsz, q_len, self.num_key_value_heads, -1).transpose(
            0, 2, 1, 3
        )

        if cache is not None:
            queries = self.rope(queries, offset=cache.offset)
            keys = self.rope(keys, offset=cache.offset)
            keys, values = cache.update_and_fetch(keys, values)
        else:
            queries = self.rope(queries)
            keys = self.rope(keys)

        output = scaled_dot_product_attention(
            queries, keys, values, cache=cache, scale=self.scale, mask=attention_mask
        )

        output = output.transpose(0, 2, 1, 3).reshape(bsz, q_len, -1)
        return self.o_proj(output)


class LlamaMLP(nn.Module):
    """Llama MLP with SiLU activation."""

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size

        self.gate_proj = nn.Linear(
            self.hidden_size, self.intermediate_size, bias=config.mlp_bias
        )
        self.up_proj = nn.Linear(
            self.hidden_size, self.intermediate_size, bias=config.mlp_bias
        )
        self.down_proj = nn.Linear(
            self.intermediate_size, self.hidden_size, bias=config.mlp_bias
        )

        self.act_fn = nn.silu

    def __call__(self, hidden_states: mx.array) -> mx.array:
        return self.down_proj(
            self.act_fn(self.gate_proj(hidden_states)) * self.up_proj(hidden_states)
        )


class LlamaDecoderLayer(nn.Module):
    """Llama decoder layer."""

    def __init__(self, config):
        super().__init__()
        self.hidden_size = config.hidden_size

        self.self_attn = LlamaAttention(config)
        self.mlp = LlamaMLP(config)
        self.input_layernorm = nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = nn.RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )

    def __call__(
        self,
        hidden_states: mx.array,
        attention_mask: Optional[mx.array] = None,
        cache: Optional[KVCache] = None,
    ) -> mx.array:
        r = self.self_attn(self.input_layernorm(hidden_states), attention_mask, cache)
        h = hidden_states + r
        r = self.mlp(self.post_attention_layernorm(h))
        out = h + r
        return out


class LlamaModel(nn.Module):
    """Llama model for Voxtral text generation."""

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.padding_idx = getattr(config, "pad_token_id", None)
        self.vocab_size = config.vocab_size

        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = [
            LlamaDecoderLayer(config) for _ in range(config.num_hidden_layers)
        ]
        self.norm = nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        
    def __call__(
        self,
        inputs: Optional[mx.array] = None,
        mask: Optional[mx.array] = None,
        cache: Optional[List[KVCache]] = None,
        inputs_embeds: Optional[mx.array] = None,
    ) -> mx.array:
        if inputs_embeds is not None:
            h = inputs_embeds
        else:
            h = self.embed_tokens(inputs)

        if mask is None:
            mask = create_attention_mask(h, cache)

        if cache is None:
            cache = [None] * len(self.layers)

        for layer, c in zip(self.layers, cache):
            h = layer(h, mask, cache=c)

        return self.norm(h)
