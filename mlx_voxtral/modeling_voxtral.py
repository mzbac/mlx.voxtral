import math
from typing import Optional, Tuple, List, Dict, Any, Generator
from dataclasses import dataclass

import mlx.core as mx
import mlx.nn as nn

from .configuration_voxtral import (
    VoxtralEncoderConfig,
    VoxtralConfig,
    VoxtralTextConfig,
)
from .models.llama import LlamaModel
from mlx_lm.models.cache import KVCache
from .utils.model_loading import load_voxtral_model

class VoxtralAttention(nn.Module):
    """Multi-head attention layer for inference."""
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        bias: bool = True,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        if self.head_dim * num_heads != self.embed_dim:
            raise ValueError(
                f"embed_dim must be divisible by num_heads "
                f"(got `embed_dim`: {self.embed_dim} and `num_heads`: {num_heads})."
            )

        self.scale = self.head_dim**-0.5

        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias)

    def __call__(
        self,
        hidden_states: mx.array,
        attention_mask: Optional[mx.array] = None,
        output_attentions: bool = False,
    ) -> Tuple[mx.array, Optional[mx.array]]:
        batch_size, seq_len, _ = hidden_states.shape

        query = self.q_proj(hidden_states)
        key = self.k_proj(hidden_states)
        value = self.v_proj(hidden_states)

        query = query.reshape(
            batch_size, seq_len, self.num_heads, self.head_dim
        ).transpose(0, 2, 1, 3)
        key = key.reshape(batch_size, seq_len, self.num_heads, self.head_dim).transpose(
            0, 2, 1, 3
        )
        value = value.reshape(
            batch_size, seq_len, self.num_heads, self.head_dim
        ).transpose(0, 2, 1, 3)

        attn_output = mx.fast.scaled_dot_product_attention(
            query, key, value, scale=self.scale, mask=attention_mask
        )

        attn_output = attn_output.transpose(0, 2, 1, 3).reshape(
            batch_size, seq_len, self.embed_dim
        )

        attn_output = self.out_proj(attn_output)

        return attn_output, None


class VoxtralEncoderLayer(nn.Module):
    """Voxtral encoder layer for inference."""

    def __init__(self, config: VoxtralEncoderConfig):
        super().__init__()
        self.embed_dim = config.hidden_size
        self.self_attn = VoxtralAttention(
            embed_dim=self.embed_dim,
            num_heads=config.num_attention_heads,
            bias=True,
        )
        self.self_attn_layer_norm = nn.LayerNorm(self.embed_dim)
        self.activation_fn = nn.GELU()
        self.fc1 = nn.Linear(self.embed_dim, config.intermediate_size)
        self.fc2 = nn.Linear(config.intermediate_size, self.embed_dim)
        self.final_layer_norm = nn.LayerNorm(self.embed_dim)

    def __call__(
        self,
        hidden_states: mx.array,
        attention_mask: Optional[mx.array] = None,
        output_attentions: bool = False,
    ) -> Tuple[mx.array, Optional[mx.array]]:
        """
        Args:
            hidden_states: input to the layer of shape (batch, seq_len, embed_dim)
            attention_mask: attention mask of size (batch, 1, seq_len, seq_len)
            output_attentions: whether to return attention weights
        """
        residual = hidden_states
        hidden_states = self.self_attn_layer_norm(hidden_states)
        hidden_states, attn_weights = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            output_attentions=output_attentions,
        )
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.final_layer_norm(hidden_states)
        hidden_states = self.activation_fn(self.fc1(hidden_states))
        hidden_states = self.fc2(hidden_states)
        hidden_states = residual + hidden_states

        return hidden_states, attn_weights


class VoxtralEncoder(nn.Module):
    """
    Transformer encoder for inference consisting of config.num_hidden_layers self attention layers.
    Each layer is a VoxtralEncoderLayer.
    """

    def __init__(self, config: VoxtralEncoderConfig):
        super().__init__()

        embed_dim = config.hidden_size
        self.num_mel_bins = config.num_mel_bins
        self.padding_idx = config.pad_token_id
        self.max_source_positions = config.max_source_positions
        self.embed_scale = math.sqrt(embed_dim) if config.scale_embedding else 1.0

        self.conv1 = nn.Conv1d(self.num_mel_bins, embed_dim, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(embed_dim, embed_dim, kernel_size=3, stride=2, padding=1)

        self.embed_positions = nn.Embedding(self.max_source_positions, embed_dim)

        self.layers = [
            VoxtralEncoderLayer(config) for _ in range(config.num_hidden_layers)
        ]
        self.layer_norm = nn.LayerNorm(embed_dim)

    def __call__(
        self,
        input_features: mx.array,
        attention_mask: Optional[mx.array] = None,
        output_attentions: bool = False,
        output_hidden_states: bool = False,
    ) -> Tuple[mx.array, Optional[Tuple[mx.array]], Optional[Tuple[mx.array]]]:
        """
        Args:
            input_features: Float tensor of shape (batch_size, n_mels, seq_len)
            attention_mask: Attention mask of shape (batch_size, seq_len)
            output_attentions: Whether to return attention weights
            output_hidden_states: Whether to return all hidden states

        Returns:
            Tuple of:
                - last_hidden_state: shape (batch_size, seq_len, hidden_size)
                - hidden_states: Tuple of all hidden states (optional)
                - attentions: Tuple of attention weights (optional)
        """
        # Transpose input from (batch, n_mels, seq_len) to (batch, seq_len, n_mels)
        # MLX Conv1d expects (batch, seq_len, channels)
        hidden_states = input_features.transpose(0, 2, 1)

        hidden_states = nn.gelu(self.conv1(hidden_states))
        hidden_states = nn.gelu(self.conv2(hidden_states))

        seq_len = hidden_states.shape[1]
        embed_pos = self.embed_positions.weight[:seq_len]
        hidden_states = hidden_states + embed_pos

        if attention_mask is not None:
            attention_mask = self._prepare_attention_mask(attention_mask)

        all_hidden_states = () if output_hidden_states else None
        all_attentions = () if output_attentions else None

        for layer in self.layers:
            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            hidden_states, attn_weights = layer(
                hidden_states,
                attention_mask=attention_mask,
                output_attentions=output_attentions,
            )

            if output_attentions:
                all_attentions += (attn_weights,)

        hidden_states = self.layer_norm(hidden_states)

        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        return hidden_states, all_hidden_states, all_attentions

    def _prepare_attention_mask(self, attention_mask: mx.array) -> mx.array:
        """Prepare 4D attention mask from 2D mask."""
        batch_size, seq_len = attention_mask.shape
        # Create 4D mask of shape (batch_size, 1, seq_len, seq_len)
        # where positions with 0 in the original mask get -inf
        attention_mask = attention_mask[:, None, None, :]
        attention_mask = (1.0 - attention_mask) * -1e4
        return mx.broadcast_to(attention_mask, (batch_size, 1, seq_len, seq_len))


class VoxtralMultiModalProjector(nn.Module):
    """Projects audio embeddings to language model space."""

    def __init__(self, config: VoxtralConfig):
        super().__init__()

        self.linear_1 = nn.Linear(
            config.audio_config.intermediate_size,
            config.text_config.hidden_size,
            bias=False,
        )
        self.act = nn.GELU()
        self.linear_2 = nn.Linear(
            config.text_config.hidden_size, config.text_config.hidden_size, bias=False
        )

    def __call__(self, audio_features: mx.array) -> mx.array:
        """
        Args:
            audio_features: Audio features of shape (batch*seq_len//4, intermediate_size)
                           where intermediate_size = hidden_size * 4

        Returns:
            Projected features of shape (batch*seq_len//4, text_hidden_size)
        """
        hidden_states = self.linear_1(audio_features)
        hidden_states = self.act(hidden_states)
        hidden_states = self.linear_2(hidden_states)
        return hidden_states


@dataclass
class VoxtralModelOutput:
    """Output type for Voxtral model."""

    logits: Optional[mx.array] = None
    past_key_values: Optional[List[Tuple[mx.array, mx.array]]] = None
    hidden_states: Optional[Tuple[mx.array, ...]] = None
    attentions: Optional[Tuple[mx.array, ...]] = None
    audio_hidden_states: Optional[mx.array] = None


class VoxtralForConditionalGeneration(nn.Module):
    """Main Voxtral model combining audio encoder and language model."""

    def __init__(self, config: VoxtralConfig):
        super().__init__()
        self.config = config

        if isinstance(config.text_config, dict):
            text_config = VoxtralTextConfig(**config.text_config)
        else:
            text_config = config.text_config

        self.audio_tower = VoxtralEncoder(config.audio_config)
        self.multi_modal_projector = VoxtralMultiModalProjector(config)
        self.language_model = LlamaModel(text_config)

        self.embed_tokens = self.language_model.embed_tokens
        self.lm_head = nn.Linear(
            text_config.hidden_size, text_config.vocab_size, bias=False
        )

        self.text_config = text_config

    def get_audio_embeds(self, input_features: mx.array) -> mx.array:
        """Process audio features through encoder and projector.

        Args:
            input_features: Audio features of shape [num_chunks, 128, 3000]
                           where num_chunks can be > 1 for multi-chunk audio

        Returns:
            audio_embeds: Shape [1, num_chunks * 375, hidden_size]
        """
        num_chunks = input_features.shape[0]

        all_audio_embeds = []

        for i in range(num_chunks):
            chunk_features = input_features[i : i + 1]  # [1, 128, 3000]

            audio_outputs = self.audio_tower(
                chunk_features, output_attentions=False, output_hidden_states=False
            )
            audio_hidden_states = audio_outputs[0]  # [1, 1500, 1280]

            # Reshape for projector: group 4 consecutive frames
            # [1, 1500, 1280] -> [1, 375, 5120]
            audio_hidden_states = audio_hidden_states.reshape(
                1, -1, self.config.audio_config.intermediate_size
            )

            # Flatten for projector
            # [1, 375, 5120] -> [375, 5120]
            audio_hidden_states = audio_hidden_states.reshape(
                -1, self.config.audio_config.intermediate_size
            )

            # Project to language model space
            chunk_embeds = self.multi_modal_projector(
                audio_hidden_states
            )  # [375, hidden_size]

            all_audio_embeds.append(chunk_embeds)

        # Concatenate all chunks
        # List of [375, hidden_size] -> [num_chunks * 375, hidden_size]
        audio_embeds = mx.concatenate(all_audio_embeds, axis=0)

        # [num_chunks * 375, hidden_size] -> [1, num_chunks * 375, hidden_size]
        audio_embeds = audio_embeds[None, :, :]


        return audio_embeds

    def _merge_input_embeddings(
        self,
        input_ids: Optional[mx.array] = None,
        input_features: Optional[mx.array] = None,
        inputs_embeds: Optional[mx.array] = None,
    ) -> mx.array:
        """Merge text and audio embeddings."""
        if inputs_embeds is None:
            if input_ids is None:
                raise ValueError("Either input_ids or inputs_embeds must be provided")
            inputs_embeds = self.embed_tokens(input_ids)

        if input_features is None or input_ids is None:
            return inputs_embeds

        audio_embeds = self.get_audio_embeds(input_features)

        audio_token_mask = input_ids == self.config.audio_token_id

        batch_size = input_ids.shape[0]
        seq_length = input_ids.shape[1]

        for i in range(batch_size):
            batch_mask = audio_token_mask[i]
            audio_positions = []
            for j in range(seq_length):
                if batch_mask[j]:
                    audio_positions.append(j)

            if len(audio_positions) > 0:
                expected_audio_tokens = audio_embeds.shape[1]
                if len(audio_positions) != expected_audio_tokens:
                    raise ValueError(
                        f"Batch {i}: Expected {expected_audio_tokens} audio tokens "
                        f"but found {len(audio_positions)} in input_ids"
                    )

                batch_embeds = inputs_embeds[i]
                new_embeds = []
                audio_idx = 0

                for j in range(seq_length):
                    if j in audio_positions:
                        new_embeds.append(audio_embeds[i, audio_idx])
                        audio_idx += 1
                    else:
                        new_embeds.append(batch_embeds[j])

                new_batch_embeds = mx.stack(new_embeds)

                if i == 0:
                    inputs_embeds = mx.concatenate(
                        [new_batch_embeds[None, :, :], inputs_embeds[1:]], axis=0
                    )
                elif i == batch_size - 1:
                    inputs_embeds = mx.concatenate(
                        [inputs_embeds[:i], new_batch_embeds[None, :, :]], axis=0
                    )
                else:
                    inputs_embeds = mx.concatenate(
                        [
                            inputs_embeds[:i],
                            new_batch_embeds[None, :, :],
                            inputs_embeds[i + 1 :],
                        ],
                        axis=0,
                    )

        return inputs_embeds

    def __call__(
        self,
        input_ids: Optional[mx.array] = None,
        attention_mask: Optional[mx.array] = None,
        input_features: Optional[mx.array] = None,
        inputs_embeds: Optional[mx.array] = None,
        labels: Optional[mx.array] = None,
        past_key_values: Optional[List[KVCache]] = None,
        return_dict: bool = True,
    ) -> VoxtralModelOutput:
        """Forward pass of Voxtral model."""

        inputs_embeds = self._merge_input_embeddings(
            input_ids=input_ids,
            input_features=input_features,
            inputs_embeds=inputs_embeds,
        )

        hidden_states = self.language_model(
            inputs_embeds=inputs_embeds,
            mask=attention_mask,
            cache=past_key_values,
        )

        logits = self.lm_head(hidden_states)

        loss = None
        if labels is not None:
            shift_logits = logits[..., :-1, :].reshape(-1, logits.shape[-1])
            shift_labels = labels[..., 1:].reshape(-1)
            loss = nn.losses.cross_entropy(shift_logits, shift_labels)

        if not return_dict:
            output = (logits,)
            return ((loss,) + output) if loss is not None else output

        return VoxtralModelOutput(
            logits=logits,
            past_key_values=past_key_values,  
            hidden_states=None,  
            attentions=None,
        )

    def generate_stream(
        self,
        input_ids: Optional[mx.array] = None,
        input_features: Optional[mx.array] = None,
        attention_mask: Optional[mx.array] = None,
        max_new_tokens: int = 128,
        temperature: float = 0.0,
        top_p: float = 0.95,
        top_k: int = 0,
        min_p: float = 0.0,
        repetition_penalty: float = 1.2,
        logit_bias: Optional[Dict[int, float]] = None,
        stop_tokens: Optional[List[int]] = None,
        **kwargs,
    ) -> Generator[Tuple[mx.array, mx.array], None, None]:
        """Generate text from audio features.
        
        Args:
            input_ids: Input token IDs with audio tokens
            input_features: Audio features from mel spectrogram
            attention_mask: Attention mask for input
            max_new_tokens: Maximum number of tokens to generate
            temperature: Sampling temperature (0.0 for greedy)
            top_p: Top-p (nucleus) sampling parameter
            top_k: Top-k sampling parameter
            min_p: Minimum probability threshold
            repetition_penalty: Penalty for repeated tokens
            logit_bias: Dictionary mapping token IDs to bias values
            stop_tokens: List of token IDs that stop generation
            **kwargs: Additional arguments
            
        Yields:
            Tuple of (token, logprobs) for each generated token
        """
        if input_ids is None:
            raise ValueError("input_ids must be provided")
            
        if stop_tokens is None:
            stop_tokens = [2, 4, 32000]  # </s>, [/INST], and potential padding token
        
        inputs_embeds = self._merge_input_embeddings(
            input_ids=input_ids,
            input_features=input_features,
        )
        
        batch_size = input_ids.shape[0]
        generated = input_ids
        
        cache = [KVCache() for _ in range(len(self.language_model.layers))]
        
        def apply_repetition_penalty(logits, tokens):
            if repetition_penalty == 1.0 or len(tokens) == 0:
                return logits
            
            unique_tokens = list(set(tokens))
            selected_logits = logits[:, unique_tokens]
            
            selected_logits = mx.where(
                selected_logits < 0,
                selected_logits * repetition_penalty,
                selected_logits / repetition_penalty,
            )
            logits[:, unique_tokens] = selected_logits
            return logits
        
        def apply_logit_bias(logits):
            if logit_bias:
                indices = mx.array(list(logit_bias.keys()))
                values = mx.array(list(logit_bias.values()))
                logits[:, indices] += values
            return logits
        
        def sample(logits):
            if temperature == 0:
                return mx.argmax(logits, axis=-1, keepdims=True)
            
            if temperature != 1.0:
                logits = logits / temperature
            
            logprobs = logits - mx.logsumexp(logits, axis=-1, keepdims=True)
            
            if top_k > 0:
                vocab_size = logprobs.shape[-1]
                k = min(top_k, vocab_size)
                mask_idx = mx.argpartition(-logprobs, kth=k - 1, axis=-1)[..., k:]
                logprobs = mx.put_along_axis(
                    logprobs, mask_idx, mx.array(-float("inf"), logprobs.dtype), axis=-1
                )
            
            if top_p < 1.0:
                probs = mx.exp(logprobs)
                sorted_indices = mx.argsort(logprobs, axis=-1)
                sorted_probs = mx.take_along_axis(probs, sorted_indices, axis=-1)
                
                cumulative_probs = mx.cumsum(sorted_probs, axis=-1)
                
                inverse_indices = mx.zeros_like(sorted_indices)
                batch_indices = mx.arange(batch_size)[:, None]
                inverse_indices[batch_indices, sorted_indices] = mx.arange(sorted_indices.shape[-1])
                
                cumulative_probs = mx.take_along_axis(cumulative_probs, inverse_indices, axis=-1)
                
                logprobs = mx.where(
                    cumulative_probs > 1 - top_p,
                    logprobs,
                    -float("inf"),
                )
            
            if min_p > 0:
                import math
                sorted_indices = mx.argsort(-logprobs, axis=-1)
                sorted_logprobs = mx.take_along_axis(logprobs, sorted_indices, axis=-1)
                
                top_logprobs = sorted_logprobs[:, 0:1]
                
                scaled_min_p = top_logprobs + math.log(min_p)
                
                tokens_to_remove = sorted_logprobs < scaled_min_p
                tokens_to_remove[:, :1] = False  
                
                selected_logprobs = mx.where(tokens_to_remove, -float("inf"), sorted_logprobs)
                
                inverse_indices = mx.zeros_like(sorted_indices)
                batch_indices = mx.arange(batch_size)[:, None]
                inverse_indices[batch_indices, sorted_indices] = mx.arange(sorted_indices.shape[-1])
                
                logprobs = mx.take_along_axis(selected_logprobs, inverse_indices, axis=-1)
            
            return mx.random.categorical(logprobs)[:, None]
        
        generated_tokens = []
        
        consecutive_repeats = 0
        last_token = None
        
        for _ in range(max_new_tokens):
               
            if cache[0].offset == 0:
                outputs = self(
                    inputs_embeds=inputs_embeds,
                    attention_mask=attention_mask,
                    past_key_values=cache,
                )
            else:
                outputs = self(
                    input_ids=generated[:, -1:],
                    attention_mask=attention_mask,
                    past_key_values=cache,
                )
            
            next_token_logits = outputs.logits[:, -1, :]
            
            if repetition_penalty != 1.0 and len(generated_tokens) > 0:
                token_list = [t.item() if hasattr(t, 'item') else int(t) for t in generated_tokens[-20:]]
                next_token_logits = apply_repetition_penalty(next_token_logits, token_list)
            
            next_token_logits = apply_logit_bias(next_token_logits)
            
            next_tokens = sample(next_token_logits)
            
            generated = mx.concatenate([generated, next_tokens], axis=1)
            current_token = next_tokens[0, 0]
            generated_tokens.append(current_token)
            
            if attention_mask is not None:
                attention_mask = mx.concatenate(
                    [
                        attention_mask,
                        mx.ones((batch_size, 1), dtype=attention_mask.dtype),
                    ],
                    axis=1,
                )
            
            logprobs = next_token_logits - mx.logsumexp(next_token_logits, axis=-1, keepdims=True)
            
            yield current_token, logprobs.squeeze(0)
            
            current_token_id = current_token.item() if hasattr(current_token, 'item') else int(current_token)
            if current_token_id in stop_tokens:
                break
                
            if last_token is not None and current_token_id == last_token:
                consecutive_repeats += 1
                if consecutive_repeats >= 10:  # Stop if same token repeated 10 times
                    print(f"Warning: Stopping due to repetition of token {current_token_id}")
                    break
            else:
                consecutive_repeats = 0
            last_token = current_token_id

    def generate(
        self,
        input_ids: Optional[mx.array] = None,
        input_features: Optional[mx.array] = None,
        attention_mask: Optional[mx.array] = None,
        max_new_tokens: int = 128,
        temperature: float = 0.0,
        top_p: float = 0.95,
        top_k: int = 0,
        min_p: float = 0.0,
        repetition_penalty: float = 1.2,
        logit_bias: Optional[Dict[int, float]] = None,
        stop_tokens: Optional[List[int]] = None,
        **kwargs,
    ) -> mx.array:
        """Generate text from audio features (non-streaming).
        
        Args:
            input_ids: Input token IDs with audio tokens
            input_features: Audio features from mel spectrogram
            attention_mask: Attention mask for input
            max_new_tokens: Maximum number of tokens to generate
            temperature: Sampling temperature (0.0 for greedy)
            top_p: Top-p (nucleus) sampling parameter
            top_k: Top-k sampling parameter
            min_p: Minimum probability threshold
            repetition_penalty: Penalty for repeated tokens
            logit_bias: Dictionary mapping token IDs to bias values
            stop_tokens: List of token IDs that stop generation
            **kwargs: Additional arguments
            
        Returns:
            Generated token sequence including input
        """
        generated = input_ids
        for token, _ in self.generate_stream(
            input_ids=input_ids,
            input_features=input_features,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            min_p=min_p,
            repetition_penalty=repetition_penalty,
            logit_bias=logit_bias,
            stop_tokens=stop_tokens,
            **kwargs,
        ):
            generated = mx.concatenate([generated, token[None, None]], axis=1)
        
        return generated

    def prepare_inputs_for_generation(
        self,
        input_ids: mx.array,
        past_key_values: Optional[List[Tuple[mx.array, mx.array]]] = None,
        attention_mask: Optional[mx.array] = None,
        input_features: Optional[mx.array] = None,
    ) -> Dict[str, Any]:
        """Prepare inputs for generation step."""

        if past_key_values is not None:
            model_inputs = {
                "input_ids": input_ids[:, -1:], 
                "input_features": None,  
                "attention_mask": attention_mask,
                "past_key_values": past_key_values,
            }
        else:
            model_inputs = {
                "input_ids": input_ids,
                "input_features": input_features,
                "attention_mask": attention_mask,
                "past_key_values": past_key_values,
            }

        return model_inputs

    def sanitize(self, weights: Dict[str, mx.array]) -> Dict[str, mx.array]:
        """Sanitize weights for loading into the model.

        This method is called by MLX's load_weights to transform weight names
        and formats from the source model to match this implementation.
        """
        import logging

        logger = logging.getLogger(__name__)

        sanitized = {}
        rotary_count = 0

        for key, value in weights.items():
            if any(
                x in key
                for x in [
                    "rotary_emb.inv_freq",
                    "rotary_emb.cos_cached",
                    "rotary_emb.sin_cached",
                ]
            ):
                rotary_count += 1
                continue
            if "position_ids" in key:
                continue

            new_key = key

            if key.startswith("language_model.model."):
                new_key = key.replace("language_model.model.", "language_model.")
                logger.debug(f"Mapping {key} -> {new_key}")

            elif key == "language_model.lm_head.weight":
                new_key = "lm_head.weight"
                logger.debug(f"Mapping {key} -> {new_key}")

            if "conv" in key and "weight" in key and value.ndim == 3:
                # Conv1d weight: (out_channels, kernel_size, in_channels)
                if value.shape[1] != 3:  # kernel_size should be 3
                    value = value.transpose(0, 2, 1)

            sanitized[new_key] = value

        logger.info(f"Filtered out {rotary_count} rotary embedding weights")

        if (
            "language_model.embed_tokens.weight" in sanitized
            and "embed_tokens.weight" not in sanitized
        ):
            sanitized["embed_tokens.weight"] = sanitized[
                "language_model.embed_tokens.weight"
            ]
            logger.info(
                "Copied language_model.embed_tokens.weight to embed_tokens.weight (shared weight)"
            )


        logger.info(f"Sanitized {len(weights)} weights to {len(sanitized)} weights")

        return sanitized

    @classmethod
    def from_pretrained(
        cls, model_id: str, dtype: mx.Dtype = mx.float16
    ) -> "VoxtralForConditionalGeneration":
        """Load pretrained Voxtral model from Hugging Face or local path.

        Args:
            model_id: Hugging Face model ID or local path
            dtype: Data type for model weights
            **kwargs: Additional arguments

        Returns:
            Loaded VoxtralForConditionalGeneration model
        """

        model, _ = load_voxtral_model(model_id, dtype=dtype)
        return model
