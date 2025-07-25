from dataclasses import dataclass
from typing import Dict, Optional, Union


class VoxtralEncoderConfig:
    """
    Configuration class for VoxtralEncoder, compatible with transformers VoxtralEncoderConfig.

    This uses the same parameter names as the native implementation for compatibility.
    """

    model_type = "voxtral_encoder"

    attribute_map = {
        "d_model": "hidden_size",
        "encoder_layers": "num_hidden_layers",
        "encoder_attention_heads": "num_attention_heads",
        "encoder_ffn_dim": "intermediate_size",
        "encoder_layerdrop": "layerdrop",
    }

    def __init__(
        self,
        vocab_size: int = 51866,
        hidden_size: int = 1280,
        intermediate_size: int = 5120,
        num_hidden_layers: int = 32,
        num_attention_heads: int = 20,
        scale_embedding: bool = False,
        activation_function: str = "gelu",
        num_mel_bins: int = 128,
        max_source_positions: int = 1500,
        initializer_range: float = 0.02,
        attention_dropout: float = 0.0,
        dropout: float = 0.0,
        layerdrop: float = 0.0,
        activation_dropout: float = 0.0,
        pad_token_id: int = 0,
        head_dim: int = 64,
        num_key_value_heads: int = 20,
        **kwargs,
    ):
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.scale_embedding = scale_embedding
        self.activation_function = activation_function
        self.num_mel_bins = num_mel_bins
        self.max_source_positions = max_source_positions
        self.initializer_range = initializer_range
        self.attention_dropout = attention_dropout
        self.dropout = dropout
        self.layerdrop = layerdrop
        self.activation_dropout = activation_dropout
        self.pad_token_id = pad_token_id
        self.head_dim = head_dim
        self.num_key_value_heads = num_key_value_heads

        for key, value in kwargs.items():
            setattr(self, key, value)

    @property
    def d_model(self) -> int:
        return self.hidden_size

    @property
    def encoder_layers(self) -> int:
        return self.num_hidden_layers

    @property
    def encoder_attention_heads(self) -> int:
        return self.num_attention_heads

    @property
    def encoder_ffn_dim(self) -> int:
        return self.intermediate_size

    @property
    def encoder_layerdrop(self) -> float:
        return self.layerdrop

    def to_dict(self) -> Dict:
        """Convert config to dictionary."""
        output = {}
        for key, value in self.__dict__.items():
            if not key.startswith("_"):
                output[key] = value
        return output

    @classmethod
    def from_dict(cls, config_dict: Dict) -> "VoxtralEncoderConfig":
        """Create config from dictionary."""
        return cls(**config_dict)


@dataclass
class VoxtralTextConfig:
    """Configuration for Mistral/Llama text decoder."""

    vocab_size: int = 131072
    hidden_size: int = 3072
    intermediate_size: int = 8192
    num_hidden_layers: int = 30
    num_attention_heads: int = 32
    num_key_value_heads: int = 8
    max_position_embeddings: int = 131072
    rms_norm_eps: float = 1e-05
    rope_theta: float = 100000000.0
    rope_scaling: Optional[Dict] = None
    tie_word_embeddings: bool = False
    use_cache: bool = True
    hidden_act: str = "silu"
    initializer_range: float = 0.02
    attention_bias: bool = False
    attention_dropout: float = 0.0
    mlp_bias: bool = False
    head_dim: int = 128
    model_type: str = "llama"
    pretraining_tp: int = 1
    sliding_window: Optional[int] = None

    def to_dict(self) -> Dict:
        """Convert config to dictionary."""
        output = {}
        for key, value in self.__dict__.items():
            if not key.startswith("_") and value is not None:
                output[key] = value
        return output

    @classmethod
    def from_dict(cls, config_dict: Dict) -> "VoxtralTextConfig":
        """Create config from dictionary."""
        known_fields = {
            "vocab_size",
            "hidden_size",
            "intermediate_size",
            "num_hidden_layers",
            "num_attention_heads",
            "num_key_value_heads",
            "max_position_embeddings",
            "rms_norm_eps",
            "rope_theta",
            "rope_scaling",
            "tie_word_embeddings",
            "use_cache",
            "hidden_act",
            "initializer_range",
            "attention_bias",
            "attention_dropout",
            "mlp_bias",
            "head_dim",
            "model_type",
            "pretraining_tp",
            "sliding_window",
        }
        filtered_dict = {k: v for k, v in config_dict.items() if k in known_fields}
        return cls(**filtered_dict)


class VoxtralConfig:
    """
    Configuration class for full Voxtral model, compatible with transformers VoxtralConfig.

    This is used to instantiate a Voxtral model with both audio encoder and text decoder.
    """

    model_type = "voxtral"

    def __init__(
        self,
        audio_config: Optional[Union[VoxtralEncoderConfig, Dict]] = None,
        text_config: Optional[Union[VoxtralTextConfig, Dict]] = None,
        audio_token_id: Optional[int] = 24, 
        projector_hidden_act: str = "gelu",
        pad_token_id: int = 11, 
        bos_token_id: int = 1,
        eos_token_id: int = 2,
        **kwargs,
    ):
        if isinstance(audio_config, dict):
            self.audio_config = VoxtralEncoderConfig(**audio_config)
        elif audio_config is None:
            self.audio_config = VoxtralEncoderConfig()
        else:
            self.audio_config = audio_config

        if isinstance(text_config, dict):
            self.text_config = VoxtralTextConfig(**text_config)
        elif text_config is None:
            self.text_config = VoxtralTextConfig()
        else:
            self.text_config = text_config

        self.audio_token_id = audio_token_id
        self.projector_hidden_act = projector_hidden_act
        self.pad_token_id = pad_token_id
        self.bos_token_id = bos_token_id
        self.eos_token_id = eos_token_id

        self.vocab_size = self.text_config.vocab_size
        self.hidden_size = self.text_config.hidden_size

        for key, value in kwargs.items():
            setattr(self, key, value)

    def to_dict(self) -> Dict:
        """Convert config to dictionary."""
        output = {
            "model_type": self.model_type,
            "audio_config": self.audio_config.to_dict()
            if hasattr(self.audio_config, "to_dict")
            else self.audio_config,
            "text_config": self.text_config.to_dict()
            if hasattr(self.text_config, "to_dict")
            else self.text_config,
            "audio_token_id": self.audio_token_id,
            "projector_hidden_act": self.projector_hidden_act,
        }

        for key, value in self.__dict__.items():
            if key not in output and not key.startswith("_"):
                output[key] = value

        return output

    @classmethod
    def from_dict(cls, config_dict: Dict) -> "VoxtralConfig":
        """Create config from dictionary."""
        return cls(**config_dict)
