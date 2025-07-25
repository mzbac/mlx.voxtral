"""
MLX Voxtral - Audio processing and model implementation for Voxtral using MLX

This package provides optimized implementations of Voxtral models and audio processing
for Apple Silicon using the MLX framework.
"""

from .audio_processing import (
    VoxtralFeatureExtractor,
    process_audio_for_voxtral,
    process_audio_chunk,
    load_audio,
    SAMPLE_RATE,
    N_MELS,
    N_FFT,
    HOP_LENGTH,
    CHUNK_LENGTH,
)
from .configuration_voxtral import (
    VoxtralConfig,
    VoxtralEncoderConfig,
    VoxtralTextConfig,
)
from .modeling_voxtral import (
    VoxtralEncoder,
    VoxtralEncoderLayer,
    VoxtralAttention,
    VoxtralMultiModalProjector,
    VoxtralForConditionalGeneration,
    VoxtralModelOutput,
)
from .processing_voxtral import VoxtralProcessor

__version__ = "0.1.0"

__all__ = [
    # Audio processing
    "VoxtralFeatureExtractor",
    "process_audio_for_voxtral",
    "process_audio_chunk",
    "load_audio",
    "SAMPLE_RATE",
    "N_MELS",
    "N_FFT",
    "HOP_LENGTH",
    "CHUNK_LENGTH",
    # Configuration
    "VoxtralConfig",
    "VoxtralEncoderConfig",
    "VoxtralTextConfig",
    # Models
    "VoxtralEncoder",
    "VoxtralEncoderLayer",
    "VoxtralAttention",
    "VoxtralMultiModalProjector",
    "VoxtralForConditionalGeneration",
    "VoxtralModelOutput",
    # Processing
    "VoxtralProcessor",
]

# Import utilities for convenience
from .utils import load_voxtral_model  # noqa: F401

__all__.extend(["load_voxtral_model"])
