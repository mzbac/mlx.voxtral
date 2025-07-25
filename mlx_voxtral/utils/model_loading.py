"""Model loading utilities for Voxtral MLX implementation."""

import json
import logging
from pathlib import Path
from typing import Dict, Optional, Tuple, Union

import mlx.core as mx
import mlx.nn as nn
from huggingface_hub import snapshot_download

logger = logging.getLogger(__name__)


def download_model(model_id: str, revision: Optional[str] = None) -> Path:
    """Download model from Hugging Face Hub."""
    model_path = Path(
        snapshot_download(
            repo_id=model_id,
            revision=revision,
            allow_patterns=["*.safetensors", "*.json", "config.json", "tekken.json", "params.json"],
            ignore_patterns=["consolidated.safetensors", "consolidated.*.safetensors"],
        )
    )
    return model_path


def load_config(model_path: Path) -> Dict:
    """Load model configuration from path."""
    config_path = model_path / "config.json"
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(config_path, "r") as f:
        config = json.load(f)

    return config


def load_weights(model_path: Path) -> Dict[str, mx.array]:
    """Load weights from safetensors files."""
    weights = {}

    weight_files = sorted(
        [f for f in model_path.glob("*.safetensors") 
         if not f.name.startswith("._") 
         and not f.name.startswith("consolidated")
         and "consolidated." not in f.name]
    )

    if not weight_files:
        raise FileNotFoundError(f"No weight files found in {model_path}")

    logger.info(f"Loading weights from {len(weight_files)} files")

    for wf in weight_files:
        logger.debug(f"Loading {wf}")
        weights.update(mx.load(str(wf)))

    return weights


def load_voxtral_model(
    model_path: Union[str, Path],
    dtype: mx.Dtype = mx.float16,
    lazy: bool = True,
) -> Tuple[nn.Module, Dict]:
    """Load Voxtral model from path or Hugging Face.

    Args:
        model_path: Path to model directory or Hugging Face model ID
        dtype: Data type for model weights
        lazy: Whether to use lazy weight loading

    Returns:
        Tuple of (model, config)
    """
    from ..modeling_voxtral import VoxtralForConditionalGeneration
    from ..configuration_voxtral import VoxtralConfig

    path = Path(model_path) if isinstance(model_path, str) else model_path
    if not path.exists():
        logger.info(f"Downloading model from Hugging Face: {model_path}")
        model_path = download_model(model_path)
    else:
        model_path = path

    config_dict = load_config(model_path)

    config = VoxtralConfig(
        audio_config=config_dict.get("audio_config", {}),
        text_config=config_dict.get("text_config", {}),
        audio_token_id=config_dict.get("audio_token_id"),
        projector_hidden_act=config_dict.get("projector_hidden_act", "gelu"),
    )

    logger.info("Initializing model")
    model = VoxtralForConditionalGeneration(config)

    logger.info("Loading weights")
    weights = load_weights(model_path)
    
    if "quantization" in config_dict:
        logger.info("Loading quantized model - applying quantization structure")
        from ..quantization import load_quantized_voxtral
        model = load_quantized_voxtral(model, weights, config_dict)

    logger.info("Calling model.sanitize on weights")
    weights = model.sanitize(weights)

    logger.info("Model structure:")
    for name, module in model.children().items():
        logger.info(f"  {name}: {type(module).__name__}")

    if dtype is not None and "quantization" not in config_dict:
        converted_weights = {}
        for name, weight in weights.items():
            if isinstance(weight, mx.array) and "embed_tokens" not in name:
                converted_weights[name] = weight.astype(dtype)
            else:
                converted_weights[name] = weight
        weights = converted_weights

    logger.info(f"Attempting to load {len(weights)} weights into model")
    model.load_weights(list(weights.items()), strict=True)

    def count_params(params_dict):
        total = 0
        count = 0
        for name, value in params_dict.items():
            if isinstance(value, dict):
                sub_total, sub_count = count_params(value)
                total += sub_total
                count += sub_count
            elif isinstance(value, mx.array):
                total += value.size
                count += 1
        return total, count

    total_params, param_count = count_params(model.parameters())
    logger.info(
        f"Model has {param_count} parameter arrays with {total_params:,} total parameters"
    )
    
    if not lazy:
        mx.eval(model.parameters())

    return model, config_dict
