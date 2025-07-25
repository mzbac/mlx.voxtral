"""
Quantization for Voxtral using mlx_lm utilities directly.

This approach leverages mlx_lm's existing quantization infrastructure
instead of duplicating code.
"""

from typing import Union, Dict
import mlx.nn as nn

import mlx_lm.utils

quantize_model = mlx_lm.utils.quantize_model
save_config = mlx_lm.utils.save_config
save_model = mlx_lm.utils.save_model

def compute_bits_per_weight(model):
    """Compute average bits per weight, handling different mlx_lm versions."""
    try:
        return mlx_lm.utils.compute_bits_per_weight(model)
    except (AttributeError, ZeroDivisionError):
        from mlx.utils import tree_flatten, tree_reduce
        import mlx.core as mx
        
        if hasattr(model, 'parameters'):
            params = model.parameters()
        else:
            params = model
            
        model_bytes = tree_reduce(
            lambda acc, x: acc + x.nbytes if isinstance(x, mx.array) else acc, 
            params, 
            0
        )
        
        params_flat = tree_flatten(params)
        model_params = sum(p.size for _, p in params_flat if isinstance(p, mx.array))
        
        if model_params == 0:
            return 0.0
            
        return model_bytes * 8 / model_params


__all__ = [
    "quantize_model", 
    "save_model", 
    "save_config",
    "compute_bits_per_weight",
    "load_quantized_voxtral",
    "voxtral_mixed_quantization_predicate"
]


def load_quantized_voxtral(
    model: nn.Module,
    weights: Dict,
    config: Dict,
) -> nn.Module:
    """
    Apply quantization to a Voxtral model based on saved configuration.
    
    This uses the same approach as mlx_lm for consistency.
    
    Args:
        model: The initialized Voxtral model
        weights: Loaded weights dictionary
        config: Model configuration with quantization info
        
    Returns:
        The model with quantization applied (weights still need to be loaded)
    """
    if "quantization" not in config:
        return model
    
    quantization = config["quantization"]
    
    def class_predicate(p, m):
        if p in quantization:
            return quantization[p]
        if not hasattr(m, "to_quantized"):
            return False
        if not weights:
            return True
        return f"{p}.scales" in weights
    
    nn.quantize(
        model,
        group_size=quantization["group_size"],
        bits=quantization["bits"],
        class_predicate=class_predicate,
    )
    
    return model


def voxtral_mixed_quantization_predicate(
    path: str,
    module: nn.Module,
    config: dict,
    default_bits: int = 4
) -> Union[bool, dict]:
    """Custom predicate for Voxtral mixed quantization."""
    if not hasattr(module, "to_quantized"):
        return False
    
    # Skip positional embeddings
    if "embed_positions" in path or "pos_emb" in path:
        return False
    
    if hasattr(module, "weight") and module.weight.shape[-1] % 64 != 0:
        if module.weight.shape[-1] % 32 == 0:
            group_size = 32
        else:
            return False
    else:
        group_size = 64
    
    # Output layer - always higher precision
    if "lm_head" in path:
        return {"group_size": min(128, module.weight.shape[-1]), "bits": min(8, default_bits + 2)}
    
    # Audio encoder and projector - always higher precision
    if any(x in path for x in ["audio_tower.", "multi_modal_projector."]):
        return {"group_size": group_size, "bits":  min(8, default_bits + 2)}
    
    if "language_model.layers." in path:
        try:
            layer_idx = int(path.split("language_model.layers.")[1].split(".")[0])
            num_layers = config.get("text_config", {}).get("num_hidden_layers", 32)
            
            # First and last layers get more bits
            if layer_idx < 2 or layer_idx >= num_layers - 2:
                if any(x in path for x in ["mlp", "down_proj", "up_proj", "gate_proj"]):
                    return {"group_size": group_size, "bits": min(8, default_bits + 2)}
        except (ValueError, IndexError):
            pass
    
    return {"group_size": group_size, "bits": default_bits}