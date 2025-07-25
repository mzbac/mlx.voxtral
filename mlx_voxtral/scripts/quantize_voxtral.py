#!/usr/bin/env python3
"""
Quantize Voxtral models using mlx_lm utilities.

This script leverages mlx_lm's quantization infrastructure directly,
ensuring compatibility and reusing well-tested code.

Usage:
    python scripts/quantize_voxtral_v2.py mistralai/Voxtral-Mini-3B-2507 \
        --output-dir ./voxtral-mini-4bit \
        --bits 4
"""

from pathlib import Path

import argparse
import logging
import shutil

import mlx.core as mx
from mlx_voxtral import load_voxtral_model

from mlx_voxtral.quantization import (
    quantize_model,
    save_model,
    save_config,
    compute_bits_per_weight,
    voxtral_mixed_quantization_predicate,
)
from mlx_lm.utils import get_model_path


def main():
    parser = argparse.ArgumentParser(
        description="Quantize Voxtral models using mlx_lm infrastructure"
    )
    parser.add_argument(
        "model",
        type=str,
        help="Path to model directory or HuggingFace model ID"
    )
    parser.add_argument(
        "--output-dir", "-o",
        type=str,
        required=True,
        help="Output directory for quantized model"
    )
    parser.add_argument(
        "--bits", "-b",
        type=int,
        default=4,
        choices=[2, 4, 8],
        help="Number of bits for quantization (default: 4)"
    )
    parser.add_argument(
        "--group-size", "-g",
        type=int,
        default=64,
        help="Group size for quantization (default: 64)"
    )
    parser.add_argument(
        "--mixed",
        action="store_true",
        help="Use mixed precision quantization (different bits for different layers)"
    )
    parser.add_argument(
        "--dtype",
        type=str,
        choices=["float16", "bfloat16", "float32"],
        default="float16",
        help="Data type to use before quantization (default: float16)"
    )
    
    args = parser.parse_args()
    
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    
    # Parse dtype
    dtype_map = {
        "float16": mx.float16,
        "bfloat16": mx.bfloat16,
        "float32": mx.float32
    }
    dtype = dtype_map[args.dtype]
    
    # Create output directory
    output_path = Path(args.output_dir)
    if output_path.exists():
        raise ValueError(f"Output directory {output_path} already exists")
    output_path.mkdir(parents=True)
    
    # Load model and config
    logger.info(f"Loading model from {args.model}")
    model, config = load_voxtral_model(args.model, dtype=dtype, lazy=True)
    
    # Check if model is already quantized
    if "quantization" in config:
        raise ValueError(
            f"Model at {args.model} is already quantized. "
            "Cannot quantize an already quantized model. "
            "Please use the original unquantized model instead."
        )
    
    logger.info("Model loaded with lazy evaluation")
    
    # Apply quantization using mlx_lm's quantize_model
    logger.info(f"Quantizing model with {args.bits} bits and group size {args.group_size}")
    
    if args.mixed:
        logger.info("Using mixed precision quantization")
        # Pass the default bits to the predicate
        quant_predicate = lambda p, m, c: voxtral_mixed_quantization_predicate(p, m, c, default_bits=args.bits)
    else:
        def uniform_with_exclusions(path, module, config):
            if "embed_positions" in path or "pos_emb" in path:
                return False
            if not hasattr(module, "to_quantized"):
                return False
            return True
        
        logger.info("Using uniform quantization with exclusions")
        quant_predicate = uniform_with_exclusions
    
    # This is the key - we use mlx_lm's quantize_model directly!
    quantized_model, quantized_config = quantize_model(
        model,
        config,
        q_group_size=args.group_size,
        q_bits=args.bits,
        quant_predicate=quant_predicate
    )
    
    # Save using mlx_lm utilities
    logger.info(f"Saving quantized model to {output_path}")
    
    # Save model weights
    save_model(output_path, quantized_model, donate_model=True)
    
    # Save config
    save_config(quantized_config, output_path / "config.json")
    
    # Get source model path
    if Path(args.model).exists():
        model_path = Path(args.model)
    else:
        model_path, _ = get_model_path(args.model)
    
    # Copy all necessary files for the processor and tokenizer
    logger.info("Copying tokenizer and processor files...")
    
    # List of essential files to copy
    essential_files = [
        "generation_config.json",
        "preprocessor_config.json", 
        "tekken.json",  # Mistral tokenizer
        "params.json",
        "tokenizer.json",
        "tokenizer_config.json",
        "special_tokens_map.json",
        "tokenizer.model",  # For other tokenizers
        "*.tiktoken",  # For tiktoken-based tokenizers
    ]
    
    # Copy all essential files
    for pattern in essential_files:
        for file in model_path.glob(pattern):
            if file.is_file():
                shutil.copy2(file, output_path / file.name)
                logger.info(f"Copied {file.name}")
    
    # Also copy any Python files (model code)
    for file in model_path.glob("*.py"):
        if file.is_file():
            shutil.copy2(file, output_path / file.name)
            logger.info(f"Copied {file.name}")
    
    # Report quantization results
    try:
        bits_per_weight = compute_bits_per_weight(quantized_model)
        logger.info(f"Quantized model to average {bits_per_weight:.3f} bits per weight")
    except Exception as e:
        logger.info(f"Could not compute bits per weight: {e}")
        if args.mixed:
            logger.info("Model quantized successfully with mixed precision")
        else:
            logger.info("Model quantized successfully")
    
    # Log quantization decisions for key layers
    if "quantization" in quantized_config:
        logger.info("\nQuantization decisions for key layers:")
        quant_info = quantized_config["quantization"]
        
        # Show some example layers
        example_layers = [
            "encoder.embed_positions",
            "encoder.layers.0.self_attn.q_proj", 
            "multi_modal_projector.linear_1",
            "language_model.layers.0.self_attn.q_proj",
            "language_model.layers.0.mlp.gate_proj",
            "lm_head"
        ]
        
        for layer in example_layers:
            if layer in quant_info:
                decision = quant_info[layer]
                if isinstance(decision, dict):
                    logger.info(f"  {layer}: {decision['bits']} bits, group_size={decision['group_size']}")
                elif decision:
                    logger.info(f"  {layer}: {args.bits} bits (default)")
                else:
                    logger.info(f"  {layer}: not quantized")
    
    logger.info(f"\n✅ Quantization complete! Model saved to: {output_path}")


if __name__ == "__main__":
    main()