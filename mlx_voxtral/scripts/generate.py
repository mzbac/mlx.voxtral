#!/usr/bin/env python3
"""
MLX-Voxtral Generate CLI

A command-line interface for transcribing audio using MLX-Voxtral models.

Usage:
    python -m mlx_voxtral.generate --model mistralai/Voxtral-Mini-3B-2507 --max-token 1024 --temperature 0.0 --audio path/to/audio.wav
"""

import argparse
import time
import mlx.core as mx
from mlx_voxtral import VoxtralProcessor, load_voxtral_model


def main():
    parser = argparse.ArgumentParser(
        description="Transcribe audio using MLX-Voxtral models",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Transcribe audio with default settings
  python -m mlx_voxtral.generate --audio audio.mp3

  # Use a specific model with custom parameters
  python -m mlx_voxtral.generate --model mistralai/Voxtral-Mini-3B-2507 --max-token 2048 --temperature 0.1 --audio audio.mp3

  # Transcribe from URL
  python -m mlx_voxtral.generate --audio https://example.com/audio.mp3

  # Use quantized model
  python -m mlx_voxtral.generate --model ./quantized_models/voxtral-mini-4bit --audio audio.mp3
        """
    )
    
    parser.add_argument(
        "--model",
        type=str,
        default="mistralai/Voxtral-Mini-3B-2507",
        help="Model name or path (default: mistralai/Voxtral-Mini-3B-2507)"
    )
    
    parser.add_argument(
        "--max-token",
        type=int,
        default=1024,
        help="Maximum number of tokens to generate (default: 1024)"
    )
    
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="Sampling temperature, 0.0 for deterministic output (default: 0.0)"
    )
    
    parser.add_argument(
        "--audio",
        type=str,
        required=True,
        help="Path to audio file or URL"
    )
    
    parser.add_argument(
        "--top-p",
        type=float,
        default=0.95,
        help="Nucleus sampling parameter (default: 0.95)"
    )
    
    parser.add_argument(
        "--dtype",
        type=str,
        default="bfloat16",
        choices=["float32", "float16", "bfloat16"],
        help="Model dtype (default: bfloat16)"
    )
    
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Show detailed output including performance metrics"
    )
    
    parser.add_argument(
        "--language",
        type=str,
        default="en",
        help="Language code for transcription (default: en)"
    )
    
    parser.add_argument(
        "--stream",
        action="store_true",
        help="Enable streaming output (generates text token by token)"
    )
    
    args = parser.parse_args()
    
    dtype_map = {
        "float32": mx.float32,
        "float16": mx.float16,
        "bfloat16": mx.bfloat16
    }
    dtype = dtype_map[args.dtype]
    
    if args.verbose:
        print(f"Loading model: {args.model}")
        start_time = time.time()
    
    model, config = load_voxtral_model(args.model, dtype=dtype)
    
    processor = VoxtralProcessor.from_pretrained(args.model)
    
    if args.verbose:
        load_time = time.time() - start_time
        print(f"Model loaded in {load_time:.2f} seconds")
        print(f"Model dtype: {args.dtype}")
    
    if args.verbose:
        print(f"\nProcessing audio: {args.audio}")
    
    inputs = processor.apply_transcrition_request(audio=args.audio, language=args.language)
    
    mlx_inputs = {
        "input_ids": inputs.input_ids,
        "input_features": inputs.input_features,
    }
    
    if args.verbose:
        print("\nGenerating transcription...")
        if args.stream:
            print("(Streaming mode enabled)")
        start_time = time.time()
    
    if args.stream:
        if args.verbose:
            print("\n" + "="*50)
            print("TRANSCRIPTION:")
            print("="*50)
        
        generated_tokens = []
        num_tokens = 0
        
        for token, _ in model.generate_stream(
            **mlx_inputs,
            max_new_tokens=args.max_token,
            temperature=args.temperature,
            top_p=args.top_p,
        ):
            token_id = token.item()
            generated_tokens.append(token_id)
            
            text = processor.decode([token_id], skip_special_tokens=False)
            if token_id not in [processor.tokenizer.eos_token_id, processor.tokenizer.pad_token_id]:
                print(text, end='', flush=True)
            
            num_tokens += 1
        
        print() 
        
        if args.verbose:
            generation_time = time.time() - start_time
            tokens_per_second = num_tokens / generation_time
            print("="*50)
            print(f"\nGenerated {num_tokens} tokens in {generation_time:.2f} seconds ({tokens_per_second:.2f} tokens/s)")
    else:
        output_ids = model.generate(
            **mlx_inputs,
            max_new_tokens=args.max_token,
            temperature=args.temperature,
            top_p=args.top_p,
        )
        
        if args.verbose:
            generation_time = time.time() - start_time
            num_tokens = output_ids.shape[1] - inputs.input_ids.shape[1]
            tokens_per_second = num_tokens / generation_time
            print(f"\nGenerated {num_tokens} tokens in {generation_time:.2f} seconds ({tokens_per_second:.2f} tokens/s)")
        
        generated_tokens = output_ids[0, inputs.input_ids.shape[1]:]
        transcription = processor.decode(generated_tokens, skip_special_tokens=True)
        
        if args.verbose:
            print("\n" + "="*50)
            print("TRANSCRIPTION:")
            print("="*50)
        
        print(transcription)
        
        if args.verbose:
            print("="*50)


if __name__ == "__main__":
    main()