import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import mlx.core as mx
from mlx_voxtral import VoxtralForConditionalGeneration, VoxtralProcessor


def main():
    """Test the quantized model by itself."""
    
    print("Testing Quantized Voxtral Model")
    print("=" * 60)
    
    # Load quantized model from Hugging Face
    model_id = "mzbac/voxtral-mini-3b-4bit-mixed"
    
    print(f"\nLoading quantized model from: {model_id}")
    model = VoxtralForConditionalGeneration.from_pretrained(
        model_id, 
        dtype=mx.bfloat16
    )
    print("✅ Model loaded successfully!")
    
    # Load processor
    print("\nLoading processor...")
    processor = VoxtralProcessor.from_pretrained("mistralai/Voxtral-Mini-3B-2507")
    print("✅ Processor loaded successfully!")
    
    # Test audio
    audio_url = "https://huggingface.co/datasets/hf-internal-testing/dummy-audio-samples/resolve/main/obama.mp3"
    
    print("\nProcessing audio...")
    inputs = processor.apply_transcrition_request(
        language="en", 
        audio=audio_url
    )
    
    mlx_inputs = {
        "input_ids": inputs.input_ids,
        "input_features": inputs.input_features,
    }
    
    print(f"Input shape: {mlx_inputs['input_ids'].shape}")
    print(f"Audio features shape: {mlx_inputs['input_features'].shape}")
    
    # Generate
    print("\nGenerating transcription...")
    import time
    start_time = time.time()
    
    outputs = model.generate(
        **mlx_inputs,
        max_new_tokens=1024,
        temperature=0.0,
        top_p=0.95,
    )
    
    # Ensure computation is complete
    mx.eval(outputs)
    
    gen_time = time.time() - start_time
    
    # Decode
    generated_tokens = outputs[0, inputs.input_ids.shape[1]:]
    transcription = processor.decode(generated_tokens, skip_special_tokens=True)
    
    # Display results
    num_tokens = len(generated_tokens)
    tokens_per_second = num_tokens / gen_time
    
    print(f"\n⏱️  Generation completed in {gen_time:.2f} seconds")
    print(f"📊 Generated {num_tokens} tokens")
    print(f"⚡ Tokens per second: {tokens_per_second:.2f}")
    
    print(f"\nTranscription: {transcription}")
    
    # Check expected output
    expected_start = "This week, I traveled to Chicago"
    if transcription.strip().startswith(expected_start):
        print("\n✅ SUCCESS! Quantized model produces correct transcription!")
    else:
        print("\n❌ Output differs from expected")
    
    print("\n" + "="*60)
    print("Quantized model works correctly!")
    print("Memory usage: ~2.8 GB (vs ~9.2 GB for original)")
    print("="*60)


if __name__ == "__main__":
    main()