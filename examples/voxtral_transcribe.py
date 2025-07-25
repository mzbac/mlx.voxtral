import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import mlx.core as mx
from mlx_voxtral import VoxtralForConditionalGeneration, VoxtralProcessor


def test_transcribe():
    """Test using the exact same format as transformers."""

    # Model and audio paths
    model_id = "mistralai/Voxtral-Mini-3B-2507"
    audio_url = "https://huggingface.co/datasets/hf-internal-testing/dummy-audio-samples/resolve/main/obama.mp3"

    print("Loading model...")
    model = VoxtralForConditionalGeneration.from_pretrained(model_id, dtype=mx.bfloat16)

    print("Loading processor...")
    processor = VoxtralProcessor.from_pretrained(model_id)

    print("Processing audio with apply_transcrition_request...")

    inputs = processor.apply_transcrition_request(
        language="en", audio=audio_url
    )

    mlx_inputs = {
        "input_ids": inputs.input_ids,
        "input_features": inputs.input_features,
    }

    print(f"\nInput shape: {mlx_inputs['input_ids'].shape}")
    print(f"First 10 tokens: {mlx_inputs['input_ids'][0, :10].tolist()}")
    print(f"Last 10 tokens: {mlx_inputs['input_ids'][0, -10:].tolist()}")

    # Decode to see the prompt
    prompt = processor.decode(inputs.input_ids[0], skip_special_tokens=False)
    print("\nPrompt structure preview:")
    print(f"Start: {prompt[:100]}...")
    print(f"End: ...{prompt[-100:]}")

    print("\nGenerating...")
    import time

    # Time the generation
    start_time = time.time()

    outputs = model.generate(
        **mlx_inputs,
        max_new_tokens=1024,
        temperature=0.0,
        top_p=0.95,
    )

    # Ensure computation is complete (MLX is lazy)
    mx.eval(outputs)

    end_time = time.time()
    generation_time = end_time - start_time

    # Decode only the generated part
    generated_tokens = outputs[0, inputs.input_ids.shape[1] :]
    transcription = processor.decode(generated_tokens, skip_special_tokens=True)

    # Calculate statistics
    num_tokens = len(generated_tokens)
    tokens_per_second = num_tokens / generation_time

    print(f"\n⏱️  Generation completed in {generation_time:.2f} seconds")
    print(f"📊 Generated {num_tokens} tokens")
    print(f"⚡ Tokens per second: {tokens_per_second:.2f}")

    print(f"\nTranscription: {transcription}")

    # Compare with expected
    expected_start = "This week, I traveled to Chicago"
    if transcription.strip().startswith(expected_start):
        print("✅ SUCCESS! Transcription matches expected output")
    else:
        print(
            f"❌ MISMATCH: Expected '{expected_start}...', got '{transcription[:50]}...'"
        )


if __name__ == "__main__":
    test_transcribe()
