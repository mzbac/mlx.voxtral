#!/usr/bin/env python3
"""
Demonstration of the apply_chat_template functionality in MLX-Voxtral.

This example shows how to use the new chat template feature for:
1. Simple transcription
2. Conversational transcription with context
3. Multi-turn conversations with mixed audio and text
"""

import mlx.core as mx
from mlx_voxtral import VoxtralProcessor, load_voxtral_model

# Load model and processor
model_id = "mzbac/voxtral-mini-3b-4bit-mixed"
print(f"Loading model: {model_id}")
model, config = load_voxtral_model(model_id, dtype=mx.bfloat16)

# Create the processor
processor = VoxtralProcessor.from_pretrained(model_id)

print("\n" + "=" * 80)
print("MLX-Voxtral Chat Template Demo")
print("=" * 80)

# Example 1: Simple audio transcription using chat template
print("\n1. Simple Audio Transcription")
print("-" * 40)

conversation = [
    {
        "role": "user",
        "content": [
            {
                "type": "audio",
                "url": "https://huggingface.co/datasets/hf-internal-testing/dummy-audio-samples/resolve/main/mary_had_lamb.mp3"
            }
        ]
    }
]

# Apply chat template
inputs = processor.apply_chat_template(conversation, return_tensors="mlx")
print(f"Input shape: {inputs['input_ids'].shape}")
print(f"Audio features shape: {inputs['input_features'].shape}")

# Generate transcription
print("\nGenerating transcription...")
outputs = model.generate(
    inputs["input_ids"],
    input_features=inputs["input_features"],
    max_new_tokens=50,
    do_sample=False,
)

# Decode the response
response = processor.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
print(f"Transcription: {response}")

# Example 2: Conversational transcription with context
print("\n\n2. Conversational Transcription")
print("-" * 40)

conversation = [
    {
        "role": "user",
        "content": [
            {"type": "text", "text": "Please transcribe the following nursery rhyme: "},
            {
                "type": "audio",
                "url": "https://huggingface.co/datasets/hf-internal-testing/dummy-audio-samples/resolve/main/mary_had_lamb.mp3"
            }
        ]
    }
]

inputs = processor.apply_chat_template(conversation, return_tensors="mlx")
print(f"Input shape: {inputs['input_ids'].shape}")

print("\nGenerating response...")
outputs = model.generate(
    inputs["input_ids"],
    input_features=inputs["input_features"],
    max_new_tokens=100,
    do_sample=False,
)

response = processor.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
print(f"Response: {response}")

# Example 3: Multi-turn conversation
print("\n\n3. Multi-turn Conversation")
print("-" * 40)

# First, transcribe an audio
conversation = [
    {
        "role": "user",
        "content": [
            {
                "type": "audio",
                "url": "https://huggingface.co/datasets/hf-internal-testing/dummy-audio-samples/resolve/main/bcn_weather.mp3"
            }
        ]
    }
]

inputs = processor.apply_chat_template(conversation, return_tensors="mlx")
outputs = model.generate(
    inputs["input_ids"],
    input_features=inputs["input_features"],
    max_new_tokens=100,
    do_sample=False,
)

transcription = processor.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
print(f"Initial transcription: {transcription}")

# Now continue the conversation
conversation.append({
    "role": "assistant",
    "content": transcription
})
conversation.append({
    "role": "user",
    "content": "What city was mentioned in the audio?"
})

# Apply template with generation prompt
inputs = processor.apply_chat_template(
    conversation, 
    add_generation_prompt=True,
    return_tensors="mlx"
)

print(f"\nContinuing conversation...")
outputs = model.generate(
    inputs["input_ids"],
    max_new_tokens=50,
    do_sample=False,
)

response = processor.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
print(f"Assistant: {response}")

# Example 4: System message with audio
print("\n\n4. System Message with Audio")
print("-" * 40)

conversation = [
    {
        "role": "system",
        "content": "You are a helpful transcription assistant. Always provide accurate transcriptions."
    },
    {
        "role": "user",
        "content": [
            {"type": "text", "text": "Transcribe this audio and tell me what language it's in: "},
            {
                "type": "audio",
                "url": "https://huggingface.co/datasets/hf-internal-testing/dummy-audio-samples/resolve/main/obama.mp3"
            }
        ]
    }
]

inputs = processor.apply_chat_template(conversation, return_tensors="mlx")
print(f"Input shape with system message: {inputs['input_ids'].shape}")

print("\nGenerating response...")
outputs = model.generate(
    inputs["input_ids"],
    input_features=inputs["input_features"],
    max_new_tokens=150,
    do_sample=False,
)

response = processor.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
print(f"Response: {response}")

print("\n" + "=" * 80)
print("Demo completed successfully!")
print("=" * 80)