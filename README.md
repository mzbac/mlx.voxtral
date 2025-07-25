# MLX Voxtral

MLX Voxtral is an optimized implementation of Mistral AI's Voxtral speech models for Apple Silicon, providing efficient audio transcription with support for model quantization and streaming processing.

## Features

- 🚀 **Optimized for Apple Silicon** - Leverages MLX framework for maximum performance on M1/M2/M3 chips
- 🗜️ **Model Quantization** - Reduce model size by 4.3x with minimal quality loss
- 🎙️ **Full Audio Pipeline** - Complete audio processing from file/URL to transcription
- 🔧 **CLI Tools** - Command-line utilities for transcription and quantization
- 📦 **Pre-quantized Models** - Ready-to-use quantized models available

## Installation

### Install from PyPI

```bash
# Install mlx-voxtral from PyPI
pip install mlx-voxtral

# Install transformers from GitHub (required)
pip install git+https://github.com/huggingface/transformers
```

### Install from Source

```bash
# Clone the repository
git clone https://github.com/mzbac/mlx.voxtral
cd mlx.voxtral

# Install in development mode
pip install -e .
```

## Quick Start

### Simple Transcription

```python
from mlx_voxtral import VoxtralForConditionalGeneration, VoxtralProcessor

# Load model and processor
model = VoxtralForConditionalGeneration.from_pretrained("mistralai/Voxtral-Mini-3B-2507")
processor = VoxtralProcessor.from_pretrained("mistralai/Voxtral-Mini-3B-2507")

# Transcribe audio
inputs = processor.apply_transcrition_request(
    language="en",
    audio="speech.mp3"
)
outputs = model.generate(**inputs, max_new_tokens=1024, temperature=0.0)
transcription = processor.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
print(transcription)
```

### Command Line Usage

```bash
# Basic transcription
mlx-voxtral.generate --audio speech.mp3

# With custom parameters
mlx-voxtral.generate --model mistralai/Voxtral-Mini-3B-2507 --max-token 2048 --temperature 0.1 --audio speech.mp3

# From URL
mlx-voxtral.generate --audio https://example.com/podcast.mp3

# Using quantized model
mlx-voxtral.generate --model ./voxtral-mini-4bit --audio speech.mp3
```

## Model Quantization

MLX Voxtral includes powerful quantization capabilities to reduce model size and improve performance:

### Quantization Tool

```bash
# Basic 4-bit quantization (recommended)
mlx-voxtral.quantize mistralai/Voxtral-Mini-3B-2507 -o ./voxtral-mini-4bit

# Mixed precision quantization (best quality)
mlx-voxtral.quantize mistralai/Voxtral-Mini-3B-2507 --output-dir ./voxtral-mini-mixed --mixed

# Custom quantization settings
mlx-voxtral.quantize mistralai/Voxtral-Mini-3B-2507 \
    --output-dir ./voxtral-mini-8bit \
    --bits 8 \
    --group-size 32
```

### Using Quantized Models

```python
# Load pre-quantized model (same API as original)
model = VoxtralForConditionalGeneration.from_pretrained("mzbac/voxtral-mini-3b-4bit-mixed")
processor = VoxtralProcessor.from_pretrained(".mzbac/voxtral-mini-3b-4bit-mixed")

# Use exactly like the original model
transcription = model.transcribe("speech.mp3", processor)
```

## Audio Processing Pipeline

### Low-Level Audio Processing

```python
from mlx_voxtral import process_audio_for_voxtral

# Process audio file for direct model input
result = process_audio_for_voxtral("speech.mp3")

# Access processed features
mel_features = result["input_features"]  # Shape: [n_chunks, 128, 3000]
print(f"Audio duration: {result['duration_seconds']:.2f}s")
print(f"Number of 30s chunks: {result['n_chunks']}")
```

The audio processing pipeline:
1. **Audio Loading**: Supports files and URLs, resamples to 16kHz mono
2. **Chunking**: Splits into 30-second chunks with proper padding
3. **STFT**: 400-point FFT with 160 hop length
4. **Mel Spectrogram**: 128 mel bins covering 0-8000 Hz
5. **Normalization**: Log scale with global max normalization

## Advanced Usage

### Streaming Transcription

```python
# Process long audio files efficiently
for chunk in model.transcribe_stream("podcast.mp3", processor, chunk_length_s=30):
    print(chunk, end="", flush=True)
```

### Custom Generation Parameters

```python
inputs = processor.apply_transcrition_request(
    language="en",
    audio="speech.mp3"
)

outputs = model.generate(
    **inputs,
    max_new_tokens=2048,
    temperature=0.1,
    top_p=0.95,
    repetition_penalty=1.1
)
```

### Processing Multiple Files

```python
# Process multiple audio files sequentially
audio_files = ["audio1.mp3", "audio2.mp3", "audio3.mp3"]
transcriptions = []

for audio_file in audio_files:
    inputs = processor.apply_transcrition_request(language="en", audio=audio_file)
    outputs = model.generate(**inputs, max_new_tokens=1024)
    text = processor.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
    transcriptions.append(text)
```

Note: The model processes one audio file at a time. For long audio files, it automatically splits them into 30-second chunks internally.

## Pre-quantized Models

For convenience, pre-quantized models are available:

```python
models = {
    "mzbac/voxtral-mini-3b-4bit-mixed": "3.2GB model with mixed precision",
    "mzbac/voxtral-mini-3b-8bit": "5.3GB model with 8-bit quantization"
}
```

## API Reference

### VoxtralProcessor

```python
processor = VoxtralProcessor.from_pretrained("mistralai/Voxtral-Mini-3B-2507")

# Apply transcription formatting
inputs = processor.apply_transcrition_request(
    language="en",  # or "fr", "de", etc.
    audio="path/to/audio.mp3",
    task="transcribe",  # or "translate"
)

# Decode model outputs
text = processor.decode(token_ids, skip_special_tokens=True)
```

### VoxtralForConditionalGeneration

```python
model = VoxtralForConditionalGeneration.from_pretrained(
    "mistralai/Voxtral-Mini-3B-2507",
    dtype=mx.bfloat16  # Optional: specify dtype
)

# Generate transcription
outputs = model.generate(
    **inputs,
    max_new_tokens=1024,
    temperature=0.0,
    do_sample=False
)
```

## Performance Tips

1. **Use Quantized Models**: 4-bit quantization provides the best balance of size and quality
2. **Temperature Settings**: Use `temperature=0.0` for deterministic transcription
3. **Chunk Size**: Default 30-second chunks are optimal for most use cases
4. **Long Audio**: The model automatically handles long audio by splitting into chunks

## Requirements

- **Python**: 3.11 or higher
- **Platform**: Apple Silicon Mac (M1/M2/M3)
- **Dependencies**:
  - MLX >= 0.26.5
  - mlx-lm >= 0.26.0
  - mistral-common >= 1.8.2
  - transformers (latest from GitHub)
  - Audio: soundfile, soxr, or ffmpeg

## TODO

- [ ] **Batch Processing Support**: Implement batched inference for processing multiple audio files simultaneously
- [ ] **Transformers Tokenizer Integration**: Add support for using Hugging Face Transformers tokenizers as an alternative to mistral-common
- [ ] **Swift Support**: Create a Swift library for Voxtral support

## License

see LICENSE file for details.

## Acknowledgments

- This implementation is based on Mistral AI's Voxtral models and the Hugging Face Transformers implementation
- Built using Apple's MLX framework for optimized performance on Apple Silicon
