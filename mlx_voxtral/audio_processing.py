import mlx.core as mx
import numpy as np
import subprocess
import shutil
import os
from typing import Optional, Dict, Union, Tuple
from pathlib import Path

SAMPLE_RATE = 16000
N_FFT = 400
HOP_LENGTH = 160
CHUNK_LENGTH = 30  # seconds
N_SAMPLES = CHUNK_LENGTH * SAMPLE_RATE  # 480000
N_FRAMES = N_SAMPLES // HOP_LENGTH  # 3000
N_MELS = 128


def load_audio(
    file_path: Union[str, Path],
    sr: int = SAMPLE_RATE,
    mono: bool = True,
    normalize: bool = False,
) -> mx.array:
    """
    Load audio file and convert to MLX array.

    Args:
        file_path: Path to audio file (supports most formats via ffmpeg)
        sr: Target sample rate (default: 16000)
        mono: Convert to mono if True (default: True)
        normalize: Normalize audio to [-1, 1] if True (default: False)

    Returns:
        audio: MLX array of audio samples

    Raises:
        RuntimeError: If audio loading fails
    """
    file_path = Path(file_path)
    if not file_path.exists():
        raise FileNotFoundError(f"Audio file not found: {file_path}")

    try:
        import soundfile as sf

        audio_array, sample_rate = sf.read(
            str(file_path), always_2d=False, dtype="float32"
        )

        if sample_rate != sr:
            try:
                import soxr

                audio_array = soxr.resample(audio_array, sample_rate, sr)
            except ImportError:
                import librosa

                audio_array = librosa.resample(
                    audio_array, orig_sr=sample_rate, target_sr=sr
                )

        if mono and audio_array.ndim > 1:
            audio_array = audio_array.mean(axis=1)

    except ImportError:
        ffmpeg_path = shutil.which("ffmpeg")
        if not ffmpeg_path:
            for path in [
                "/opt/homebrew/bin/ffmpeg",
                "/usr/local/bin/ffmpeg",
                "/usr/bin/ffmpeg",
            ]:
                if os.path.exists(path):
                    ffmpeg_path = path
                    break
            else:
                raise RuntimeError(
                    "ffmpeg not found. Please install ffmpeg or soundfile."
                )

        cmd = [
            ffmpeg_path,
            "-nostdin",
            "-threads",
            "0",
            "-i",
            str(file_path),
            "-f",
            "f32le",
            "-acodec",
            "pcm_f32le",
            "-ar",
            str(sr),
        ]

        if mono:
            cmd.extend(["-ac", "1"])

        cmd.append("-")

        try:
            out = subprocess.run(cmd, capture_output=True, check=True).stdout
            audio_array = np.frombuffer(out, np.float32)
        except subprocess.CalledProcessError as e:
            raise RuntimeError(f"Failed to load audio: {e.stderr.decode()}")

    # Normalize if requested
    if normalize:
        max_val = np.abs(audio_array).max()
        if max_val > 0:
            audio_array = audio_array / max_val

    return mx.array(audio_array)


def pad_to_multiple(array: mx.array, multiple: int) -> mx.array:
    """
    Pad audio to the next multiple of `multiple` samples.

    Args:
        array: Input audio array
        multiple: Pad to multiple of this value

    Returns:
        padded_array: Zero-padded audio
    """
    current_length = array.shape[0]
    remainder = current_length % multiple

    if remainder == 0:
        return array
    else:
        padding = multiple - remainder
        return mx.pad(array, [(0, padding)])


def hanning(size: int) -> mx.array:
    """
    Create Hanning (Hann) window.

    Args:
        size: Window size

    Returns:
        window: Hanning window
    """
    window_np = np.hanning(size + 1)[:-1].astype(np.float32)
    return mx.array(window_np)


def stft_mlx(
    x: mx.array,
    window: mx.array,
    nperseg: int = 256,
    noverlap: Optional[int] = None,
    nfft: Optional[int] = None,
    center: bool = True,
) -> mx.array:
    """
    Compute Short-Time Fourier Transform using MLX.

    Args:
        x: Input signal
        window: Window function
        nperseg: Length of each segment
        noverlap: Number of points to overlap (default: nperseg // 4)
        nfft: Length of the FFT (default: nperseg)
        center: If True, apply reflection padding (default: True)

    Returns:
        stft: Complex STFT matrix of shape [n_frames, nfft // 2 + 1]
    """
    if nfft is None:
        nfft = nperseg
    if noverlap is None:
        noverlap = nfft // 4

    hop_length = nperseg - noverlap

    if center:
        padding = nperseg // 2
        prefix = x[padding:0:-1]
        suffix = x[-2 : -padding - 2 : -1]
        x = mx.concatenate([prefix, x, suffix])

    n_frames = (x.size - nperseg) // hop_length + 1

    shape = [n_frames, nfft]
    strides = [hop_length, 1]
    x_strided = mx.as_strided(x, shape=shape, strides=strides)

    x_windowed = x_strided[:, :nperseg] * window

    if nfft > nperseg:
        pad_width = [(0, 0), (0, nfft - nperseg)]
        x_windowed = mx.pad(x_windowed, pad_width)

    return mx.fft.rfft(x_windowed)


def mel_filter_bank_slaney(
    sr: int, n_fft: int, n_mels: int, fmin: float = 0.0, fmax: Optional[float] = None
) -> mx.array:
    """
    Create mel filter bank with Slaney-style normalization (matching Whisper).

    Args:
        sr: Sample rate
        n_fft: FFT size
        n_mels: Number of mel bins
        fmin: Minimum frequency (default: 0.0)
        fmax: Maximum frequency (default: sr/2)

    Returns:
        mel_filters: Mel filter bank of shape [n_mels, n_fft // 2 + 1]
    """
    if fmax is None:
        fmax = sr / 2

    def hz_to_mel(freq):
        min_log_hz = 1000.0
        min_log_mel = 15.0
        logstep = 27.0 / np.log(6.4)

        if isinstance(freq, (int, float)):
            if freq >= min_log_hz:
                return min_log_mel + np.log(freq / min_log_hz) * logstep
            else:
                return 3.0 * freq / 200.0
        else:
            freq_np = np.array(freq) if not isinstance(freq, np.ndarray) else freq
            mels = 3.0 * freq_np / 200.0
            log_region = freq_np >= min_log_hz
            mels[log_region] = (
                min_log_mel + np.log(freq_np[log_region] / min_log_hz) * logstep
            )
            return mels

    def mel_to_hz(mels):
        min_log_hz = 1000.0
        min_log_mel = 15.0
        logstep = np.log(6.4) / 27.0

        if isinstance(mels, (int, float)):
            if mels >= min_log_mel:
                return min_log_hz * np.exp(logstep * (mels - min_log_mel))
            else:
                return 200.0 * mels / 3.0
        else:
            mels_np = np.array(mels) if not isinstance(mels, np.ndarray) else mels
            freq = 200.0 * mels_np / 3.0
            log_region = mels_np >= min_log_mel
            freq[log_region] = min_log_hz * np.exp(
                logstep * (mels_np[log_region] - min_log_mel)
            )
            return freq

    mel_min = hz_to_mel(fmin)
    mel_max = hz_to_mel(fmax)
    mel_points = np.linspace(mel_min, mel_max, n_mels + 2)
    hz_points = mel_to_hz(mel_points)

    fft_freqs = np.linspace(0, sr / 2, n_fft // 2 + 1)

    filters = np.zeros((n_mels, n_fft // 2 + 1))

    for i in range(n_mels):
        left = hz_points[i]
        center = hz_points[i + 1]
        right = hz_points[i + 2]

        rising = (fft_freqs - left) / (center - left)
        rising = np.maximum(0, np.minimum(1, rising))

        falling = (right - fft_freqs) / (right - center)
        falling = np.maximum(0, np.minimum(1, falling))

        filters[i] = rising * falling

        enorm = 2.0 / (hz_points[i + 2] - hz_points[i])
        filters[i] *= enorm

    return mx.array(filters, dtype=mx.float32)


_mel_filters_cache = {}


def get_mel_filters(n_mels: int = N_MELS) -> mx.array:
    """Get cached mel filters."""
    if n_mels not in _mel_filters_cache:
        _mel_filters_cache[n_mels] = mel_filter_bank_slaney(
            SAMPLE_RATE, N_FFT, n_mels, fmax=8000
        )
    return _mel_filters_cache[n_mels]


def log_mel_spectrogram(
    audio: mx.array,
    n_mels: int = N_MELS,
    n_fft: int = N_FFT,
    hop_length: int = HOP_LENGTH,
    global_max: Optional[float] = None,
) -> Union[mx.array, Tuple[mx.array, float]]:
    """
    Compute log-mel spectrogram matching Whisper/Voxtral implementation.

    Args:
        audio: Input audio array
        n_mels: Number of mel bins (default: 128)
        n_fft: FFT size (default: 400)
        hop_length: Hop length (default: 160)
        global_max: If provided, use this for normalization (for multi-chunk consistency)

    Returns:
        log_mel_spec: Log-mel spectrogram of shape [n_mels, n_frames]
        If global_max is None, also returns the computed log_max value
    """
    window = hanning(n_fft)

    freqs = stft_mlx(audio, window, nperseg=n_fft, noverlap=n_fft - hop_length)

    freqs = freqs[:-1, :]

    magnitudes = mx.abs(freqs) ** 2

    filters = get_mel_filters(n_mels)
    mel_spec = magnitudes @ filters.T

    log_spec = mx.log10(mx.maximum(mel_spec, 1e-10))
    mx.eval(log_spec)

    if global_max is None:
        log_max = mx.max(log_spec)
        mx.eval(log_max)
        return_log_max = True
    else:
        log_max = global_max
        return_log_max = False

    log_spec = mx.maximum(log_spec, log_max - 8.0)
    log_spec = (log_spec + 4.0) / 4.0

    log_spec = log_spec.T

    if return_log_max:
        return log_spec, log_max
    else:
        return log_spec


def process_audio_for_voxtral(
    audio_path: Union[str, Path],
    chunk_length_s: int = 30,
    normalize_audio: bool = False,
    return_attention_mask: bool = False,
) -> Dict[str, Union[mx.array, np.ndarray, int, float]]:
    """
    Process audio file for Voxtral model input with automatic chunking.

    This function implements the complete audio processing pipeline matching
    the native Voxtral/Whisper implementation, including:
    - Loading and resampling to 16kHz
    - Padding to multiples of 30 seconds
    - Computing log-mel spectrograms with global normalization
    - Automatic chunking for long audio

    Args:
        audio_path: Path to audio file
        chunk_length_s: Maximum chunk length in seconds (default: 30)
        normalize_audio: Normalize input audio to [-1, 1] (default: False)
        return_attention_mask: Return attention mask for model (default: False)

    Returns:
        Dictionary containing:
            - input_features: Mel spectrograms [n_chunks, 128, 3000]
            - n_chunks: Number of 30-second chunks
            - duration_seconds: Original audio duration
            - chunk_length_s: Chunk length used
            - attention_mask: (optional) Attention mask [n_chunks, 3000]

    Example:
        ```python
        # Basic usage
        result = process_audio_for_voxtral("speech.mp3")
        mel_features = result["input_features"]

        # Process with model
        for i in range(result["n_chunks"]):
            chunk = mel_features[i:i+1]  # [1, 128, 3000]
            output = model.generate(input_features=chunk)
        ```
    """
    audio = load_audio(audio_path, sr=SAMPLE_RATE, mono=True, normalize=normalize_audio)
    original_length = audio.shape[0]
    duration_seconds = original_length / SAMPLE_RATE

    chunk_samples = chunk_length_s * SAMPLE_RATE
    audio_padded = pad_to_multiple(audio, chunk_samples)
    padded_length = audio_padded.shape[0]

    n_chunks = padded_length // chunk_samples

    mel_full, global_log_max = log_mel_spectrogram(audio_padded)

    mel_chunks = []

    for i in range(n_chunks):
        start_sample = i * chunk_samples
        end_sample = (i + 1) * chunk_samples
        chunk_audio = audio_padded[start_sample:end_sample]

        chunk_mel = log_mel_spectrogram(chunk_audio, global_max=global_log_max)
        mel_chunks.append(chunk_mel[None, :, :])  

    input_features = mx.concatenate(mel_chunks, axis=0) 

    mx.eval(input_features)

    result = {
        "input_features": input_features,
        "n_chunks": n_chunks,
        "duration_seconds": duration_seconds,
        "chunk_length_s": chunk_length_s,
        "sample_rate": SAMPLE_RATE,
        "original_samples": original_length,
        "padded_samples": padded_length,
    }

    if return_attention_mask:
        attention_mask = mx.ones((n_chunks, N_FRAMES), dtype=mx.int32)
        result["attention_mask"] = attention_mask
        result["attention_mask_numpy"] = np.array(attention_mask)

    return result


class VoxtralFeatureExtractor:
    """Feature extractor for Voxtral audio processing."""

    def __init__(
        self,
        feature_size: int = N_MELS,
        sampling_rate: int = SAMPLE_RATE,
        hop_length: int = HOP_LENGTH,
        chunk_length: int = CHUNK_LENGTH,
        n_fft: int = N_FFT,
        padding_value: float = 0.0,
    ):
        self.feature_size = feature_size
        self.sampling_rate = sampling_rate
        self.hop_length = hop_length
        self.chunk_length = chunk_length
        self.n_fft = n_fft
        self.padding_value = padding_value
        self.nb_max_frames = self.chunk_length * self.sampling_rate // self.hop_length

    def _process_audio_array_with_chunking(self, audio_array: np.ndarray) -> np.ndarray:
        """Process audio array with proper chunking into 30-second segments."""
        chunk_samples = self.chunk_length * self.sampling_rate
        n_samples = len(audio_array)
        n_chunks = int(np.ceil(n_samples / chunk_samples))
        total_samples = n_chunks * chunk_samples

        if n_samples < total_samples:
            audio_array = np.pad(
                audio_array, (0, total_samples - n_samples), mode="constant"
            )

        all_features = []
        for i in range(n_chunks):
            start = i * chunk_samples
            end = start + chunk_samples
            chunk = audio_array[start:end]
            chunk_features = process_audio_chunk(chunk)
            all_features.append(chunk_features)

        return np.stack(all_features, axis=0)

    def __call__(
        self,
        raw_speech: Union[np.ndarray, list, str],
        sampling_rate: Optional[int] = None,
        return_tensors: Optional[str] = "np",
        **kwargs,
    ) -> Dict[str, Union[np.ndarray, mx.array]]:
        """
        Process raw speech into features.

        Args:
            raw_speech: Audio input - can be:
                - URL string (http:// or https://)
                - Local file path string
                - NumPy array
                - List of floats
            sampling_rate: Sampling rate of the audio (for array inputs)
            return_tensors: Type of tensors to return ("np" or "mlx")

        Returns:
            Dictionary with 'input_features' key containing mel spectrogram
            features of shape (batch_size, n_mels, n_frames)
        """
        if isinstance(raw_speech, str):
            if raw_speech.startswith(("http://", "https://")):
                import requests
                import io
                import soundfile as sf

                try:
                    response = requests.get(raw_speech, timeout=30)
                    response.raise_for_status()

                    with io.BytesIO(response.content) as audio_file:
                        audio_array, sr = sf.read(audio_file, dtype="float32")

                    if audio_array.ndim > 1:
                        audio_array = audio_array.mean(axis=1)

                    if sr != self.sampling_rate:
                        try:
                            import soxr

                            audio_array = soxr.resample(
                                audio_array, sr, self.sampling_rate
                            )
                        except ImportError:
                            raise ImportError(
                                "soxr is required for resampling. Install with: pip install soxr"
                            )

                    mel_features = self._process_audio_array_with_chunking(audio_array)

                except Exception as e:
                    raise ValueError(f"Error loading audio from URL: {e}")
            else:
                result = process_audio_for_voxtral(raw_speech)
                mel_features = np.array(result["input_features"])
        elif isinstance(raw_speech, list):
            audio_array = np.array(raw_speech, dtype=np.float32)
            mel_features = self._process_audio_array_with_chunking(audio_array)
        else:
            audio_array = raw_speech

            # Convert stereo to mono if needed
            if audio_array.ndim > 1:
                audio_array = audio_array.mean(axis=1)
                
            if sampling_rate is not None and sampling_rate != self.sampling_rate:
                try:
                    import soxr

                    audio_array = soxr.resample(
                        audio_array, sampling_rate, self.sampling_rate
                    )
                except ImportError:
                    raise ImportError(
                        "soxr is required for resampling. Install with: pip install soxr"
                    )

            mel_features = self._process_audio_array_with_chunking(audio_array)

        if return_tensors == "mlx":
            mel_features = mx.array(mel_features)

        return {"input_features": mel_features}


def process_audio_chunk(
    audio_array: Union[np.ndarray, mx.array], sample_rate: int = SAMPLE_RATE
) -> mx.array:
    """
    Process a single audio chunk (≤30s) to mel spectrogram.

    Args:
        audio_array: Audio samples
        sample_rate: Sample rate of input audio

    Returns:
        mel_features: Log-mel spectrogram [128, n_frames]
    """
    if isinstance(audio_array, np.ndarray):
        audio_array = mx.array(audio_array)

    if sample_rate != SAMPLE_RATE:
        raise NotImplementedError(
            "Resampling not implemented. Please provide 16kHz audio."
        )

    if audio_array.shape[0] > N_SAMPLES:
        audio_array = audio_array[:N_SAMPLES]
    elif audio_array.shape[0] < N_SAMPLES:
        padding = N_SAMPLES - audio_array.shape[0]
        audio_array = mx.pad(audio_array, [(0, padding)])

    mel_features, _ = log_mel_spectrogram(audio_array)

    return mel_features
