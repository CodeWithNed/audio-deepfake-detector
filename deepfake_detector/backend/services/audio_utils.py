"""Audio processing utilities for loading, resampling, and chunking audio files."""

import librosa
import numpy as np
import soundfile as sf
import torch
from pathlib import Path


def load_audio(file_path: str | Path, target_sr: int = 16000) -> tuple[np.ndarray, int]:
    """
    Load audio file and resample to target sample rate.

    Args:
        file_path: Path to audio file
        target_sr: Target sample rate (default: 16kHz)

    Returns:
        Tuple of (audio_array, sample_rate)
    """
    # Load audio using librosa (automatically resamples)
    audio, sr = librosa.load(file_path, sr=target_sr, mono=True)

    return audio, sr


def normalize_audio(audio: np.ndarray, target_db: float = -20.0) -> np.ndarray:
    """
    Normalize audio to target dB level.

    Args:
        audio: Audio array
        target_db: Target RMS dB level

    Returns:
        Normalized audio array
    """
    # Calculate RMS
    rms = np.sqrt(np.mean(audio ** 2))

    # Avoid division by zero
    if rms < 1e-8:
        return audio

    # Convert target dB to amplitude
    target_amp = 10 ** (target_db / 20)

    # Normalize
    normalized = audio * (target_amp / rms)

    # Clip to prevent overflow
    normalized = np.clip(normalized, -1.0, 1.0)

    return normalized


def chunk_audio(audio: np.ndarray, chunk_size_seconds: int, sample_rate: int, overlap: float = 0.5) -> list[np.ndarray]:
    """
    Split long audio into overlapping chunks.

    Args:
        audio: Audio array
        chunk_size_seconds: Length of each chunk in seconds
        sample_rate: Sample rate of audio
        overlap: Overlap ratio between chunks (0-1)

    Returns:
        List of audio chunks
    """
    chunk_size = chunk_size_seconds * sample_rate
    hop_size = int(chunk_size * (1 - overlap))

    chunks = []
    start = 0

    while start < len(audio):
        end = min(start + chunk_size, len(audio))
        chunk = audio[start:end]

        # Pad last chunk if needed
        if len(chunk) < chunk_size:
            chunk = np.pad(chunk, (0, chunk_size - len(chunk)), mode='constant')

        chunks.append(chunk)
        start += hop_size

        # Break if we've reached the end
        if end >= len(audio):
            break

    return chunks


def audio_to_tensor(audio: np.ndarray, device: str = "cpu") -> torch.Tensor:
    """
    Convert numpy audio array to PyTorch tensor.

    Args:
        audio: Audio array
        device: Target device

    Returns:
        Audio tensor
    """
    tensor = torch.from_numpy(audio).float()
    tensor = tensor.to(device)
    return tensor


def trim_silence(audio: np.ndarray, sample_rate: int, threshold_db: float = -40.0) -> np.ndarray:
    """
    Remove leading and trailing silence from audio.

    Args:
        audio: Audio array
        sample_rate: Sample rate
        threshold_db: Silence threshold in dB

    Returns:
        Trimmed audio
    """
    # Convert threshold to amplitude
    threshold = librosa.db_to_amplitude(threshold_db)

    # Trim silence
    trimmed, _ = librosa.effects.trim(audio, top_db=-threshold_db)

    return trimmed


def save_audio(audio: np.ndarray, file_path: str | Path, sample_rate: int = 16000):
    """
    Save audio array to file.

    Args:
        audio: Audio array
        file_path: Output file path
        sample_rate: Sample rate
    """
    sf.write(file_path, audio, sample_rate)


def get_audio_duration(file_path: str | Path) -> float:
    """
    Get duration of audio file in seconds.

    Args:
        file_path: Path to audio file

    Returns:
        Duration in seconds
    """
    duration = librosa.get_duration(path=file_path)
    return duration


def validate_audio_file(file_path: str | Path, max_duration: int = 30) -> dict:
    """
    Validate audio file format and duration.

    Args:
        file_path: Path to audio file
        max_duration: Maximum allowed duration in seconds

    Returns:
        Dictionary with validation results and metadata
    """
    result = {
        'valid': True,
        'errors': [],
        'duration': None,
        'sample_rate': None,
        'channels': None
    }

    try:
        # Get audio info without loading entire file
        info = sf.info(file_path)

        result['duration'] = info.duration
        result['sample_rate'] = info.samplerate
        result['channels'] = info.channels

        # Check duration
        if info.duration > max_duration:
            result['valid'] = False
            result['errors'].append(f"Audio duration ({info.duration:.1f}s) exceeds maximum ({max_duration}s)")

        # Check if mono or stereo (we'll convert to mono anyway)
        if info.channels > 2:
            result['valid'] = False
            result['errors'].append(f"Too many channels ({info.channels}). Maximum is 2.")

    except Exception as e:
        result['valid'] = False
        result['errors'].append(f"Failed to read audio file: {str(e)}")

    return result


if __name__ == "__main__":
    # Test with a dummy audio file
    print("Audio utils module loaded successfully")
    print("Functions available:")
    print("  - load_audio()")
    print("  - normalize_audio()")
    print("  - chunk_audio()")
    print("  - audio_to_tensor()")
    print("  - trim_silence()")
    print("  - save_audio()")
    print("  - get_audio_duration()")
    print("  - validate_audio_file()")
