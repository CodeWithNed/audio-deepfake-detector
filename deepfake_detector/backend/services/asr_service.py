"""ASR (Automatic Speech Recognition) service using OpenAI Whisper."""

import whisper
import numpy as np
from pathlib import Path


class ASRService:
    """Whisper-based ASR service for transcribing audio files."""

    def __init__(self, model_name: str = "base", device: str = "cpu"):
        """
        Initialize ASR service with Whisper model.

        Args:
            model_name: Whisper model size (tiny, base, small, medium, large)
            device: Device to run inference on
        """
        self.model_name = model_name
        self.device = device
        self.model = None

    def load_model(self):
        """Load the Whisper model (lazy loading)."""
        if self.model is None:
            print(f"Loading Whisper {self.model_name} model...")
            self.model = whisper.load_model(self.model_name, device=self.device)
            print("Whisper model loaded successfully")

    def transcribe(self, audio_path: str | Path | np.ndarray, language: str = None) -> dict:
        """
        Transcribe audio to text using Whisper.

        Args:
            audio_path: Path to audio file or numpy array of audio
            language: Language code (e.g., 'en', 'es'). If None, auto-detect.

        Returns:
            Dictionary containing:
            - transcript: Full transcription text
            - segments: List of segments with timestamps and confidence
            - language: Detected language
            - confidence: Mean confidence score across all tokens
        """
        self.load_model()

        # Transcribe with word-level timestamps
        result = self.model.transcribe(
            audio_path,
            language=language,
            verbose=False,
            word_timestamps=True
        )

        # Extract transcript
        transcript = result['text'].strip()

        # Calculate mean confidence from segments
        confidences = []
        for segment in result.get('segments', []):
            if 'words' in segment:
                for word in segment['words']:
                    if 'probability' in word:
                        confidences.append(word['probability'])

        mean_confidence = np.mean(confidences) if confidences else 0.0

        return {
            'transcript': transcript,
            'segments': result.get('segments', []),
            'language': result.get('language', 'unknown'),
            'confidence': float(mean_confidence)
        }

    def transcribe_with_timing(self, audio_path: str | Path) -> dict:
        """
        Transcribe with detailed timing information.

        Args:
            audio_path: Path to audio file

        Returns:
            Dictionary with transcript and timing details
        """
        self.load_model()

        result = self.model.transcribe(
            audio_path,
            word_timestamps=True,
            verbose=False
        )

        # Extract word-level information
        words = []
        for segment in result.get('segments', []):
            if 'words' in segment:
                for word_info in segment['words']:
                    words.append({
                        'word': word_info.get('word', '').strip(),
                        'start': word_info.get('start', 0.0),
                        'end': word_info.get('end', 0.0),
                        'probability': word_info.get('probability', 0.0)
                    })

        return {
            'transcript': result['text'].strip(),
            'words': words,
            'language': result.get('language', 'unknown')
        }


# Global ASR service instance (singleton pattern)
_asr_service_instance = None


def get_asr_service(model_name: str = "base", device: str = "cpu") -> ASRService:
    """
    Get or create global ASR service instance.

    Args:
        model_name: Whisper model size
        device: Device to run on

    Returns:
        ASRService instance
    """
    global _asr_service_instance

    if _asr_service_instance is None:
        _asr_service_instance = ASRService(model_name=model_name, device=device)

    return _asr_service_instance


if __name__ == "__main__":
    # Test the ASR service
    print("ASR Service initialized")
    print("Available Whisper models: tiny, base, small, medium, large")
    print("Usage:")
    print("  asr = ASRService(model_name='base')")
    print("  result = asr.transcribe('audio.wav')")
    print("  print(result['transcript'])")
