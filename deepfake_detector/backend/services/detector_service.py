"""Main deepfake detector service orchestrating the full pipeline."""

import torch
import numpy as np
from pathlib import Path

from config import settings
from models.acoustic_model import AcousticDetector
from models.linguistic_model import LinguisticDetector
from models.fusion import ScoreFusion
from services.asr_service import get_asr_service
from services.audio_utils import load_audio, normalize_audio, audio_to_tensor


class DeepfakeDetectorService:
    """
    Main service for deepfake audio detection.

    Pipeline:
    1. Load and preprocess audio
    2. Stage 1: Acoustic detection (WavLM)
    3. Stage 2: ASR transcription (Whisper)
    4. Stage 3: Linguistic detection (RoBERTa)
    5. Fusion: Combine scores and make decision
    """

    def __init__(self):
        """Initialize detector service with models."""
        self.device = self._get_device()
        print(f"Using device: {self.device}")

        # Initialize models (lazy loading)
        self.acoustic_model = None
        self.linguistic_model = None
        self.asr_service = None
        self.fusion = ScoreFusion(
            acoustic_weight=settings.ACOUSTIC_WEIGHT,
            linguistic_weight=settings.LINGUISTIC_WEIGHT
        )

    def _get_device(self) -> str:
        """Determine the best available device."""
        if settings.DEVICE == "cuda" and torch.cuda.is_available():
            return "cuda"
        return "cpu"

    def load_models(self):
        """Load all models (lazy loading)."""
        if self.acoustic_model is None:
            print("Loading acoustic model...")
            self.acoustic_model = AcousticDetector(
                model_name=settings.ACOUSTIC_MODEL_NAME
            )

            # Try to load trained weights
            checkpoint_path = settings.CHECKPOINTS_DIR / settings.ACOUSTIC_CHECKPOINT
            if checkpoint_path.exists():
                checkpoint = torch.load(checkpoint_path, map_location=self.device)
                self.acoustic_model.load_state_dict(checkpoint['model_state_dict'])
                print(f"Loaded acoustic checkpoint from {checkpoint_path}")
            else:
                print("Warning: No acoustic checkpoint found. Using pretrained encoder only.")

            self.acoustic_model.to(self.device)
            self.acoustic_model.eval()

        if self.linguistic_model is None:
            print("Loading linguistic model...")
            self.linguistic_model = LinguisticDetector(
                model_name=settings.LINGUISTIC_MODEL_NAME
            )

            # Try to load trained weights
            checkpoint_path = settings.CHECKPOINTS_DIR / settings.LINGUISTIC_CHECKPOINT
            if checkpoint_path.exists():
                checkpoint = torch.load(checkpoint_path, map_location=self.device)
                self.linguistic_model.load_state_dict(checkpoint['model_state_dict'])
                print(f"Loaded linguistic checkpoint from {checkpoint_path}")
            else:
                print("Warning: No linguistic checkpoint found. Using pretrained encoder only.")

            self.linguistic_model.to(self.device)
            self.linguistic_model.eval()

        if self.asr_service is None:
            print("Loading ASR service...")
            self.asr_service = get_asr_service(
                model_name=settings.WHISPER_MODEL_NAME,
                device=self.device
            )

    async def detect(self, audio_path: str | Path) -> dict:
        """
        Run full deepfake detection pipeline on audio file.

        Args:
            audio_path: Path to audio file

        Returns:
            Dictionary containing detection results
        """
        # Load models if not already loaded
        self.load_models()

        # Stage 1: Acoustic Detection
        acoustic_score = self._run_acoustic_detection(audio_path)

        # Stage 2: ASR Transcription
        transcript_result = self.asr_service.transcribe(audio_path)
        transcript = transcript_result['transcript']
        asr_confidence = transcript_result['confidence']

        # Stage 3: Linguistic Detection
        linguistic_score = self._run_linguistic_detection(transcript)

        # Fusion: Combine scores
        fused_score = self.fusion.fuse(acoustic_score, linguistic_score)
        decision = self.fusion.decide(fused_score, threshold=settings.FAKE_THRESHOLD)
        confidence = self.fusion.get_confidence(fused_score)

        # Build result dictionary
        result = {
            'decision': decision,
            'final_score': float(fused_score),
            'confidence': float(confidence),
            'acoustic_score': float(acoustic_score),
            'linguistic_score': float(linguistic_score),
            'transcript': transcript,
            'asr_confidence': float(asr_confidence),
            'language': transcript_result.get('language', 'unknown')
        }

        return result

    def _run_acoustic_detection(self, audio_path: str | Path) -> float:
        """
        Run acoustic detection on audio file.

        Args:
            audio_path: Path to audio file

        Returns:
            Acoustic fake probability (0-1)
        """
        # Load and preprocess audio
        audio, sr = load_audio(audio_path, target_sr=settings.SAMPLE_RATE)
        audio = normalize_audio(audio)

        # Convert to tensor
        audio_tensor = audio_to_tensor(audio, device=self.device).unsqueeze(0)

        # Run inference
        with torch.no_grad():
            score = self.acoustic_model(audio_tensor)

        return float(score.squeeze().item())

    def _run_linguistic_detection(self, transcript: str) -> float:
        """
        Run linguistic detection on transcript.

        Args:
            transcript: Text transcript from ASR

        Returns:
            Linguistic fake probability (0-1)
        """
        # Handle empty transcript
        if not transcript or len(transcript.strip()) == 0:
            return 0.5  # Neutral score if no transcript

        # Tokenize transcript
        encoded = self.linguistic_model.tokenize([transcript])
        input_ids = encoded['input_ids'].to(self.device)
        attention_mask = encoded['attention_mask'].to(self.device)

        # Run inference
        with torch.no_grad():
            score = self.linguistic_model(input_ids, attention_mask)

        return float(score.squeeze().item())


# Global detector service instance (singleton)
_detector_service_instance = None


def get_detector_service() -> DeepfakeDetectorService:
    """
    Get or create global detector service instance.

    Returns:
        DeepfakeDetectorService instance
    """
    global _detector_service_instance

    if _detector_service_instance is None:
        _detector_service_instance = DeepfakeDetectorService()

    return _detector_service_instance


if __name__ == "__main__":
    # Quick test
    detector = DeepfakeDetectorService()
    print("Deepfake Detector Service initialized")
    print(f"Device: {detector.device}")
    print(f"Acoustic weight: {detector.fusion.acoustic_weight}")
    print(f"Linguistic weight: {detector.fusion.linguistic_weight}")
