"""Acoustic model using WavLM for audio-based deepfake detection."""

import torch
import torch.nn as nn
from transformers import Wav2Vec2FeatureExtractor, WavLMModel


class AcousticDetector(nn.Module):
    """
    WavLM-based acoustic deepfake detector.

    Architecture:
    - WavLM base encoder (pretrained)
    - Linear projection -> ReLU -> Linear -> Sigmoid
    - Output: probability of audio being fake (0-1)
    """

    def __init__(self, model_name: str = "microsoft/wavlm-base-plus", freeze_encoder: bool = False):
        super().__init__()

        # Load pretrained WavLM model
        self.wavlm = WavLMModel.from_pretrained(model_name)
        self.feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(model_name)

        # Optionally freeze the encoder during training
        if freeze_encoder:
            for param in self.wavlm.parameters():
                param.requires_grad = False

        # Get hidden size from WavLM config
        hidden_size = self.wavlm.config.hidden_size  # 768 for base-plus

        # Binary classification head
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )

    def forward(self, audio_input: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the acoustic detector.

        Args:
            audio_input: Raw audio waveform tensor [batch_size, sequence_length]

        Returns:
            Fake probability scores [batch_size, 1]
        """
        # Extract WavLM embeddings
        with torch.set_grad_enabled(self.training):
            outputs = self.wavlm(audio_input)
            # Use mean pooling over the sequence dimension
            embeddings = outputs.last_hidden_state.mean(dim=1)  # [batch_size, hidden_size]

        # Pass through classification head
        logits = self.classifier(embeddings)  # [batch_size, 1]

        return logits

    def extract_embeddings(self, audio_input: torch.Tensor) -> torch.Tensor:
        """
        Extract WavLM embeddings without classification.

        Args:
            audio_input: Raw audio waveform tensor [batch_size, sequence_length]

        Returns:
            Audio embeddings [batch_size, hidden_size]
        """
        with torch.no_grad():
            outputs = self.wavlm(audio_input)
            embeddings = outputs.last_hidden_state.mean(dim=1)
        return embeddings


def load_acoustic_model(checkpoint_path: str, device: str = "cpu") -> AcousticDetector:
    """
    Load a trained acoustic model from checkpoint.

    Args:
        checkpoint_path: Path to the saved model checkpoint
        device: Device to load the model onto

    Returns:
        Loaded AcousticDetector model
    """
    model = AcousticDetector()
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    return model


if __name__ == "__main__":
    # Quick test
    model = AcousticDetector()
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Test forward pass with dummy audio (16kHz, 3 seconds)
    dummy_audio = torch.randn(2, 16000 * 3)
    output = model(dummy_audio)
    print(f"Output shape: {output.shape}")
    print(f"Output values: {output}")
