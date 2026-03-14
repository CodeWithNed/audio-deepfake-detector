"""Linguistic model using RoBERTa for transcript-based deepfake detection."""

import torch
import torch.nn as nn
from transformers import RobertaModel, RobertaTokenizer


class LinguisticDetector(nn.Module):
    """
    RoBERTa-based linguistic deepfake detector.

    Architecture:
    - RoBERTa base encoder (pretrained)
    - Classification head on [CLS] token
    - Linear -> ReLU -> Linear -> Sigmoid
    - Output: probability of transcript being from fake audio (0-1)
    """

    def __init__(self, model_name: str = "roberta-base", freeze_encoder: bool = False):
        super().__init__()

        # Load pretrained RoBERTa model
        self.roberta = RobertaModel.from_pretrained(model_name)
        self.tokenizer = RobertaTokenizer.from_pretrained(model_name)

        # Optionally freeze the encoder during training
        if freeze_encoder:
            for param in self.roberta.parameters():
                param.requires_grad = False

        # Get hidden size from RoBERTa config
        hidden_size = self.roberta.config.hidden_size  # 768 for base

        # Binary classification head
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the linguistic detector.

        Args:
            input_ids: Tokenized input [batch_size, seq_length]
            attention_mask: Attention mask [batch_size, seq_length]

        Returns:
            Fake probability scores [batch_size, 1]
        """
        # Extract RoBERTa embeddings
        with torch.set_grad_enabled(self.training):
            outputs = self.roberta(input_ids=input_ids, attention_mask=attention_mask)
            # Use [CLS] token embedding (first token)
            cls_embedding = outputs.last_hidden_state[:, 0, :]  # [batch_size, hidden_size]

        # Pass through classification head
        logits = self.classifier(cls_embedding)  # [batch_size, 1]

        return logits

    def tokenize(self, texts: list[str], max_length: int = 512) -> dict:
        """
        Tokenize input texts using RoBERTa tokenizer.

        Args:
            texts: List of input texts
            max_length: Maximum sequence length

        Returns:
            Dictionary with input_ids and attention_mask tensors
        """
        encoded = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt"
        )
        return encoded

    def extract_embeddings(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """
        Extract RoBERTa [CLS] embeddings without classification.

        Args:
            input_ids: Tokenized input [batch_size, seq_length]
            attention_mask: Attention mask [batch_size, seq_length]

        Returns:
            Text embeddings [batch_size, hidden_size]
        """
        with torch.no_grad():
            outputs = self.roberta(input_ids=input_ids, attention_mask=attention_mask)
            cls_embedding = outputs.last_hidden_state[:, 0, :]
        return cls_embedding


def load_linguistic_model(checkpoint_path: str, device: str = "cpu") -> LinguisticDetector:
    """
    Load a trained linguistic model from checkpoint.

    Args:
        checkpoint_path: Path to the saved model checkpoint
        device: Device to load the model onto

    Returns:
        Loaded LinguisticDetector model
    """
    model = LinguisticDetector()
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    return model


if __name__ == "__main__":
    # Quick test
    model = LinguisticDetector()
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Test forward pass with dummy text
    dummy_texts = [
        "This is a test transcript from an audio file.",
        "Another sample transcript for testing purposes."
    ]

    encoded = model.tokenize(dummy_texts)
    output = model(encoded['input_ids'], encoded['attention_mask'])

    print(f"Output shape: {output.shape}")
    print(f"Output values: {output}")
