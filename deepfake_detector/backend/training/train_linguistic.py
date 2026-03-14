"""Training script for linguistic model (RoBERTa)."""

import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, random_split
from tqdm import tqdm
import argparse

from config import settings
from models.linguistic_model import LinguisticDetector
from training.dataset import TranscriptDeepfakeDataset


class TranscriptDataLoader(Dataset):
    """Wrapper for transcript dataset with tokenization."""

    def __init__(self, transcript_dataset, tokenizer):
        self.dataset = transcript_dataset
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        transcript, label = self.dataset[idx]
        return transcript, torch.tensor(label, dtype=torch.float32)


def collate_fn(batch, tokenizer):
    """Custom collate function for batching transcripts."""
    transcripts, labels = zip(*batch)

    # Tokenize batch
    encoded = tokenizer(
        list(transcripts),
        padding=True,
        truncation=True,
        max_length=512,
        return_tensors="pt"
    )

    labels = torch.stack(labels).unsqueeze(1)

    return encoded, labels


def train_linguistic_model(
    data_dir: str,
    epochs: int = 10,
    batch_size: int = 16,
    learning_rate: float = 1e-4,
    device: str = "cuda",
    cache_file: str = "transcript_cache.pkl"
):
    """
    Train the linguistic deepfake detection model.

    Args:
        data_dir: Path to data directory
        epochs: Number of training epochs
        batch_size: Batch size
        learning_rate: Learning rate
        device: Device to train on
        cache_file: Cache file for transcripts
    """
    print(f"Training linguistic model on {device}")

    # Load dataset
    dataset = TranscriptDeepfakeDataset(data_dir, cache_file=cache_file)

    # Split into train/val (80/20)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    # Initialize model
    model = LinguisticDetector(model_name=settings.LINGUISTIC_MODEL_NAME, freeze_encoder=False)
    model.to(device)

    # Create data loaders with custom collate function
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=lambda batch: collate_fn(batch, model.tokenizer)
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=lambda batch: collate_fn(batch, model.tokenizer)
    )

    # Loss and optimizer
    criterion = nn.BCELoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

    # Training loop
    best_val_loss = float('inf')

    for epoch in range(epochs):
        # Training phase
        model.train()
        train_loss = 0.0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} [Train]")
        for encoded, labels in pbar:
            input_ids = encoded['input_ids'].to(device)
            attention_mask = encoded['attention_mask'].to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = model(input_ids, attention_mask)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})

        train_loss /= len(train_loader)

        # Validation phase
        model.eval()
        val_loss = 0.0

        with torch.no_grad():
            for encoded, labels in tqdm(val_loader, desc=f"Epoch {epoch+1}/{epochs} [Val]"):
                input_ids = encoded['input_ids'].to(device)
                attention_mask = encoded['attention_mask'].to(device)
                labels = labels.to(device)

                outputs = model(input_ids, attention_mask)
                loss = criterion(outputs, labels)
                val_loss += loss.item()

        val_loss /= len(val_loader)

        print(f"Epoch {epoch+1}: Train Loss = {train_loss:.4f}, Val Loss = {val_loss:.4f}")

        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            checkpoint_path = settings.CHECKPOINTS_DIR / settings.LINGUISTIC_CHECKPOINT
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
            }, checkpoint_path)
            print(f"Saved best model to {checkpoint_path}")

    print("Training complete!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train linguistic deepfake detection model")
    parser.add_argument("--data_dir", type=str, required=True, help="Path to data directory")
    parser.add_argument("--epochs", type=int, default=10, help="Number of epochs")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--device", type=str, default="cuda", help="Device (cuda/cpu)")
    parser.add_argument("--cache", type=str, default="transcript_cache.pkl", help="Transcript cache file")

    args = parser.parse_args()

    train_linguistic_model(
        data_dir=args.data_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        device=args.device,
        cache_file=args.cache
    )
