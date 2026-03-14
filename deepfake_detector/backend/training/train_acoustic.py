"""Training script for acoustic model (WavLM)."""

import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm
import argparse

from config import settings
from models.acoustic_model import AcousticDetector
from training.dataset import AudioDeepfakeDataset


def train_acoustic_model(
    data_dir: str,
    epochs: int = 10,
    batch_size: int = 16,
    learning_rate: float = 1e-4,
    device: str = "cuda"
):
    """
    Train the acoustic deepfake detection model.

    Args:
        data_dir: Path to data directory
        epochs: Number of training epochs
        batch_size: Batch size
        learning_rate: Learning rate
        device: Device to train on
    """
    print(f"Training acoustic model on {device}")

    # Load dataset
    dataset = AudioDeepfakeDataset(data_dir, sample_rate=settings.SAMPLE_RATE)

    # Split into train/val (80/20)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

    # Initialize model
    model = AcousticDetector(model_name=settings.ACOUSTIC_MODEL_NAME, freeze_encoder=False)
    model.to(device)

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
        for audio, labels in pbar:
            audio, labels = audio.to(device), labels.to(device).unsqueeze(1)

            optimizer.zero_grad()
            outputs = model(audio)
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
            for audio, labels in tqdm(val_loader, desc=f"Epoch {epoch+1}/{epochs} [Val]"):
                audio, labels = audio.to(device), labels.to(device).unsqueeze(1)
                outputs = model(audio)
                loss = criterion(outputs, labels)
                val_loss += loss.item()

        val_loss /= len(val_loader)

        print(f"Epoch {epoch+1}: Train Loss = {train_loss:.4f}, Val Loss = {val_loss:.4f}")

        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            checkpoint_path = settings.CHECKPOINTS_DIR / settings.ACOUSTIC_CHECKPOINT
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
            }, checkpoint_path)
            print(f"Saved best model to {checkpoint_path}")

    print("Training complete!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train acoustic deepfake detection model")
    parser.add_argument("--data_dir", type=str, required=True, help="Path to data directory")
    parser.add_argument("--epochs", type=int, default=10, help="Number of epochs")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--device", type=str, default="cuda", help="Device (cuda/cpu)")

    args = parser.parse_args()

    train_acoustic_model(
        data_dir=args.data_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        device=args.device
    )
