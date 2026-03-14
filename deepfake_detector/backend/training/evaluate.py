"""Evaluation script for deepfake detection models."""

import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, roc_curve
import numpy as np
import argparse

from config import settings
from models.acoustic_model import load_acoustic_model
from models.linguistic_model import load_linguistic_model
from training.dataset import AudioDeepfakeDataset, TranscriptDeepfakeDataset


def calculate_eer(y_true, y_scores):
    """Calculate Equal Error Rate (EER)."""
    fpr, tpr, thresholds = roc_curve(y_true, y_scores)
    fnr = 1 - tpr
    eer_threshold = thresholds[np.nanargmin(np.absolute((fnr - fpr)))]
    eer = fpr[np.nanargmin(np.absolute((fnr - fpr)))]
    return eer, eer_threshold


def evaluate_acoustic_model(model_path: str, data_dir: str, device: str = "cuda"):
    """Evaluate acoustic model."""
    print("Evaluating acoustic model...")

    # Load model
    model = load_acoustic_model(model_path, device=device)

    # Load test dataset
    dataset = AudioDeepfakeDataset(data_dir, sample_rate=settings.SAMPLE_RATE)
    loader = DataLoader(dataset, batch_size=16, shuffle=False, num_workers=4)

    # Evaluate
    all_labels = []
    all_scores = []

    model.eval()
    with torch.no_grad():
        for audio, labels in loader:
            audio = audio.to(device)
            outputs = model(audio)

            all_labels.extend(labels.cpu().numpy())
            all_scores.extend(outputs.cpu().numpy().squeeze())

    all_labels = np.array(all_labels)
    all_scores = np.array(all_scores)

    # Calculate metrics
    predictions = (all_scores >= 0.5).astype(int)

    accuracy = accuracy_score(all_labels, predictions)
    precision = precision_score(all_labels, predictions)
    recall = recall_score(all_labels, predictions)
    f1 = f1_score(all_labels, predictions)
    auc = roc_auc_score(all_labels, all_scores)
    eer, eer_threshold = calculate_eer(all_labels, all_scores)

    print("\nAcoustic Model Results:")
    print(f"  Accuracy:  {accuracy:.4f}")
    print(f"  Precision: {precision:.4f}")
    print(f"  Recall:    {recall:.4f}")
    print(f"  F1-Score:  {f1:.4f}")
    print(f"  AUC:       {auc:.4f}")
    print(f"  EER:       {eer:.4f} (threshold: {eer_threshold:.4f})")

    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'auc': auc,
        'eer': eer
    }


def evaluate_linguistic_model(model_path: str, data_dir: str, device: str = "cuda", cache_file: str = "transcript_cache.pkl"):
    """Evaluate linguistic model."""
    print("Evaluating linguistic model...")

    # Load model
    model = load_linguistic_model(model_path, device=device)

    # Load test dataset
    dataset = TranscriptDeepfakeDataset(data_dir, cache_file=cache_file)

    all_labels = []
    all_scores = []

    model.eval()
    with torch.no_grad():
        for transcript, label in dataset:
            # Tokenize
            encoded = model.tokenize([transcript])
            input_ids = encoded['input_ids'].to(device)
            attention_mask = encoded['attention_mask'].to(device)

            # Predict
            output = model(input_ids, attention_mask)

            all_labels.append(label)
            all_scores.append(output.cpu().item())

    all_labels = np.array(all_labels)
    all_scores = np.array(all_scores)

    # Calculate metrics
    predictions = (all_scores >= 0.5).astype(int)

    accuracy = accuracy_score(all_labels, predictions)
    precision = precision_score(all_labels, predictions)
    recall = recall_score(all_labels, predictions)
    f1 = f1_score(all_labels, predictions)
    auc = roc_auc_score(all_labels, all_scores)
    eer, eer_threshold = calculate_eer(all_labels, all_scores)

    print("\nLinguistic Model Results:")
    print(f"  Accuracy:  {accuracy:.4f}")
    print(f"  Precision: {precision:.4f}")
    print(f"  Recall:    {recall:.4f}")
    print(f"  F1-Score:  {f1:.4f}")
    print(f"  AUC:       {auc:.4f}")
    print(f"  EER:       {eer:.4f} (threshold: {eer_threshold:.4f})")

    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'auc': auc,
        'eer': eer
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate deepfake detection models")
    parser.add_argument("--model_type", type=str, choices=["acoustic", "linguistic", "both"], required=True)
    parser.add_argument("--model_path", type=str, help="Path to model checkpoint")
    parser.add_argument("--data_dir", type=str, required=True, help="Path to test data directory")
    parser.add_argument("--device", type=str, default="cuda", help="Device (cuda/cpu)")
    parser.add_argument("--cache", type=str, default="transcript_cache.pkl", help="Transcript cache file")

    args = parser.parse_args()

    if args.model_type in ["acoustic", "both"]:
        acoustic_path = args.model_path or str(settings.CHECKPOINTS_DIR / settings.ACOUSTIC_CHECKPOINT)
        evaluate_acoustic_model(acoustic_path, args.data_dir, args.device)

    if args.model_type in ["linguistic", "both"]:
        linguistic_path = args.model_path or str(settings.CHECKPOINTS_DIR / settings.LINGUISTIC_CHECKPOINT)
        evaluate_linguistic_model(linguistic_path, args.data_dir, args.device, args.cache)
