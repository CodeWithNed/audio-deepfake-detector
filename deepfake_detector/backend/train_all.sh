#!/bin/bash

# Deepfake Audio Detector - Full Training Pipeline
# This script trains both acoustic and linguistic models

set -e  # Exit on error

echo "=========================================="
echo "  Deepfake Audio Detector Training"
echo "=========================================="
echo ""
echo "Dataset: 349 fake + 855 real = 1204 files"
echo ""

# Check if CUDA is available
if command -v nvidia-smi &> /dev/null; then
    echo "✓ GPU detected"
    DEVICE="cuda"
else
    echo "⚠ No GPU detected, using CPU (will be slow)"
    DEVICE="cpu"
fi
echo ""

# Step 1: Train Acoustic Model (WavLM)
echo "=========================================="
echo "Step 1/3: Training Acoustic Model (WavLM)"
echo "=========================================="
echo "This will take 30-60 minutes with GPU"
echo ""

python training/train_acoustic.py \
    --data_dir data/ \
    --epochs 10 \
    --batch_size 8 \
    --lr 1e-4 \
    --device $DEVICE

echo ""
echo "✓ Acoustic model training complete!"
echo ""

# Step 2: Train Linguistic Model (RoBERTa)
echo "=========================================="
echo "Step 2/3: Training Linguistic Model (RoBERTa)"
echo "=========================================="
echo "First run will transcribe all audio (60-90 min)"
echo "Subsequent runs use cache (15-30 min)"
echo ""

python training/train_linguistic.py \
    --data_dir data/ \
    --epochs 10 \
    --batch_size 16 \
    --lr 1e-4 \
    --device $DEVICE \
    --cache transcripts.pkl

echo ""
echo "✓ Linguistic model training complete!"
echo ""

# Step 3: Evaluate Models
echo "=========================================="
echo "Step 3/3: Evaluating Models"
echo "=========================================="
echo ""

python training/evaluate.py \
    --model_type both \
    --data_dir data/ \
    --device $DEVICE \
    --cache transcripts.pkl

echo ""
echo "=========================================="
echo "  Training Complete! 🎉"
echo "=========================================="
echo ""
echo "Trained models saved in: checkpoints/"
echo "  - acoustic_best.pt"
echo "  - linguistic_best.pt"
echo ""
echo "To start the API server:"
echo "  python main.py"
echo ""
