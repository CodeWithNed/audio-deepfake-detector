"""Dataset loader for training acoustic and linguistic models."""

import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
from torch.utils.data import Dataset
import numpy as np

from services.audio_utils import load_audio, normalize_audio
from services.asr_service import get_asr_service


class AudioDeepfakeDataset(Dataset):
    """
    Dataset for audio deepfake detection.

    Directory structure expected:
    data/
      ├── real/
      │   ├── audio1.wav
      │   ├── audio2.wav
      │   └── ...
      └── fake/
          ├── audio1.wav
          ├── audio2.wav
          └── ...
    """

    def __init__(self, data_dir: str, sample_rate: int = 16000, max_length: int = 48000):
        """
        Initialize dataset.

        Args:
            data_dir: Root data directory
            sample_rate: Audio sample rate
            max_length: Maximum audio length in samples
        """
        self.data_dir = Path(data_dir)
        self.sample_rate = sample_rate
        self.max_length = max_length

        # Collect all audio files
        self.samples = []

        # Real audio files (label=0)
        real_dir = self.data_dir / "real"
        if real_dir.exists():
            for audio_file in real_dir.glob("*.wav"):
                self.samples.append((audio_file, 0))

        # Fake audio files (label=1)
        fake_dir = self.data_dir / "fake"
        if fake_dir.exists():
            for audio_file in fake_dir.glob("*.wav"):
                self.samples.append((audio_file, 1))

        print(f"Loaded {len(self.samples)} audio files")
        print(f"  Real: {sum(1 for _, label in self.samples if label == 0)}")
        print(f"  Fake: {sum(1 for _, label in self.samples if label == 1)}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        """Get audio sample and label."""
        audio_path, label = self.samples[idx]

        # Load and preprocess audio
        audio, sr = load_audio(audio_path, target_sr=self.sample_rate)
        audio = normalize_audio(audio)

        # Pad or truncate to fixed length
        if len(audio) > self.max_length:
            audio = audio[:self.max_length]
        else:
            audio = np.pad(audio, (0, self.max_length - len(audio)), mode='constant')

        # Convert to tensor
        audio_tensor = torch.from_numpy(audio).float()
        label_tensor = torch.tensor(label, dtype=torch.float32)

        return audio_tensor, label_tensor


class TranscriptDeepfakeDataset(Dataset):
    """
    Dataset for transcript-based deepfake detection.

    This dataset transcribes audio files and uses the transcripts for training.
    """

    def __init__(self, data_dir: str, cache_file: str = None):
        """
        Initialize transcript dataset.

        Args:
            data_dir: Root data directory
            cache_file: Optional cache file for transcripts
        """
        self.data_dir = Path(data_dir)
        self.cache_file = cache_file

        # Try to load from cache
        if cache_file and Path(cache_file).exists():
            import pickle
            with open(cache_file, 'rb') as f:
                self.samples = pickle.load(f)
            print(f"Loaded {len(self.samples)} transcripts from cache")
        else:
            # Transcribe all audio files
            self.samples = self._transcribe_all()

            # Save to cache
            if cache_file:
                import pickle
                Path(cache_file).parent.mkdir(parents=True, exist_ok=True)
                with open(cache_file, 'wb') as f:
                    pickle.dump(self.samples, f)
                print(f"Saved transcripts to cache: {cache_file}")

    def _transcribe_all(self):
        """Transcribe all audio files in the dataset."""
        asr = get_asr_service(model_name="base")

        samples = []

        # Real audio files
        real_dir = self.data_dir / "real"
        if real_dir.exists():
            for audio_file in real_dir.glob("*.wav"):
                try:
                    result = asr.transcribe(audio_file)
                    samples.append((result['transcript'], 0))
                except Exception as e:
                    print(f"Failed to transcribe {audio_file}: {e}")

        # Fake audio files
        fake_dir = self.data_dir / "fake"
        if fake_dir.exists():
            for audio_file in fake_dir.glob("*.wav"):
                try:
                    result = asr.transcribe(audio_file)
                    samples.append((result['transcript'], 1))
                except Exception as e:
                    print(f"Failed to transcribe {audio_file}: {e}")

        print(f"Transcribed {len(samples)} audio files")
        return samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        """Get transcript and label."""
        transcript, label = self.samples[idx]
        return transcript, label
