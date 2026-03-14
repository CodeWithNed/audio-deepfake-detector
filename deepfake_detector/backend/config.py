"""Configuration settings for the deepfake detector backend."""

from pathlib import Path
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    # Application
    APP_NAME: str = "Deepfake Audio Detector"
    VERSION: str = "1.0.0"
    DEBUG: bool = True

    # Server
    HOST: str = "0.0.0.0"
    PORT: int = 8000

    # Paths
    BASE_DIR: Path = Path(__file__).parent
    CHECKPOINTS_DIR: Path = BASE_DIR / "checkpoints"
    DATA_DIR: Path = BASE_DIR / "data"
    UPLOAD_DIR: Path = BASE_DIR / "uploads"

    # Model settings
    ACOUSTIC_MODEL_NAME: str = "microsoft/wavlm-base-plus"
    LINGUISTIC_MODEL_NAME: str = "roberta-base"
    WHISPER_MODEL_NAME: str = "base"

    # Model checkpoints (trained weights)
    ACOUSTIC_CHECKPOINT: str = "acoustic_best.pt"
    LINGUISTIC_CHECKPOINT: str = "linguistic_best.pt"

    # Fusion weights
    ACOUSTIC_WEIGHT: float = 0.6
    LINGUISTIC_WEIGHT: float = 0.4

    # Detection threshold
    FAKE_THRESHOLD: float = 0.5

    # Audio processing
    SAMPLE_RATE: int = 16000
    MAX_AUDIO_LENGTH: int = 30  # seconds
    AUDIO_CHUNK_SIZE: int = 10  # seconds for long audio

    # Training
    BATCH_SIZE: int = 16
    LEARNING_RATE: float = 1e-4
    NUM_EPOCHS: int = 10

    # Device
    DEVICE: str = "cuda"  # Will fallback to cpu if CUDA not available

    # CORS
    CORS_ORIGINS: list[str] = ["http://localhost:5173", "http://localhost:3000"]

    class Config:
        env_file = ".env"
        case_sensitive = True


# Create singleton settings instance
settings = Settings()

# Create necessary directories
settings.CHECKPOINTS_DIR.mkdir(parents=True, exist_ok=True)
settings.DATA_DIR.mkdir(parents=True, exist_ok=True)
settings.UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
