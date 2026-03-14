"""Pydantic schemas for detection API requests and responses."""

from pydantic import BaseModel, Field
from datetime import datetime
from typing import Optional


class DetectionResponse(BaseModel):
    """Response schema for deepfake detection."""

    decision: str = Field(..., description="Detection decision: REAL or FAKE")
    final_score: float = Field(..., ge=0.0, le=1.0, description="Final fused score (0-1)")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Confidence of prediction (0-1)")
    acoustic_score: float = Field(..., ge=0.0, le=1.0, description="Acoustic model score (0-1)")
    linguistic_score: float = Field(..., ge=0.0, le=1.0, description="Linguistic model score (0-1)")
    transcript: str = Field(..., description="ASR transcript")
    asr_confidence: float = Field(..., ge=0.0, le=1.0, description="ASR confidence score")
    language: str = Field(..., description="Detected language code")
    timestamp: datetime = Field(default_factory=datetime.now, description="Detection timestamp")

    class Config:
        json_schema_extra = {
            "example": {
                "decision": "FAKE",
                "final_score": 0.72,
                "confidence": 0.44,
                "acoustic_score": 0.85,
                "linguistic_score": 0.52,
                "transcript": "This is a sample audio transcript.",
                "asr_confidence": 0.95,
                "language": "en",
                "timestamp": "2024-01-15T10:30:00"
            }
        }


class HealthResponse(BaseModel):
    """Response schema for health check."""

    status: str = Field(..., description="Service status")
    version: str = Field(..., description="API version")
    models_loaded: bool = Field(..., description="Whether models are loaded")
    device: str = Field(..., description="Device being used (cpu/cuda)")

    class Config:
        json_schema_extra = {
            "example": {
                "status": "healthy",
                "version": "1.0.0",
                "models_loaded": True,
                "device": "cuda"
            }
        }


class DetectionHistoryItem(BaseModel):
    """Schema for a single detection history item."""

    id: str = Field(..., description="Unique detection ID")
    filename: str = Field(..., description="Original filename")
    decision: str = Field(..., description="Detection decision")
    final_score: float = Field(..., description="Final score")
    confidence: float = Field(..., description="Confidence score")
    transcript: str = Field(..., description="Transcript")
    timestamp: datetime = Field(..., description="Detection timestamp")


class DetectionHistoryResponse(BaseModel):
    """Response schema for detection history."""

    total: int = Field(..., description="Total number of detections")
    items: list[DetectionHistoryItem] = Field(..., description="List of detection history items")

    class Config:
        json_schema_extra = {
            "example": {
                "total": 2,
                "items": [
                    {
                        "id": "abc123",
                        "filename": "sample.wav",
                        "decision": "FAKE",
                        "final_score": 0.72,
                        "confidence": 0.44,
                        "transcript": "Sample transcript",
                        "timestamp": "2024-01-15T10:30:00"
                    }
                ]
            }
        }


class ErrorResponse(BaseModel):
    """Response schema for errors."""

    error: str = Field(..., description="Error message")
    detail: Optional[str] = Field(None, description="Detailed error information")

    class Config:
        json_schema_extra = {
            "example": {
                "error": "Invalid audio file",
                "detail": "Audio duration (35s) exceeds maximum (30s)"
            }
        }
