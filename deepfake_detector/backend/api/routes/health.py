"""Health check endpoint."""

from fastapi import APIRouter
from schemas.detection import HealthResponse
from config import settings
from services.detector_service import get_detector_service

router = APIRouter()


@router.get("/health", response_model=HealthResponse, tags=["Health"])
async def health_check():
    """
    Health check endpoint to verify service is running.

    Returns service status, version, and model loading status.
    """
    detector = get_detector_service()

    # Check if models are loaded
    models_loaded = (
        detector.acoustic_model is not None and
        detector.linguistic_model is not None and
        detector.asr_service is not None
    )

    return HealthResponse(
        status="healthy",
        version=settings.VERSION,
        models_loaded=models_loaded,
        device=detector.device
    )
