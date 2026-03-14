"""Main detection endpoint for deepfake audio analysis."""

import os
import uuid
from pathlib import Path
from fastapi import APIRouter, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse

from schemas.detection import DetectionResponse, ErrorResponse
from services.detector_service import get_detector_service
from services.audio_utils import validate_audio_file
from config import settings

router = APIRouter()


@router.post("/detect", response_model=DetectionResponse, tags=["Detection"])
async def detect_deepfake(file: UploadFile = File(...)):
    """
    Detect if uploaded audio is real or fake.

    This endpoint:
    1. Validates the uploaded audio file
    2. Runs acoustic analysis (WavLM)
    3. Transcribes audio (Whisper)
    4. Runs linguistic analysis (RoBERTa)
    5. Fuses scores and returns verdict

    Args:
        file: Uploaded audio file (WAV, MP3, FLAC, etc.)

    Returns:
        DetectionResponse with verdict and detailed scores

    Raises:
        HTTPException: If file is invalid or processing fails
    """
    # Validate file type
    if not file.content_type or not file.content_type.startswith('audio/'):
        raise HTTPException(
            status_code=400,
            detail="Invalid file type. Please upload an audio file."
        )

    # Generate unique filename
    file_id = str(uuid.uuid4())
    file_extension = Path(file.filename).suffix
    temp_file_path = settings.UPLOAD_DIR / f"{file_id}{file_extension}"

    try:
        # Save uploaded file temporarily
        with open(temp_file_path, "wb") as f:
            content = await file.read()
            f.write(content)

        # Validate audio file
        validation = validate_audio_file(temp_file_path, max_duration=settings.MAX_AUDIO_LENGTH)
        if not validation['valid']:
            errors = "; ".join(validation['errors'])
            raise HTTPException(
                status_code=400,
                detail=f"Audio validation failed: {errors}"
            )

        # Run detection
        detector = get_detector_service()
        result = await detector.detect(temp_file_path)

        # Add timestamp
        from datetime import datetime
        result['timestamp'] = datetime.now()

        return DetectionResponse(**result)

    except HTTPException:
        raise

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Detection failed: {str(e)}"
        )

    finally:
        # Clean up temporary file
        if temp_file_path.exists():
            os.remove(temp_file_path)


@router.post("/detect/batch", tags=["Detection"])
async def detect_deepfake_batch(files: list[UploadFile] = File(...)):
    """
    Batch detection endpoint for multiple audio files.

    Args:
        files: List of uploaded audio files

    Returns:
        List of detection results
    """
    if len(files) > 10:
        raise HTTPException(
            status_code=400,
            detail="Maximum 10 files allowed per batch request"
        )

    results = []
    for file in files:
        try:
            result = await detect_deepfake(file)
            results.append({
                "filename": file.filename,
                "success": True,
                "result": result.dict()
            })
        except HTTPException as e:
            results.append({
                "filename": file.filename,
                "success": False,
                "error": e.detail
            })

    return {"results": results}
