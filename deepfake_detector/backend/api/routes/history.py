"""Detection history endpoint (simplified in-memory storage)."""

from fastapi import APIRouter
from schemas.detection import DetectionHistoryResponse, DetectionHistoryItem
from datetime import datetime

router = APIRouter()

# In-memory storage for detection history (for demo purposes)
# In production, use a database like PostgreSQL or MongoDB
_detection_history: list[dict] = []


@router.get("/history", response_model=DetectionHistoryResponse, tags=["History"])
async def get_history(limit: int = 50):
    """
    Get detection history.

    Args:
        limit: Maximum number of items to return (default: 50)

    Returns:
        List of past detection results
    """
    # Return most recent detections
    recent_history = _detection_history[-limit:] if _detection_history else []

    items = [
        DetectionHistoryItem(**item)
        for item in reversed(recent_history)  # Most recent first
    ]

    return DetectionHistoryResponse(
        total=len(_detection_history),
        items=items
    )


@router.delete("/history", tags=["History"])
async def clear_history():
    """Clear all detection history."""
    global _detection_history
    count = len(_detection_history)
    _detection_history = []

    return {
        "message": f"Cleared {count} detection records",
        "success": True
    }


@router.delete("/history/{detection_id}", tags=["History"])
async def delete_history_item(detection_id: str):
    """Delete a specific detection from history."""
    global _detection_history

    initial_count = len(_detection_history)
    _detection_history = [
        item for item in _detection_history
        if item.get('id') != detection_id
    ]

    if len(_detection_history) < initial_count:
        return {
            "message": f"Deleted detection {detection_id}",
            "success": True
        }
    else:
        return {
            "message": f"Detection {detection_id} not found",
            "success": False
        }


def add_to_history(detection_result: dict, filename: str):
    """
    Add a detection result to history (internal function).

    Args:
        detection_result: Detection result dictionary
        filename: Original filename
    """
    import uuid

    history_item = {
        "id": str(uuid.uuid4()),
        "filename": filename,
        "decision": detection_result.get('decision'),
        "final_score": detection_result.get('final_score'),
        "confidence": detection_result.get('confidence'),
        "transcript": detection_result.get('transcript', '')[:200],  # Truncate long transcripts
        "timestamp": detection_result.get('timestamp', datetime.now())
    }

    _detection_history.append(history_item)

    # Keep only last 1000 items to prevent memory issues
    if len(_detection_history) > 1000:
        _detection_history.pop(0)
