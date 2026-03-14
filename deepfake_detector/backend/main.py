"""FastAPI application entry point."""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from config import settings
from api.routes import health, detect, history

# Create FastAPI app
app = FastAPI(
    title=settings.APP_NAME,
    version=settings.VERSION,
    description="Multimodal deepfake audio detection API using WavLM, Whisper, and RoBERTa",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(health.router, prefix="/api")
app.include_router(detect.router, prefix="/api")
app.include_router(history.router, prefix="/api")


@app.on_event("startup")
async def startup_event():
    """Initialize services on startup."""
    print(f"Starting {settings.APP_NAME} v{settings.VERSION}")
    print(f"Environment: {'DEBUG' if settings.DEBUG else 'PRODUCTION'}")
    print(f"Server: {settings.HOST}:{settings.PORT}")

    # Pre-load models (optional, can also do lazy loading)
    if not settings.DEBUG:
        from services.detector_service import get_detector_service
        print("Pre-loading models...")
        detector = get_detector_service()
        detector.load_models()
        print("Models loaded successfully")


@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown."""
    print(f"Shutting down {settings.APP_NAME}")


@app.get("/", tags=["Root"])
async def root():
    """Root endpoint with API information."""
    return {
        "name": settings.APP_NAME,
        "version": settings.VERSION,
        "status": "running",
        "docs": "/docs",
        "health": "/api/health"
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "main:app",
        host=settings.HOST,
        port=settings.PORT,
        reload=settings.DEBUG,
        log_level="info"
    )
