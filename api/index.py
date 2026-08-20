import os
import io
import base64
import requests
from contextlib import asynccontextmanager
from typing import Optional, List
from fastapi import FastAPI, File, UploadFile, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
from PIL import Image

from api.engine import AtlasEngine
from api.queue_manager import InferenceQueue

engine = AtlasEngine.get_instance()
inference_queue = InferenceQueue.get_instance()
PUBLIC_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "public")

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: Load model in background & start inference queue worker
    print("[API LIFESPAN] Initializing Atlas Engine and starting Queue Worker...")
    engine.load_model()
    await inference_queue.start_worker()
    yield
    # Shutdown: Gracefully stop worker
    print("[API LIFESPAN] Stopping Queue Worker...")
    await inference_queue.stop_worker()

app = FastAPI(
    title="Atlas Product Categorization API",
    description="Hierarchical Product Taxonomy Prediction using Constrained Beam Search & Spatial Attention (Queue-Backed)",
    version="2.0.0",
    docs_url="/api/docs",
    openapi_url="/api/openapi.json",
    lifespan=lifespan
)

# Enable CORS for frontend integrations
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class UrlPredictionRequest(BaseModel):
    image_url: str
    beam_size: Optional[int] = 5

class Base64PredictionRequest(BaseModel):
    image_base64: str
    beam_size: Optional[int] = 5

class CategoryPredictionResponse(BaseModel):
    gender: str
    master_category: str
    sub_category: str
    taxonomy_path: List[str]
    confidence_score: float
    log_prob_score: float
    attention_image_base64: Optional[str] = None

@app.get("/api/health")
def health_check():
    weights_info = engine.get_weights_status()
    queue_info = inference_queue.get_status()
    return {
        "status": "healthy",
        "model_loaded": engine.loaded,
        "device": str(engine.device),
        "total_taxonomy_categories": len(engine.valid_wordmap_seq),
        "weights_status": weights_info,
        "queue_status": queue_info
    }

@app.get("/api/queue-status")
def get_queue_status():
    """Returns real-time inference queue telemetry and worker metrics."""
    return inference_queue.get_status()

@app.get("/api/weights-status")
def check_weights():
    """Returns detailed weight verification and memory retention status."""
    return engine.get_weights_status()

@app.get("/api/taxonomy")
def get_taxonomy():
    """Returns the complete 52-category hierarchical taxonomy tree."""
    return {
        "total_paths": len(engine.valid_wordmap_seq),
        "taxonomy_tree": engine.taxonomy_tree
    }

@app.post("/api/predict", response_model=CategoryPredictionResponse)
async def predict_image_file(
    file: UploadFile = File(...),
    beam_size: int = Query(5, ge=1, le=10, description="Beam size for Constrained Beam Search")
):
    """
    Predict hierarchical clothing taxonomy from an uploaded image file (JPEG/PNG/WebP/AVIF).
    Enqueued into async task queue for concurrency-safe inference.
    """
    try:
        contents = await file.read()
        image = Image.open(io.BytesIO(contents))
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid image file: {str(e)}")

    result = await inference_queue.submit(image, beam_size=beam_size)
    if "error" in result:
        raise HTTPException(status_code=500, detail=result["error"])
    return result

@app.post("/api/predict-url", response_model=CategoryPredictionResponse)
async def predict_image_url(request: UrlPredictionRequest):
    """
    Predict hierarchical clothing taxonomy from a public image URL.
    Enqueued into async task queue for concurrency-safe inference.
    """
    try:
        headers = {"User-Agent": "Mozilla/5.0"}
        resp = requests.get(request.image_url, headers=headers, timeout=10)
        resp.raise_for_status()
        image = Image.open(io.BytesIO(resp.content))
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to fetch image from URL: {str(e)}")

    result = await inference_queue.submit(image, beam_size=request.beam_size)
    if "error" in result:
        raise HTTPException(status_code=500, detail=result["error"])
    return result

@app.post("/api/predict-base64", response_model=CategoryPredictionResponse)
async def predict_image_base64(request: Base64PredictionRequest):
    """
    Predict hierarchical clothing taxonomy from a Base64-encoded image string.
    Enqueued into async task queue for concurrency-safe inference.
    """
    try:
        b64_str = request.image_base64
        if "," in b64_str:
            b64_str = b64_str.split(",", 1)[1]
        decoded = base64.b64decode(b64_str)
        image = Image.open(io.BytesIO(decoded))
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid base64 payload: {str(e)}")

    result = await inference_queue.submit(image, beam_size=request.beam_size)
    if "error" in result:
        raise HTTPException(status_code=500, detail=result["error"])
    return result

# Serve static web frontend for testing
if os.path.exists(PUBLIC_DIR):
    @app.get("/")
    def serve_index():
        return FileResponse(os.path.join(PUBLIC_DIR, "index.html"))

    app.mount("/", StaticFiles(directory=PUBLIC_DIR, html=True), name="public")


