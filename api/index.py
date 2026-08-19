import io
import base64
import requests
from typing import Optional, List
from fastapi import FastAPI, File, UploadFile, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, HttpUrl
from PIL import Image

from api.engine import AtlasEngine

app = FastAPI(
    title="Atlas Product Categorization API",
    description="Hierarchical Product Taxonomy Prediction using Constrained Beam Search & Spatial Attention",
    version="1.0.0",
    docs_url="/api/docs",
    openapi_url="/api/openapi.json"
)

# Enable CORS for frontend integrations
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

engine = AtlasEngine.get_instance()

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
    return {
        "status": "healthy",
        "model_loaded": engine.loaded,
        "device": str(engine.device),
        "total_taxonomy_categories": len(engine.valid_wordmap_seq)
    }

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
    Predict hierarchical clothing taxonomy from an uploaded image file (JPEG/PNG/WebP).
    """
    try:
        contents = await file.read()
        image = Image.open(io.BytesIO(contents))
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid image file: {str(e)}")

    result = engine.predict_image(image, beam_size=beam_size)
    if "error" in result:
        raise HTTPException(status_code=500, detail=result["error"])
    return result

@app.post("/api/predict-url", response_model=CategoryPredictionResponse)
async def predict_image_url(request: UrlPredictionRequest):
    """
    Predict hierarchical clothing taxonomy from a public image URL.
    """
    try:
        headers = {"User-Agent": "Mozilla/5.0"}
        resp = requests.get(request.image_url, headers=headers, timeout=10)
        resp.raise_for_status()
        image = Image.open(io.BytesIO(resp.content))
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to fetch image from URL: {str(e)}")

    result = engine.predict_image(image, beam_size=request.beam_size)
    if "error" in result:
        raise HTTPException(status_code=500, detail=result["error"])
    return result

@app.post("/api/predict-base64", response_model=CategoryPredictionResponse)
async def predict_image_base64(request: Base64PredictionRequest):
    """
    Predict hierarchical clothing taxonomy from a Base64-encoded image string.
    """
    try:
        b64_str = request.image_base64
        if "," in b64_str:
            b64_str = b64_str.split(",", 1)[1]
        decoded = base64.b64decode(b64_str)
        image = Image.open(io.BytesIO(decoded))
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid base64 payload: {str(e)}")

    result = engine.predict_image(image, beam_size=request.beam_size)
    if "error" in result:
        raise HTTPException(status_code=500, detail=result["error"])
    return result
