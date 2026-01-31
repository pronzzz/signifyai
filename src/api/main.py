from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.requests import Request
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import shutil
from pathlib import Path
import sys
import torch
from PIL import Image
import io

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(PROJECT_ROOT))

from src.inference.detector import GestureDetector
from src.core.config import MODEL_PATH

app = FastAPI(title="SignifyAI - Hand Gesture Recognition")

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Static files and templates
app.mount("/static", StaticFiles(directory=str(PROJECT_ROOT / "src/api/static")), name="static")
templates = Jinja2Templates(directory=str(PROJECT_ROOT / "src/api/templates"))

# Load model
detector = None

@app.on_event("startup")
def load_model():
    global detector
    print("Loading model...")
    # Initialize detector (which loads model)
    # Note: In prod, might want to load this more carefully or lazily
    try:
        detector = GestureDetector(model_path=MODEL_PATH)
    except Exception as e:
        print(f"Failed to load model: {e}")
        detector = None

@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/predict")
async def predict_image(file: UploadFile = File(...)):
    if detector is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
        
    try:
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert('RGB')
        
        # Transform for inference (reuse detector logic)
        # Note: Detector expects (frame, roi) or we can expose a direct predict_image method
        # Let's add a direct image prediction method to detector or handle it here
        # For simplicity, we'll implement a preprocess helper here essentially duplicating detector's logic 
        # or better yet, assume the user sends a cropped hand image or we accept full image
        
        # Let's assume the web app sends the full image and we pretend it's the ROI for now
        # or better: The frontend captures the video frame.
        
        input_tensor = detector.transform(image).unsqueeze(0).to(detector.device)
        
        with torch.no_grad():
            outputs = detector.model(input_tensor)
            probs = torch.nn.functional.softmax(outputs, dim=1)
            conf, preds = torch.max(probs, 1)
            
        label = detector.classes[preds.item()]
        confidence = float(conf.item())
        
        return {"input": file.filename, "label": label, "confidence": confidence}
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
