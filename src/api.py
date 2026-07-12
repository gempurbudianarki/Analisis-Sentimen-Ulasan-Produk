import os
import sys
from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
import pandas as pd
import io

# Add path so we can import modules
base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(base_dir)
sys.path.append(os.path.join(base_dir, 'src'))

from predict import SentimentPredictor
import database as db

# Initialize FastAPI App
app = FastAPI(
    title="Sentiment Analysis API",
    description="Industrial-grade decoupled NLP Sentiment Prediction engine for Indonesian e-commerce ulasan.",
    version="1.0.0"
)

# Enable CORS for external/browser clients
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Lazy initialization of predictor
predictor = None

@app.on_event("startup")
def startup_event():
    global predictor
    # Initialize SQLite schema
    db.init_db()
    # Initialize Predictor
    try:
        predictor = SentimentPredictor()
        print("[API] Sentiment model loaded successfully on startup.")
    except Exception as e:
        print(f"[API ERROR] Model files could not be loaded: {e}")

# Request Models
class PredictionRequest(BaseModel):
    text: str

class FeedbackRequest(BaseModel):
    log_id: int
    correct_sentiment: str
    notes: Optional[str] = None

@app.post("/api/predict")
async def predict_single(request: PredictionRequest):
    global predictor
    if predictor is None:
        raise HTTPException(status_code=503, detail="Sentiment model is not loaded/compiled.")
        
    text = request.text
    if not text.strip():
        raise HTTPException(status_code=400, detail="Text cannot be empty.")
        
    try:
        # Run prediction
        res = predictor.predict(text)
        
        # Log to Database
        log_id = db.log_prediction(
            review_text=text,
            cleaned_text=res['clean_text'],
            predicted_sentiment=res['sentiment'],
            confidence=res['confidence'],
            model_name=type(predictor.model).__name__
        )
        
        # Add log_id to response so UI can link feedback
        res['log_id'] = log_id
        return res
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")

@app.post("/api/predict-batch")
async def predict_batch(file: UploadFile = File(...), text_column: str = Form(...)):
    global predictor
    if predictor is None:
        raise HTTPException(status_code=503, detail="Sentiment model is not loaded/compiled.")
        
    try:
        contents = await file.read()
        df = pd.read_csv(io.BytesIO(contents))
        
        if text_column not in df.columns:
            raise HTTPException(status_code=400, detail=f"Column '{text_column}' not found in CSV.")
            
        results = []
        for i, text in enumerate(df[text_column]):
            txt_str = str(text)
            res = predictor.predict(txt_str)
            
            # Log each prediction to DB
            log_id = db.log_prediction(
                review_text=txt_str,
                cleaned_text=res['clean_text'],
                predicted_sentiment=res['sentiment'],
                confidence=res['confidence'],
                model_name=type(predictor.model).__name__
            )
            
            results.append({
                "review_text": txt_str,
                "cleaned_text": res['clean_text'],
                "predicted_sentiment": res['sentiment'],
                "confidence": res['confidence'],
                "log_id": log_id
            })
            
        return results
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Batch processing error: {str(e)}")

@app.post("/api/feedback")
async def submit_feedback(request: FeedbackRequest):
    try:
        db.log_feedback(
            log_id=request.log_id,
            correct_sentiment=request.correct_sentiment,
            notes=request.notes
        )
        return {"success": True, "message": "Feedback submitted successfully."}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Feedback log error: {str(e)}")

@app.get("/api/stats")
async def get_statistics():
    try:
        return db.get_stats()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error fetching stats: {str(e)}")

@app.get("/api/recent")
async def get_recent(limit: int = 25):
    try:
        return db.get_recent_logs(limit)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error fetching logs: {str(e)}")
