from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Dict, Any
import numpy as np
import pandas as pd
import language_tool_python
import textstat
import pickle
import os
import sys
import re
from pathlib import Path
from scipy.sparse import hstack, csr_matrix

# Path setup
backend_dir = Path(__file__).parent
sys.path.append(str(backend_dir))

# Import rewrite module
try:
    from rewrite import rewrite_text
except ImportError:
    def rewrite_text(text: str) -> str:
        return f"[Rewrite module not available] {text}"

# App setup
app = FastAPI(title="AI vs Human Text Detector API", version="1.0.0")

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Globals
model = None
vectorizer = None
engineered_dim = None
grammar_tool = None

# Input/output models
class TextRequest(BaseModel):
    text: str

class PredictionResponse(BaseModel):
    prediction: str
    confidence: float

class HumanizeResponse(BaseModel):
    rewritten_text: str

# Load models
def load_models():
    global model, vectorizer, engineered_dim, grammar_tool
    try:
        with open("backend/model/ai_model_bundle.pkl", "rb") as f:
            bundle = pickle.load(f)
        model = bundle["model"]
        vectorizer = bundle["vectorizer"]
        engineered_dim = bundle["engineered_dim"]
        grammar_tool = language_tool_python.LanguageTool('en-US')
        print("✅ Models loaded successfully!")
    except Exception as e:
        print(f"❌ Error loading models: {e}")
        raise e

# Feature engineering
def extract_linguistic_features(text: str) -> np.ndarray:
    sentences = [s for s in re.split(r'[.!?]+', text) if s.strip()]
    words = re.findall(r'\b\w+\b', text)

    num_words = len(words)
    num_sentences = len(sentences)
    characters = len(text)

    lexical_diversity = len(set(words)) / num_words if num_words else 0
    avg_sentence_length = num_words / num_sentences if num_sentences else 0
    avg_word_length = sum(len(w) for w in words) / num_words if num_words else 0
    punctuation_ratio = len(re.findall(r'[^\w\s]', text)) / characters if characters else 0
    flesch_reading_ease = textstat.flesch_reading_ease(text)
    gunning_fog_index = textstat.gunning_fog(text)
    grammar_errors = len(grammar_tool.check(text)) if grammar_tool else 0
    passive_voice_ratio = len(re.findall(r'\b(is|was|were|been|being|are|am)\b\s+\w+ed', text)) / num_sentences if num_sentences else 0
    common_words = set(["the", "and", "is", "in", "it", "you", "that", "he", "was", "for", "on", "are", "with", "as", "I", "his", "they", "be", "at", "one", "have", "this"])
    predictability_score = sum(1 for w in words if w.lower() in common_words) / num_words if num_words else 0
    burstiness = lexical_diversity * avg_sentence_length
    sentiment_score = textstat.flesch_kincaid_grade(text)

    return np.array([
        num_words, characters, num_sentences, lexical_diversity,
        avg_sentence_length, avg_word_length, punctuation_ratio,
        flesch_reading_ease, gunning_fog_index, grammar_errors,
        passive_voice_ratio, predictability_score, burstiness,
        sentiment_score
    ])

# On startup
@app.on_event("startup")
async def startup_event():
    load_models()

@app.get("/")
async def root():
    return {"message": "AI vs Human Text Detector API is running!"}

@app.post("/predict", response_model=PredictionResponse)
async def predict_text(request: TextRequest):
    if not model or not vectorizer:
        raise HTTPException(status_code=500, detail="Models not loaded properly")
    if not request.text.strip():
        raise HTTPException(status_code=400, detail="Text cannot be empty")

    try:
        feats = extract_linguistic_features(request.text)
        feats_sparse = csr_matrix(feats.reshape(1, -1))
        
        # Check feature dimensions
        if feats_sparse.shape[1] != engineered_dim:
            raise HTTPException(status_code=500, detail=f"Feature mismatch: expected {engineered_dim}, got {feats_sparse.shape[1]}")
       

        tfidf_vec = vectorizer.transform([request.text])
        combined = hstack([feats_sparse, tfidf_vec])
        print("TF-IDF shape:", tfidf_vec.shape)
        print("Engineered shape:", feats_sparse.shape)
        print("Expected engineered_dim:", engineered_dim)
        pred = model.predict(combined)[0]
        prob = model.predict_proba(combined)[0]

        label = "AI" if pred == 1 else "Human"
        confidence = round(100 * max(prob), 2)

        return PredictionResponse(prediction=label, confidence=confidence)

    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")


@app.post("/humanize", response_model=HumanizeResponse)
async def humanize_text(request: TextRequest):
    if not request.text.strip():
        raise HTTPException(status_code=400, detail="Text cannot be empty")

    try:
        rewritten = rewrite_text(request.text)
        return HumanizeResponse(rewritten_text=rewritten)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Humanization error: {str(e)}")

# Run the server
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
