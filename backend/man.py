from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import pickle
import numpy as np
import pandas as pd
import language_tool_python
import textstat
from typing import Dict, Any
import os
import sys
from pathlib import Path
import joblib

# Add the backend directory to Python path for imports
backend_dir = Path(__file__).parent
sys.path.append(str(backend_dir))

# Import the rewrite module
try:
    from rewrite import rewrite_text
except ImportError:
    print("Warning: rewrite.py module not found. Humanize functionality will not work.")
    def rewrite_text(text: str) -> str:
        return f"[Rewrite module not available] Original text: {text}"

app = FastAPI(title="AI vs Human Text Detector API", version="1.0.0")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins for development; restrict in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables for models
model = None
vectorizer = None
grammar_tool = None

class TextRequest(BaseModel):
    text: str

class PredictionResponse(BaseModel):
    prediction: str
    confidence: float

class HumanizeResponse(BaseModel):
    rewritten_text: str

def load_models():
    """Load the trained models and tools"""
    global model, vectorizer, grammar_tool
    
    try:
        # Load the trained RandomForest model
        with open("backend/model/ai_model_bundle.pkl", "rb") as f:
                bundle = pickle.load(f)

        model = bundle["model"]
        vectorizer = bundle["vectorizer"]
        engineered_dim = bundle["engineered_dim"]
        
        # Initialize grammar tool
        grammar_tool = language_tool_python.LanguageTool('en-US')
        
        print("Models loaded successfully!")
        
    except Exception as e:
        print(f"Error loading models: {e}")
        raise e

def extract_linguistic_features(text: str) -> Dict[str, Any]:
    """Extract linguistic features from text for model prediction"""
    try:
        features = {}
        
        # Basic text statistics
        features['char_count'] = len(text)
        features['word_count'] = len(text.split())
        features['sentence_count'] = len([s for s in text.split('.') if s.strip()])
        features['avg_word_length'] = np.mean([len(word) for word in text.split()]) if text.split() else 0
        features['avg_sentence_length'] = features['word_count'] / max(features['sentence_count'], 1)
        
        # Readability scores using textstat
        features['flesch_reading_ease'] = textstat.flesch_reading_ease(text)
        features['flesch_kincaid_grade'] = textstat.flesch_kincaid_grade(text)
        features['automated_readability_index'] = textstat.automated_readability_index(text)
        features['coleman_liau_index'] = textstat.coleman_liau_index(text)
        features['gunning_fog'] = textstat.gunning_fog(text)
        features['smog_index'] = textstat.smog_index(text)
        features['dale_chall_readability_score'] = textstat.dale_chall_readability_score(text)
        
        # Grammar and style features using language_tool_python
        if grammar_tool:
            grammar_errors = grammar_tool.check(text)
            features['grammar_errors'] = len(grammar_errors)
            features['grammar_error_rate'] = len(grammar_errors) / max(features['word_count'], 1)
        else:
            features['grammar_errors'] = 0
            features['grammar_error_rate'] = 0
        
        # Punctuation features
        features['exclamation_count'] = text.count('!')
        features['question_count'] = text.count('?')
        features['comma_count'] = text.count(',')
        features['semicolon_count'] = text.count(';')
        features['colon_count'] = text.count(':')
        
        # Additional features
        features['uppercase_ratio'] = sum(1 for c in text if c.isupper()) / max(len(text), 1)
        features['digit_ratio'] = sum(1 for c in text if c.isdigit()) / max(len(text), 1)
        
        return features
        
    except Exception as e:
        print(f"Error extracting features: {e}")
        # Return basic features if advanced extraction fails
        return {
            'char_count': len(text),
            'word_count': len(text.split()),
            'sentence_count': len([s for s in text.split('.') if s.strip()]),
            'avg_word_length': np.mean([len(word) for word in text.split()]) if text.split() else 0,
            'flesch_reading_ease': 0,
            'grammar_errors': 0
        }

@app.on_event("startup")
async def startup_event():
    """Load models when the app starts"""
    load_models()

@app.get("/")
async def root():
    """Health check endpoint"""
    return {"message": "AI vs Human Text Detector API is running!"}

@app.post("/predict", response_model=PredictionResponse)
async def predict_text(request: TextRequest):
    """Predict if text is AI-generated or human-written"""
    if not model or not vectorizer:
        raise HTTPException(status_code=500, detail="Models not loaded properly")
    
    if not request.text.strip():
        raise HTTPException(status_code=400, detail="Text cannot be empty")
    
    try:
        # Extract TF-IDF features
        tfidf_features = vectorizer.transform([request.text])
        
        # Extract linguistic features
        linguistic_features = extract_linguistic_features(request.text)
        
        # Combine features (assuming the model was trained with combined features)
        # Note: This assumes your original model combines TF-IDF with linguistic features
        # You may need to adjust this based on your actual model training approach
        
        # For now, we'll use just TF-IDF features for prediction
        # You should modify this to match your original feature combination method
        prediction_proba = model.predict_proba(tfidf_features)[0]
        prediction = model.predict(tfidf_features)[0]
        
        # Assuming binary classification: 0 = Human, 1 = AI
        # Adjust these mappings based on your model's training labels
        if prediction == 1:
            pred_label = "AI"
            confidence = prediction_proba[1] * 100
        else:
            pred_label = "Human"
            confidence = prediction_proba[0] * 100
        
        return PredictionResponse(
            prediction=pred_label,
            confidence=round(confidence, 2)
        )
        
    except Exception as e:
        print(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail=f"Error making prediction: {str(e)}")

@app.post("/humanize", response_model=HumanizeResponse)
async def humanize_text(request: TextRequest):
    """Rewrite AI-generated text to make it more human-like"""
    if not request.text.strip():
        raise HTTPException(status_code=400, detail="Text cannot be empty")
    
    try:
        # Use the rewrite module to humanize the text
        rewritten_text = rewrite_text(request.text)
        
        return HumanizeResponse(
            rewritten_text=rewritten_text
        )
        
    except Exception as e:
        print(f"Humanization error: {e}")
        raise HTTPException(status_code=500, detail=f"Error humanizing text: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)