# Setup Instructions for AI vs Human Text Detector

## Quick Start Guide

### 1. Frontend Setup (Already Done)
The React frontend is already configured and running. You can see it at `http://localhost:5173`

### 2. Backend Setup

#### Step 1: Navigate to backend directory
```bash
cd backend
```

#### Step 2: Create virtual environment
```bash
python -m venv venv

# On Windows:
venv\Scripts\activate

# On macOS/Linux:
source venv/bin/activate
```

#### Step 3: Install dependencies
```bash
pip install -r requirements.txt
```

#### Step 4: Add your model files
Place these files in the `backend/model/` directory:
- `ai_detector_tfidf.pkl` (your trained RandomForestClassifier)
- `vectorizer.pkl` (your trained TfidfVectorizer)

#### Step 5: Start the backend server
```bash
python main.py
```

The backend will be available at `http://localhost:8000`

### 3. Test the Application

1. Make sure both frontend (port 5173) and backend (port 8000) are running
2. Open `http://localhost:5173` in your browser
3. Enter some text and click "Detect AI vs Human"
4. If detected as AI, use the "Humanize Text" button

## File Structure

```
├── src/                    # React Frontend
│   ├── App.tsx            # Main application component
│   └── ...
├── backend/               # FastAPI Backend
│   ├── main.py           # FastAPI application
│   ├── rewrite.py        # T5 text rewriting module
│   ├── model/            # ML model files (add your models here)
│   │   ├── ai_detector_tfidf.pkl
│   │   └── vectorizer.pkl
│   └── requirements.txt  # Python dependencies
└── README.md
```

## Important Notes

1. **Model Files**: You must add your trained model files to `backend/model/`
2. **Feature Engineering**: The backend includes linguistic feature extraction using `language_tool_python` and `textstat`
3. **T5 Model**: The T5 paraphraser model will be downloaded automatically on first use
4. **GPU Support**: CUDA will be used automatically if available

## Troubleshooting

- If you get CORS errors, make sure the backend is running on port 8000
- If model loading fails, check that your `.pkl` files are in the correct location
- For T5 model issues, ensure you have sufficient RAM (model is ~1GB)

## API Endpoints

- `GET /` - Health check
- `POST /predict` - Analyze text for AI vs Human
- `POST /humanize` - Rewrite AI text to be more human-like