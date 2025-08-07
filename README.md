# AI vs Human Text Detector + Humanizer

A full-stack web application that detects whether text is AI-generated or human-written, and provides humanization capabilities for AI-detected content.

## Features

- **AI Detection**: Uses a trained RandomForestClassifier with TF-IDF and linguistic features
- **Confidence Scoring**: Provides confidence percentages for predictions
- **Text Humanization**: Rewrites AI-generated text using T5 paraphraser model
- **Modern UI**: Responsive React frontend with Tailwind CSS
- **Fast API**: High-performance Python backend with FastAPI

## Architecture

```
├── src/                    # React Frontend
│   ├── App.tsx            # Main application component
│   └── ...
├── backend/               # FastAPI Backend
│   ├── main.py           # FastAPI application
│   ├── rewrite.py        # T5 text rewriting module
│   ├── model/            # ML model files
│   │   ├── ai_detector_tfidf.pkl
│   │   └── vectorizer.pkl
│   └── requirements.txt  # Python dependencies
└── README.md
```

## Setup Instructions

### Prerequisites

- Node.js 18+ and npm
- Python 3.8+
- Your trained model files:
  - `ai_detector_tfidf.pkl` (RandomForestClassifier)
  - `vectorizer.pkl` (TfidfVectorizer)

### Frontend Setup

The frontend is already configured. Start the development server:

```bash
npm run dev
```

The frontend will be available at `http://localhost:5173`

### Backend Setup

1. Navigate to the backend directory:
```bash
cd backend
```

2. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. **Add your model files**: Place your trained models in the `backend/model/` directory:
   - `ai_detector_tfidf.pkl`
   - `vectorizer.pkl`

5. Start the FastAPI server:
```bash
python main.py
```

Or using uvicorn directly:
```bash
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

The backend API will be available at `http://localhost:8000`

## API Endpoints

### POST /predict
Analyze text to determine if it's AI-generated or human-written.

**Request:**
```json
{
  "text": "Your text to analyze here..."
}
```

**Response:**
```json
{
  "prediction": "AI",  // or "Human"
  "confidence": 85.7
}
```

### POST /humanize
Rewrite AI-generated text to make it more human-like.

**Request:**
```json
{
  "text": "AI-generated text to humanize..."
}
```

**Response:**
```json
{
  "rewritten_text": "Humanized version of the text..."
}
```

## Model Requirements

This application expects your trained models to have the following characteristics:

1. **RandomForestClassifier**: Binary classifier where:
   - 0 = Human-written text
   - 1 = AI-generated text

2. **TfidfVectorizer**: Trained on the same dataset used for the classifier

3. **Feature Engineering**: The system includes linguistic feature extraction using:
   - `language_tool_python` for grammar analysis
   - `textstat` for readability metrics

## Usage

1. Start both frontend and backend servers
2. Open `http://localhost:5173` in your browser
3. Enter text in the input field
4. Click "Detect AI vs Human" to analyze the text
5. If the text is detected as AI-generated, use "Humanize Text" to rewrite it

## Development Notes

- The T5 model (`ramsrigouthamg/t5_paraphraser`) will be downloaded automatically on first use
- GPU acceleration is used if available (CUDA)
- The application includes comprehensive error handling and fallback mechanisms
- CORS is configured to allow frontend-backend communication

## Customization

To adapt this application for your specific models:

1. Modify the feature extraction in `backend/main.py` to match your training pipeline
2. Adjust the prediction logic based on your model's output format
3. Update the model file paths if using different names
4. Customize the T5 model in `rewrite.py` if using a different paraphraser

## Production Deployment

For production deployment:

1. Use a production WSGI server like Gunicorn for the backend
2. Build the frontend with `npm run build`
3. Serve the frontend through a web server like Nginx
4. Configure proper CORS policies
5. Set up environment variables for model paths and configurations