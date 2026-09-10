"""
FastAPI backend for Human vs AI Text Classification

Flow:
1. Accept text input from user
2. Extract features using text_feature_extractor.py -> produces a DataFrame
   used ONLY for the frontend's stats display (word count, char count, etc.)
   -- it plays no role in the model's prediction.
3. Load ai_vs_human_model_LR50k.pkl -- a sklearn Pipeline:
     Pipeline([
         ('preprocessor', ColumnTransformer([('text', TfidfVectorizer(), 'text')])),
         ('classifier', LogisticRegression()),
     ])
   The ColumnTransformer selects a column literally named "text", so the
   pipeline must be called with a pandas DataFrame shaped like
   pd.DataFrame({'text': [input_text]}) -- not a bare string, and not a
   list of strings.
4. Return classification result

IMPORTANT: this model must be loaded with joblib.load(), not pickle.load().
scikit-learn Pipelines with large numpy arrays (TF-IDF vocab, coefficients)
are commonly saved via joblib.dump(), which stores big arrays outside the
main pickle opcode stream. Calling plain pickle.load() on such a file causes
the unpickler to desync partway through and fail with
`UnpicklingError: STACK_GLOBAL requires str` -- this is NOT a scikit-learn
version mismatch, it's the wrong loader function.
"""

import warnings
from pathlib import Path
from typing import Optional

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from text_feature_extractor import extract_text_features


# Initialize FastAPI app
app = FastAPI(
    title="Human vs AI Text Classifier",
    description="Classify text as human-written or AI-generated",
    version="1.0.0"
)

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Pydantic models
# ---------------------------------------------------------------------------

class TextInput(BaseModel):
    """Request model for text classification"""
    text: str = Field(..., min_length=1, description="Text to classify")
    example: Optional[str] = Field(
        None, description="Example text for testing"
    )


class PredictionResult(BaseModel):
    """Response model for classification result"""
    input_text: str
    prediction: str
    confidence: Optional[float] = None
    features_extracted: Optional[int] = None
    model_used: str = "ai_vs_human_model_LR50k.pkl"


class HealthResponse(BaseModel):
    """Health check response"""
    status: str
    model_loaded: bool
    model_path: str


class PredictRequest(BaseModel):
    """Request model for the /predict endpoint used by the TextGuard frontend"""
    text: str = Field(..., min_length=1, description="Text to classify")


class PredictResponse(BaseModel):
    """Response model matching the TextGuard frontend's expected contract"""
    prediction: int
    label: str
    ai_probability: float
    human_probability: float
    confidence: str
    features: dict


def _confidence_bucket(max_probability: float) -> str:
    """Map a top-class probability to a coarse confidence label"""
    if max_probability >= 0.85:
        return "High"
    if max_probability >= 0.65:
        return "Medium"
    return "Low"


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

_model = None
_model_path = None


def load_model():
    """Load the pre-trained sklearn Pipeline from pickle file"""
    global _model, _model_path

    if _model is not None:
        return _model

    current_dir = Path(__file__).parent
    model_path = current_dir / "ai_vs_human_model_LR50k.pkl"

    if not model_path.exists():
        raise FileNotFoundError(
            f"Model file not found at {model_path}. "
            "Please ensure ai_vs_human_model_LR50k.pkl is in the project directory."
        )

    try:
        with open(model_path, "rb") as f:
            _model = joblib.load(f)
            _model_path = str(model_path)
        print(f"Model loaded successfully from {model_path}")
        return _model
    except Exception as e:
        # Most likely cause: scikit-learn version mismatch between the
        # environment the model was pickled in and this one.
        raise RuntimeError(
            f"Failed to load model: {str(e)}. "
            "If this is an UnpicklingError, check that the deployed "
            "scikit-learn version matches the one used to train/save the model."
        )


def get_model():
    """Get the loaded model, loading it if necessary"""
    if _model is None:
        load_model()
    return _model


def _predict_with_model(model, text: str):
    """
    Run the sklearn Pipeline on a single text.

    The fitted pipeline expects a pandas DataFrame containing
    a column named exactly 'text'.
    """

    # IMPORTANT: [text] creates a one-row DataFrame
    textdf = pd.DataFrame({
        "text": [text]
    })

    # Pass the DataFrame to the Pipeline
    prediction = int(model.predict(textdf)[0])

    try:
        probabilities = model.predict_proba(textdf)[0]

        # Assuming:
        # class 0 = Human
        # class 1 = AI
        human_probability = float(probabilities[0])
        ai_probability = float(probabilities[1])

    except (AttributeError, IndexError):
        ai_probability = 1.0 if prediction == 1 else 0.0
        human_probability = 1.0 - ai_probability

    return prediction, human_probability, ai_probability


@app.on_event("startup")
async def startup_event():
    """Load model on application startup"""
    try:
        load_model()
    except Exception as e:
        print(f"Warning: Could not load model on startup: {e}")


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    try:
        model = get_model()
        model_loaded = model is not None
    except Exception:
        model_loaded = False

    return HealthResponse(
        status="healthy" if model_loaded else "model_not_loaded",
        model_loaded=model_loaded,
        model_path=_model_path or "not_loaded"
    )


@app.post("/classify", response_model=PredictionResult)
async def classify_text(input_data: TextInput):
    """
    Classify text as human-written or AI-generated

    Parameters:
    - text: The text to classify

    Returns:
    - Prediction result with confidence score
    """
    try:
        features_df = extract_text_features(
            input_data.text)  # for feature count only
        model = get_model()

        prediction, human_probability, ai_probability = _predict_with_model(
            model, input_data.text
        )
        confidence = max(human_probability, ai_probability)

        label_map = {0: "Human", 1: "AI"}
        predicted_label = label_map.get(prediction, str(prediction))

        return PredictionResult(
            input_text=input_data.text[:200] + "..."
            if len(input_data.text) > 200 else input_data.text,
            prediction=predicted_label,
            confidence=confidence,
            features_extracted=len(features_df.columns),
            model_used="ai_vs_human_model_LR50k.pkl"
        )

    except FileNotFoundError as e:
        raise HTTPException(status_code=503, detail=str(e))
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Classification error: {str(e)}")


@app.post("/predict", response_model=PredictResponse)
async def predict_text(input_data: PredictRequest):
    """
    Classify text and return AI/human probabilities plus the deterministic
    text features, in the shape expected by the TextGuard frontend.
    """
    try:
        features_df = extract_text_features(
            input_data.text)  # frontend display only
        model = get_model()

        prediction, human_probability, ai_probability = _predict_with_model(
            model, input_data.text
        )

        confidence = _confidence_bucket(max(ai_probability, human_probability))
        label_map = {0: "Human", 1: "AI"}
        label = label_map.get(prediction, str(prediction))

        # Raw_Text (if present) is dropped -- the frontend only needs the
        # numeric/deterministic display features, not the raw string back.
        features = features_df.to_dict(orient="records")[0]
        features.pop("Raw_Text", None)

        return PredictResponse(
            prediction=prediction,
            label=label,
            ai_probability=ai_probability,
            human_probability=human_probability,
            confidence=confidence,
            features=features,
        )

    except FileNotFoundError as e:
        raise HTTPException(status_code=503, detail=str(e))
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Prediction error: {str(e)}")


@app.post("/features")
async def extract_features(input_data: TextInput):
    """
    Extract features from text without making a prediction

    Useful for debugging and understanding feature extraction
    """
    try:
        features_df = extract_text_features(input_data.text)
        return {
            "text": input_data.text[:200] + "..." if len(input_data.text) > 200 else input_data.text,
            "features": features_df.to_dict(orient="records")[0]
        }
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Feature extraction error: {str(e)}")


@app.get("/")
async def root():
    """Root endpoint with API documentation"""
    return {
        "message": "Human vs AI Text Classification API",
        "endpoints": {
            "POST /predict": "Classify text and return AI/human probabilities (used by the TextGuard frontend)",
            "POST /classify": "Classify text as human or AI",
            "POST /features": "Extract features from text",
            "GET /health": "Health check",
            "GET /docs": "Interactive API documentation"
        }
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "app:app",
        host="0.0.0.0",
        port=2200,
        reload=True,
        log_level="info"
    )
