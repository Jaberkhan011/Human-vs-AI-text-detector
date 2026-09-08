"""
FastAPI backend + CLI entry point for Human vs AI Text Classification.

Pipeline:
    text
      -> tokenizer_cl.pkl
      -> padded sequence (MAX_LEN tokens)
      -> bgru_model_new.keras
      -> Human / AI probability

extract_text_features() is retained only for displaying / explainability
information on the frontend. It is NOT used as input to the BGru neural
network.

Usage:
    Run as an API server (default):
        python app.py

    Run a quick CLI smoke test against sample texts instead:
        python app.py test
"""

import pickle
import sys
from pathlib import Path
from typing import Optional, Any

import numpy as np
import pandas as pd
import tensorflow as tf

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from tensorflow.keras.preprocessing.sequence import pad_sequences

from text_feature_extractor import extract_text_features


BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "bgru_model_new.keras"
TOKENIZER_PATH = BASE_DIR / "tokenizer_cl.pkl"

# Exact preprocessing used during BGru training.
MAX_LEN = 842
PADDING = "post"
TRUNCATING = "post"

# Model was trained with a sigmoid output for binary classification.
AI_CLASS_INDEX = 1
HUMAN_CLASS_INDEX = 0


# --------------------------------------------------------------------------
# FastAPI app
# --------------------------------------------------------------------------

app = FastAPI(
    title="Human vs AI Text Classifier",
    description="Classify text with a trained Bidirectional GRU and expose extracted text features.",
    version="2.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class TextInput(BaseModel):
    text: str = Field(..., min_length=1, description="Text to classify")


class PredictionResult(BaseModel):
    input_text: str
    prediction: str
    confidence: Optional[float] = None
    features_extracted: int
    model_used: str = "bgru_model_new.keras"
    tokenizer_used: str = "tokenizer_cl.pkl"


class PredictResponse(BaseModel):
    prediction: int
    label: str
    ai_probability: float
    human_probability: float
    confidence: str
    features: dict


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    tokenizer_loaded: bool
    model_path: str
    tokenizer_path: str
    max_len: int


_model: Optional[Any] = None
_tokenizer: Optional[Any] = None


# --------------------------------------------------------------------------
# Model / tokenizer loading
# --------------------------------------------------------------------------

def load_model():
    """Load bgru_model_new.keras once and cache it."""
    global _model

    if _model is not None:
        return _model

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model file not found: {MODEL_PATH}. "
            "Place bgru_model_new.keras in the same directory as app.py."
        )

    try:
        _model = tf.keras.models.load_model(MODEL_PATH, compile=False)
        print(f"BGru model loaded: {MODEL_PATH}")
        print(f"Model input shape: {_model.input_shape}")
        print(f"Model output shape: {_model.output_shape}")
        return _model
    except Exception as e:
        raise RuntimeError(f"Failed to load bgru_model_new.keras: {e}")


def load_tokenizer():
    """Load the tokenizer used during model training, once, and cache it."""
    global _tokenizer

    if _tokenizer is not None:
        return _tokenizer

    if not TOKENIZER_PATH.exists():
        raise FileNotFoundError(
            f"Tokenizer file not found: {TOKENIZER_PATH}. "
            "Place tokenizer_cl.pkl in the same directory as app.py."
        )

    try:
        with open(TOKENIZER_PATH, "rb") as f:
            _tokenizer = pickle.load(f)

        print(f"Tokenizer loaded: {TOKENIZER_PATH}")
        print(f"Tokenizer vocabulary size: {len(_tokenizer.word_index)}")
        return _tokenizer
    except Exception as e:
        raise RuntimeError(f"Failed to load tokenizer_cl.pkl: {e}")


def get_model():
    return _model if _model is not None else load_model()


def get_tokenizer():
    return _tokenizer if _tokenizer is not None else load_tokenizer()


# --------------------------------------------------------------------------
# Inference helpers
# --------------------------------------------------------------------------

def tokenize_text(text: str, tokenizer=None) -> np.ndarray:
    """
    Convert raw text into the exact kind of integer sequence expected by
    the Embedding layer.

    IMPORTANT:
    padding/truncation must match the preprocessing used during training.
    This currently uses post-padding and post-truncation.
    """
    if tokenizer is None:
        tokenizer = get_tokenizer()

    sequences = tokenizer.texts_to_sequences([text])

    padded = pad_sequences(
        sequences,
        maxlen=MAX_LEN,
        padding=PADDING,
        truncating=TRUNCATING,
        dtype="int32",
    )

    return padded


def predict_with_bgru(text: str, model=None, tokenizer=None):
    """Run tokenizer -> padding -> BGru model."""
    if model is None:
        model = get_model()

    tokenized = tokenize_text(text, tokenizer=tokenizer)

    raw_output = model.predict(tokenized, verbose=0)
    ai_probability = float(np.asarray(raw_output).reshape(-1)[0])

    # Sigmoid output = probability of class 1 (AI).
    ai_probability = float(np.clip(ai_probability, 0.0, 1.0))
    human_probability = 1.0 - ai_probability

    prediction = 1 if ai_probability >= 0.5 else 0
    label = "AI" if prediction == AI_CLASS_INDEX else "Human"

    return prediction, label, ai_probability, human_probability


def _confidence_bucket(max_probability: float) -> str:
    if max_probability >= 0.85:
        return "High"
    if max_probability >= 0.65:
        return "Medium"
    return "Low"


def _json_safe(value):
    """Convert numpy/pandas scalar values into JSON-compatible values."""
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if pd.isna(value):
        return None
    return value


def get_frontend_features(text: str) -> dict:
    """
    Keep extract_text_features() for frontend information.

    These features are NOT fed into bgru_model_new.keras. The BGru model
    receives only the tokenizer/padded token sequence.
    """
    features_df = extract_text_features(text)

    if features_df.empty:
        return {}

    features = features_df.iloc[0].to_dict()

    # Raw text is already returned separately and is not useful as a feature.
    features.pop("Raw_Text", None)

    return {str(k): _json_safe(v) for k, v in features.items()}


def classify_text(text: str, model=None, tokenizer=None) -> dict:
    """
    High-level convenience wrapper used by both the CLI test runner and
    (indirectly, via predict_with_bgru/get_frontend_features) the API
    endpoints below. Runs the full pipeline and returns a single dict.
    """
    prediction, label, ai_probability, human_probability = predict_with_bgru(
        text, model=model, tokenizer=tokenizer
    )
    features = get_frontend_features(text)

    return {
        "text": text,
        "prediction": prediction,
        "label": label,
        "ai_probability": ai_probability,
        "human_probability": human_probability,
        "confidence": max(ai_probability, human_probability),
        "features": features,
    }


# --------------------------------------------------------------------------
# FastAPI lifecycle
# --------------------------------------------------------------------------

@app.on_event("startup")
async def startup_event():
    try:
        load_model()
        load_tokenizer()
    except Exception as e:
        # Do not prevent FastAPI from starting; /health will report the issue.
        print(f"Warning: startup loading failed: {e}")


# --------------------------------------------------------------------------
# Endpoints
# --------------------------------------------------------------------------

@app.get("/health", response_model=HealthResponse)
async def health_check():
    model_loaded = False
    tokenizer_loaded = False

    try:
        load_model()
        model_loaded = True
    except Exception:
        pass

    try:
        load_tokenizer()
        tokenizer_loaded = True
    except Exception:
        pass

    healthy = model_loaded and tokenizer_loaded

    return HealthResponse(
        status="healthy" if healthy else "model_or_tokenizer_not_loaded",
        model_loaded=model_loaded,
        tokenizer_loaded=tokenizer_loaded,
        model_path=str(MODEL_PATH),
        tokenizer_path=str(TOKENIZER_PATH),
        max_len=MAX_LEN,
    )


@app.post("/predict", response_model=PredictResponse)
async def predict_text(input_data: TextInput):
    """
    Main endpoint for the frontend.

    Pipeline:
        raw text
          -> tokenizer_cl.pkl
          -> padded integer sequence (MAX_LEN tokens)
          -> bgru_model_new.keras
          -> AI/Human probabilities

    extract_text_features() is kept separately for frontend information.
    """
    try:
        prediction, label, ai_probability, human_probability = predict_with_bgru(
            input_data.text
        )

        features = get_frontend_features(input_data.text)

        confidence = _confidence_bucket(
            max(ai_probability, human_probability)
        )

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
            status_code=500,
            detail=f"Prediction error: {e}",
        )


@app.post("/classify", response_model=PredictionResult)
async def classify_text_endpoint(input_data: TextInput):
    """Backward-compatible endpoint with the simpler response format."""
    try:
        prediction, label, ai_probability, human_probability = predict_with_bgru(
            input_data.text
        )

        features = get_frontend_features(input_data.text)

        return PredictionResult(
            input_text=(
                input_data.text[:200] + "..."
                if len(input_data.text) > 200
                else input_data.text
            ),
            prediction=label,
            confidence=max(ai_probability, human_probability),
            features_extracted=len(features),
        )

    except FileNotFoundError as e:
        raise HTTPException(status_code=503, detail=str(e))
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Classification error: {e}",
        )


@app.post("/features")
async def extract_features(input_data: TextInput):
    """
    Extract deterministic text features for frontend display/debugging.

    This endpoint does not run the BGru model.
    """
    try:
        features = get_frontend_features(input_data.text)

        return {
            "text": (
                input_data.text[:200] + "..."
                if len(input_data.text) > 200
                else input_data.text
            ),
            "features": features,
        }

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Feature extraction error: {e}",
        )


@app.get("/")
async def root():
    return {
        "message": "Human vs AI Text Classification API",
        "model": "bgru_model_new.keras",
        "tokenizer": "tokenizer_cl.pkl",
        "max_sequence_length": MAX_LEN,
        "endpoints": {
            "POST /predict": "Run BGru prediction and return frontend text features",
            "POST /classify": "Run BGru prediction with a simpler response",
            "POST /features": "Extract frontend information without prediction",
            "GET /health": "Check model/tokenizer loading",
            "GET /docs": "Interactive API documentation",
        },
    }


# --------------------------------------------------------------------------
# CLI smoke test (replaces standalone main.py)
# --------------------------------------------------------------------------

def run_cli_test():
    """Quick sanity check against a couple of sample texts, no server needed."""
    model = load_model()
    tokenizer = load_tokenizer()

    sample_texts = [
        "I love programming in Python. It's fun and powerful.",
        (
            "The implementation of machine learning algorithms requires careful "
            "consideration of computational complexity and data preprocessing "
            "strategies to ensure optimal model performance."
        ),
    ]

    print("Testing BGru Text Classification\n" + "=" * 60)

    for text in sample_texts:
        result = classify_text(text, model=model, tokenizer=tokenizer)

        print(f"\nText: {result['text'][:100]}...")
        print(f"Prediction: {result['label']}")
        print(f"AI probability: {result['ai_probability']:.4f}")
        print(f"Human probability: {result['human_probability']:.4f}")
        print(f"Confidence: {result['confidence']:.4f}")
        print(f"Frontend features extracted: {len(result['features'])}")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "test":
        # python app.py test
        run_cli_test()
    else:
        # python app.py  -> start the API server
        import uvicorn

        uvicorn.run(
            "app:app",
            host="0.0.0.0",
            port=8000,
            reload=True,
            log_level="info",
        )
