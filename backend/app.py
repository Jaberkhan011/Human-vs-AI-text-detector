"""
FastAPI backend + CLI entry point for Human vs AI Text Classification.

Pipeline:
    text
      -> TF-IDF vectorizer
      -> Logistic Regression
      -> Human / AI probability

Model:
    ai_vs_human_model_LR50k.pkl

Usage:
    Run API:
        python app.py

    CLI test:
        python app.py test
"""

import pickle
import sys
from pathlib import Path
from typing import Optional, Any

import numpy as np
import pandas as pd

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from text_feature_extractor import extract_text_features


# --------------------------------------------------------------------------
# Paths
# --------------------------------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent

MODEL_PATH = BASE_DIR / "ai_vs_human_model_LR50k.pkl"


# --------------------------------------------------------------------------
# Classification configuration
# --------------------------------------------------------------------------

# IMPORTANT:
# Change these only if your training labels are reversed.
#
# 0 = Human
# 1 = AI
HUMAN_CLASS = 0
AI_CLASS = 1


# --------------------------------------------------------------------------
# FastAPI app
# --------------------------------------------------------------------------

app = FastAPI(
    title="Human vs AI Text Classifier",
    description=(
        "Human vs AI text classification using "
        "TF-IDF and Logistic Regression."
    ),
    version="3.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# --------------------------------------------------------------------------
# Pydantic models
# --------------------------------------------------------------------------

class TextInput(BaseModel):
    text: str = Field(
        ...,
        min_length=1,
        description="Text to classify",
    )


class PredictResponse(BaseModel):
    prediction: int
    label: str
    ai_probability: float
    human_probability: float
    confidence: str
    features: dict


class PredictionResult(BaseModel):
    input_text: str
    prediction: str
    confidence: Optional[float] = None
    features_extracted: int
    model_used: str = "ai_vs_human_model_LR50k.pkl"


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    model_path: str


# --------------------------------------------------------------------------
# Global model
# --------------------------------------------------------------------------

_model: Optional[Any] = None


# --------------------------------------------------------------------------
# Model loading
# --------------------------------------------------------------------------

def load_model():
    """
    Load the TF-IDF + Logistic Regression model once.

    The pickle can contain either:

        1. A complete sklearn Pipeline

    or:

        2. A dictionary containing vectorizer + classifier

    or:

        3. A classifier that already expects TF-IDF vectors.
    """

    global _model

    if _model is not None:
        return _model

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model file not found: {MODEL_PATH}"
        )

    try:
        with open(MODEL_PATH, "rb") as f:
            _model = pickle.load(f)

        print(f"Model loaded: {MODEL_PATH}")
        print(f"Model type: {type(_model)}")

        return _model

    except Exception as e:
        raise RuntimeError(
            f"Failed to load {MODEL_PATH}: {e}"
        )


def get_model():
    return _model if _model is not None else load_model()


# --------------------------------------------------------------------------
# Prediction
# --------------------------------------------------------------------------

def predict_text(text: str, model=None):
    """
    Run TF-IDF + Logistic Regression prediction.

    Supports:
        sklearn Pipeline
        vectorizer/classifier dictionary
        classifier with pre-vectorized input
    """

    if model is None:
        model = get_model()

    # --------------------------------------------------------------
    # Case 1:
    # Complete sklearn Pipeline
    # --------------------------------------------------------------

    if hasattr(model, "predict_proba") and hasattr(model, "predict"):

        # Pipeline can directly accept raw text.
        try:
            prediction_raw = model.predict([text])
            probabilities = model.predict_proba([text])

            prediction = int(prediction_raw[0])

            probabilities = np.asarray(probabilities)[0]

            # Find probability belonging to class 1.
            if hasattr(model, "classes_"):
                classes = list(model.classes_)

                if AI_CLASS in classes:
                    ai_index = classes.index(AI_CLASS)
                    ai_probability = float(probabilities[ai_index])
                else:
                    ai_probability = float(probabilities[-1])
            else:
                ai_probability = float(probabilities[-1])

            human_probability = 1.0 - ai_probability

            label = (
                "AI"
                if prediction == AI_CLASS
                else "Human"
            )

            return (
                prediction,
                label,
                ai_probability,
                human_probability,
            )

        except Exception:
            pass

    # --------------------------------------------------------------
    # Case 2:
    # Dictionary containing vectorizer + classifier
    # --------------------------------------------------------------

    if isinstance(model, dict):

        vectorizer = (
            model.get("vectorizer")
            or model.get("tfidf")
            or model.get("tfidf_vectorizer")
        )

        classifier = (
            model.get("model")
            or model.get("classifier")
            or model.get("lr")
            or model.get("logistic_regression")
        )

        if vectorizer is None or classifier is None:
            raise RuntimeError(
                "Dictionary model does not contain a recognizable "
                "TF-IDF vectorizer and classifier."
            )

        X = vectorizer.transform([text])

        prediction = int(classifier.predict(X)[0])

        probabilities = classifier.predict_proba(X)[0]

        if hasattr(classifier, "classes_"):
            classes = list(classifier.classes_)

            if AI_CLASS in classes:
                ai_index = classes.index(AI_CLASS)
                ai_probability = float(probabilities[ai_index])
            else:
                ai_probability = float(probabilities[-1])
        else:
            ai_probability = float(probabilities[-1])

        human_probability = 1.0 - ai_probability

        label = (
            "AI"
            if prediction == AI_CLASS
            else "Human"
        )

        return (
            prediction,
            label,
            ai_probability,
            human_probability,
        )

    raise RuntimeError(
        "Unsupported model format. The pickle must contain either "
        "an sklearn Pipeline or a vectorizer/classifier pair."
    )


# --------------------------------------------------------------------------
# Confidence
# --------------------------------------------------------------------------

def confidence_bucket(max_probability: float) -> str:

    if max_probability >= 0.85:
        return "High"

    if max_probability >= 0.65:
        return "Medium"

    return "Low"


# --------------------------------------------------------------------------
# JSON conversion
# --------------------------------------------------------------------------

def json_safe(value):

    if isinstance(value, np.integer):
        return int(value)

    if isinstance(value, np.floating):
        return float(value)

    if isinstance(value, np.bool_):
        return bool(value)

    if pd.isna(value):
        return None

    return value


# --------------------------------------------------------------------------
# Frontend features
# --------------------------------------------------------------------------

def get_frontend_features(text: str) -> dict:
    """
    Extract deterministic text features for frontend display.

    IMPORTANT:

    These features are NOT used by the Logistic Regression model
    unless your original training pipeline explicitly used them.
    """

    features_df = extract_text_features(text)

    if features_df.empty:
        return {}

    features = features_df.iloc[0].to_dict()

    features.pop("Raw_Text", None)

    return {
        str(k): json_safe(v)
        for k, v in features.items()
    }


# --------------------------------------------------------------------------
# High-level classification
# --------------------------------------------------------------------------

def classify_text(
    text: str,
    model=None,
):

    (
        prediction,
        label,
        ai_probability,
        human_probability,
    ) = predict_text(
        text,
        model=model,
    )

    features = get_frontend_features(text)

    return {
        "text": text,
        "prediction": prediction,
        "label": label,
        "ai_probability": ai_probability,
        "human_probability": human_probability,
        "confidence": max(
            ai_probability,
            human_probability,
        ),
        "features": features,
    }


# --------------------------------------------------------------------------
# Startup
# --------------------------------------------------------------------------

@app.on_event("startup")
async def startup_event():

    try:
        load_model()

    except Exception as e:
        print(
            f"Warning: model loading failed: {e}"
        )


# --------------------------------------------------------------------------
# Health
# --------------------------------------------------------------------------

@app.get(
    "/health",
    response_model=HealthResponse,
)
async def health_check():

    model_loaded = False

    try:
        load_model()
        model_loaded = True

    except Exception:
        pass

    return HealthResponse(
        status=(
            "healthy"
            if model_loaded
            else "model_not_loaded"
        ),
        model_loaded=model_loaded,
        model_path=str(MODEL_PATH),
    )


# --------------------------------------------------------------------------
# Main prediction endpoint
# --------------------------------------------------------------------------

@app.post(
    "/predict",
    response_model=PredictResponse,
)
async def predict_text_endpoint(
    input_data: TextInput,
):

    try:

        (
            prediction,
            label,
            ai_probability,
            human_probability,
        ) = predict_text(
            input_data.text
        )

        features = get_frontend_features(
            input_data.text
        )

        confidence = confidence_bucket(
            max(
                ai_probability,
                human_probability,
            )
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

        raise HTTPException(
            status_code=503,
            detail=str(e),
        )

    except Exception as e:

        raise HTTPException(
            status_code=500,
            detail=f"Prediction error: {e}",
        )


# --------------------------------------------------------------------------
# Backward-compatible endpoint
# --------------------------------------------------------------------------

@app.post(
    "/classify",
    response_model=PredictionResult,
)
async def classify_text_endpoint(
    input_data: TextInput,
):

    try:

        (
            prediction,
            label,
            ai_probability,
            human_probability,
        ) = predict_text(
            input_data.text
        )

        features = get_frontend_features(
            input_data.text
        )

        return PredictionResult(
            input_text=(
                input_data.text[:200] + "..."
                if len(input_data.text) > 200
                else input_data.text
            ),
            prediction=label,
            confidence=max(
                ai_probability,
                human_probability,
            ),
            features_extracted=len(features),
        )

    except FileNotFoundError as e:

        raise HTTPException(
            status_code=503,
            detail=str(e),
        )

    except Exception as e:

        raise HTTPException(
            status_code=500,
            detail=f"Classification error: {e}",
        )


# --------------------------------------------------------------------------
# Features endpoint
# --------------------------------------------------------------------------

@app.post("/features")
async def extract_features_endpoint(
    input_data: TextInput,
):

    try:

        features = get_frontend_features(
            input_data.text
        )

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


# --------------------------------------------------------------------------
# Root endpoint
# --------------------------------------------------------------------------

@app.get("/")
async def root():

    return {
        "message": "Human vs AI Text Classification API",

        "model": (
            "ai_vs_human_model_LR50k.pkl"
        ),

        "classifier": (
            "TF-IDF + Logistic Regression"
        ),

        "endpoints": {

            "POST /predict":
                "Run Logistic Regression prediction "
                "and return frontend features",

            "POST /classify":
                "Run prediction with simpler response",

            "POST /features":
                "Extract frontend features without prediction",

            "GET /health":
                "Check model loading",

            "GET /docs":
                "Interactive API documentation",
        },
    }


# --------------------------------------------------------------------------
# CLI smoke test
# --------------------------------------------------------------------------

def run_cli_test():

    model = load_model()

    sample_texts = [

        "I love programming in Python. "
        "It's fun and powerful.",

        (
            "The implementation of machine learning "
            "algorithms requires careful consideration "
            "of computational complexity and data "
            "preprocessing strategies to ensure "
            "optimal model performance."
        ),

    ]

    print(
        "Testing TF-IDF + Logistic Regression"
        "\n" + "=" * 60
    )

    for text in sample_texts:

        result = classify_text(
            text,
            model=model,
        )

        print(
            f"\nText: {result['text'][:100]}..."
        )

        print(
            f"Prediction: {result['label']}"
        )

        print(
            f"AI probability: "
            f"{result['ai_probability']:.4f}"
        )

        print(
            f"Human probability: "
            f"{result['human_probability']:.4f}"
        )

        print(
            f"Confidence: "
            f"{result['confidence']:.4f}"
        )

        print(
            "Frontend features extracted: "
            f"{len(result['features'])}"
        )


# --------------------------------------------------------------------------
# Entry point
# --------------------------------------------------------------------------

if __name__ == "__main__":

    if (
        len(sys.argv) > 1
        and sys.argv[1] == "test"
    ):

        # python app.py test
        run_cli_test()

    else:

        # python app.py
        import uvicorn

        uvicorn.run(
            "app:app",
            host="0.0.0.0",
            port=8000,
            reload=True,
            log_level="info",
        )
