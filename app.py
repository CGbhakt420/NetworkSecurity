import sys
import os
import certifi
import pandas as pd
from dotenv import load_dotenv
load_dotenv()

from networkSecurity.utils.ml_utils.model.estimator import NetworkModel
from networkSecurity.utils.feature_extraction import URLFeatureExtractor
from networkSecurity.exceptionHandling.exception import NetworkSecurityException
from networkSecurity.logging.logger import logging
from networkSecurity.pipeline.training_pipeline import TrainingPipeline
from networkSecurity.utils.main_utils.utils import load_object
from networkSecurity.constant.training_pipeline import DATA_INGESTION_DATABASE_NAME, DATA_INGESTION_COLLECTION_NAME

import pymongo
from fastapi.middleware.cors import CORSMiddleware
from fastapi import FastAPI, File, UploadFile, Request, Form, BackgroundTasks
from fastapi.responses import Response, JSONResponse
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
import threading
from datetime import datetime

app = FastAPI(title="Network Security - Phishing Detection API")
origins = ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

templates = Jinja2Templates(directory="./templates")

def predict_from_features(features_df: pd.DataFrame):
    preprocessor = load_object("final_models/preprocessor.pkl")
    final_model = load_object("final_models/model.pkl")
    network_model = NetworkModel(preprocessor=preprocessor, model=final_model)

    pred = network_model.predict(features_df)[0]
    # pred: 0.0 -> Phishing, 1.0 -> Legitimate
    is_phishing = bool(pred == 0.0)

    phishing_prob = None
    legit_prob = None
    confidence_label = "Classified"
    if hasattr(final_model, "predict_proba"):
        transformed = preprocessor.transform(features_df)
        probs = final_model.predict_proba(transformed)[0]
        phishing_prob = round(float(probs[0]) * 100, 2)
        legit_prob = round(float(probs[1]) * 100, 2)
        confidence_label = f"{phishing_prob}% Phishing Risk" if is_phishing else f"{legit_prob}% Legitimate Confidence"

    return is_phishing, phishing_prob, legit_prob, confidence_label

class URLRequest(BaseModel):
    url: str

@app.get("/")
async def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request, "result": None, "url": ""})

@app.post("/detect")
async def detect_website(request: Request, url: str = Form(...)):
    try:
        extractor = URLFeatureExtractor(timeout=1.5)
        features_df = extractor.extract_features(url)
        is_phishing, phishing_prob, legit_prob, confidence_label = predict_from_features(features_df)
        explanations = URLFeatureExtractor.get_feature_explanations(features_df)

        result = {
            "url": url,
            "is_phishing": is_phishing,
            "phishing_prob": phishing_prob if phishing_prob is not None else (100.0 if is_phishing else 0.0),
            "legit_prob": legit_prob if legit_prob is not None else (0.0 if is_phishing else 100.0),
            "confidence_label": confidence_label,
            "explanations": explanations
        }
        return templates.TemplateResponse("index.html", {"request": request, "result": result, "url": url})
    except Exception as e:
        raise NetworkSecurityException(e, sys)

@app.post("/predict-url")
async def predict_url_api(payload: URLRequest):
    try:
        extractor = URLFeatureExtractor(timeout=1.5)
        features_df = extractor.extract_features(payload.url)
        is_phishing, phishing_prob, legit_prob, confidence_label = predict_from_features(features_df)
        explanations = URLFeatureExtractor.get_feature_explanations(features_df)

        return JSONResponse(content={
            "url": payload.url,
            "status": "Malicious / Phishing" if is_phishing else "Legitimate / Safe",
            "is_phishing": is_phishing,
            "probabilities": {
                "phishing": f"{phishing_prob}%" if phishing_prob is not None else "N/A",
                "legitimate": f"{legit_prob}%" if legit_prob is not None else "N/A"
            },
            "confidence_label": confidence_label,
            "features": {k: (None if pd.isna(v) else int(v)) for k, v in features_df.iloc[0].to_dict().items()},
            "explanations": explanations
        })
    except Exception as e:
        raise NetworkSecurityException(e, sys)

@app.post("/predict")
async def predict_route(request: Request, file: UploadFile = File(...)):
    try:
        df = pd.read_csv(file.file)
        preprocessor = load_object("final_models/preprocessor.pkl")
        final_model = load_object("final_models/model.pkl")
        network_model = NetworkModel(preprocessor=preprocessor, model=final_model)

        y_pred = network_model.predict(df)
        df['Predicted_column'] = y_pred
        os.makedirs("prediction_output", exist_ok=True)
        df.to_csv("prediction_output/output.csv")
        table_html = df.to_html(classes="table table-striped")
        return templates.TemplateResponse("table.html", {"request": request, "table": table_html})
    except Exception as e:
        raise NetworkSecurityException(e, sys)

# Background Training Job State Tracker
training_status = {
    "is_training": False,
    "last_status": "idle",
    "message": "No training job currently running. Call /train to trigger retraining.",
    "started_at": None,
    "completed_at": None,
    "metrics": None,
    "error": None
}
training_lock = threading.Lock()

def run_training_job():
    global training_status
    with training_lock:
        training_status["is_training"] = True
        training_status["last_status"] = "running"
        training_status["started_at"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        training_status["message"] = "Training pipeline execution in progress..."
        training_status["error"] = None

    try:
        logging.info("Starting background training pipeline execution...")
        train_pipeline = TrainingPipeline()
        model_trainer_artifact = train_pipeline.run_pipeline()

        with training_lock:
            training_status["is_training"] = False
            training_status["last_status"] = "success"
            training_status["completed_at"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            training_status["message"] = "Model training completed successfully! New models and preprocessors are active."
            training_status["metrics"] = {
                "best_model": getattr(model_trainer_artifact, "best_model_name", "Best Model"),
                "train_metrics": {
                    "f1_score": round(float(model_trainer_artifact.train_metric_artifact.f1_score), 4),
                    "precision": round(float(model_trainer_artifact.train_metric_artifact.precision_score), 4),
                    "recall": round(float(model_trainer_artifact.train_metric_artifact.recall_score), 4)
                },
                "test_metrics": {
                    "f1_score": round(float(model_trainer_artifact.test_metric_artifact.f1_score), 4),
                    "precision": round(float(model_trainer_artifact.test_metric_artifact.precision_score), 4),
                    "recall": round(float(model_trainer_artifact.test_metric_artifact.recall_score), 4)
                }
            }
        logging.info("Background training pipeline completed successfully.")
    except Exception as e:
        error_msg = str(e)
        logging.error(f"Background training pipeline failed: {error_msg}")
        with training_lock:
            training_status["is_training"] = False
            training_status["last_status"] = "failed"
            training_status["completed_at"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            training_status["message"] = f"Training failed: {error_msg}"
            training_status["error"] = error_msg

@app.get("/train")
async def train_route(background_tasks: BackgroundTasks):
    global training_status
    if training_status["is_training"]:
        return JSONResponse(
            status_code=202,
            content={
                "status": "already_running",
                "message": "Training pipeline is already running in background. Check /train-status for progress.",
                "started_at": training_status["started_at"],
                "status_endpoint": "/train-status"
            }
        )

    background_tasks.add_task(run_training_job)
    return JSONResponse(
        status_code=200,
        content={
            "status": "started",
            "message": "Model training pipeline triggered in background! It will train and update artifacts without timing out.",
            "status_endpoint": "/train-status"
        }
    )

@app.get("/train-status")
async def train_status_route():
    return JSONResponse(content=training_status)
