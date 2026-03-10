import sys
import os
import json
import time
import asyncio
import pandas as pd
from contextlib import asynccontextmanager
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
import joblib
import random

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.config import MODELS_DIR, TEST_DATA_PATH
from utils.helpers import get_logger

logger = get_logger(__name__)

# Global variables to hold loaded models and test data
models = {}
test_df = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    global models, test_df
    logger.info("Starting up FastAPI Backend for Quantum IDS")
    
    # Try to load Classical Models
    try:
        models['SVM'] = joblib.load(os.path.join(MODELS_DIR, "svm_model.pkl"))
        logger.info("Successfully loaded SVM Model")
    except Exception as e:
        logger.warning(f"Could not load SVM: {e}")
        
    try:
        models['RandomForest'] = joblib.load(os.path.join(MODELS_DIR, "random_forest_model.pkl"))
        logger.info("Successfully loaded Random Forest Model")
    except Exception as e:
        logger.warning(f"Could not load Random Forest: {e}")
        
    try:
        models['QSVM'] = joblib.load(os.path.join(MODELS_DIR, "qsvm_model.pkl"))
        logger.info("Successfully loaded QSVM Model")
    except Exception as e:
        logger.warning(f"Could not load QSVM: {e}")
        
    try:
        models['VQC'] = joblib.load(os.path.join(MODELS_DIR, "vqc_model.pkl"))
        logger.info("Successfully loaded VQC Model")
    except Exception as e:
        logger.warning(f"Could not load VQC: {e}")

    # Load test dataset to act as "live traffic" pool
    try:
        test_df = pd.read_csv(TEST_DATA_PATH)
        logger.info(f"Loaded {len(test_df)} rows of test data for simulation")
    except Exception as e:
        logger.error(f"Could not load test dataset: {e}")

    yield
    
    logger.info("Shutting down FastAPI Backend")
    models.clear()

app = FastAPI(title="Quantum IDS Real-Time API", lifespan=lifespan)

# Allow CORS for React frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def read_root():
    return {"status": "Online", "message": "Quantum IDS Real-Time Engine is running."}

@app.websocket("/ws/traffic")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    logger.info("New WebSocket connection established for Live Traffic.")
    
    if test_df is None or len(models) == 0:
        await websocket.send_text(json.dumps({"error": "System not fully initialized (models/data missing)."}))
        await websocket.close()
        return

    # Simulate live traffic by yielding random rows from the test set
    try:
        while True:
            # Pick a random row
            sample = test_df.sample(1)
            X = sample.drop("label", axis=1)
            true_y = int(sample["label"].values[0])
            
            # Predict with Random Forest for speed of simulation, if available
            prediction_result = -1
            confidence = 0.0
            used_model = "None"
            
            start_t = time.time()
            if "RandomForest" in models:
                used_model = "RandomForest"
                pred = models['RandomForest'].predict(X)
                prediction_result = int(pred[0])
                probs = models['RandomForest'].predict_proba(X)
                confidence = float(probs[0].max() * 100)
            elif "SVM" in models:
                used_model = "SVM"
                pred = models['SVM'].predict(X)
                prediction_result = int(pred[0])
                try:
                    probs = models['SVM'].predict_proba(X)
                    confidence = float(probs[0].max() * 100)
                except:
                    confidence = 85.0 # fallback if probability=False
            
            latency = (time.time() - start_t) * 1000 # ms
            
            # Additional Quantum metrics simulation for the dashboard comparison
            q_latency = latency * random.uniform(15, 30) # Simulated higher latency for quantum on classical hw
            
            packet_data = {
                "timestamp": pd.Timestamp.now().isoformat(),
                "src_bytes": int(sample["src_bytes"].values[0]) if "src_bytes" in sample else random.randint(0, 1000),
                "dst_bytes": int(sample["dst_bytes"].values[0]) if "dst_bytes" in sample else random.randint(0, 1000),
                "protocol_type": int(sample["protocol_type"].values[0]) if "protocol_type" in sample else 6,
                "true_label": true_y, # 0 for Benign, 1 for Attack
                "predicted": prediction_result,
                "confidence_percent": round(confidence, 2),
                "model_used": used_model,
                "latency_ms": round(latency, 2),
                "qsvm_simulated_latency_ms": round(q_latency, 2)
            }
            
            await websocket.send_text(json.dumps(packet_data))
            
            # Wait 1-3 seconds before sending the next packet
            await asyncio.sleep(random.uniform(1.0, 3.0))
            
    except WebSocketDisconnect:
        logger.info("WebSocket connection closed by client.")
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
