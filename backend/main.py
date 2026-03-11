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
import threading
from queue import Queue

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.config import MODELS_DIR, TEST_DATA_PATH
from utils.helpers import get_logger
try:
    from utils.sniffer import PacketSniffer
except ImportError:
    PacketSniffer = None

logger = get_logger(__name__)

# Global variables
models = {}
test_df = None
live_packet_queue = Queue()
sniff_mode = "simulation" # "simulation" or "live"
sniffer = None

# Real-time Accuracy Stats
session_stats = {
    "classical_correct": 0,
    "classical_total": 0,
    "quantum_correct": 0,
    "quantum_total": 0,
    "hybrid_correct": 0,
    "hybrid_total": 0,
    "quantum_catches": 0
}

def packet_callback(pkt_data):
    live_packet_queue.put(pkt_data)

@asynccontextmanager
async def lifespan(app: FastAPI):
    global models, test_df, sniffer
    logger.info("Starting up FastAPI Backend for Quantum IDS")
    
    # Load Unified Models (Classical Brain)
    path = os.path.join(MODELS_DIR, "unified_rf_model.pkl")
    if os.path.exists(path):
        try:
            models["RandomForest"] = joblib.load(path)
            pre_dir = os.path.join(MODELS_DIR, "preprocessing")
            models["scaler_unified"] = joblib.load(os.path.join(pre_dir, "scaler_unified.pkl"))
            models["pca_unified"] = joblib.load(os.path.join(pre_dir, "pca_unified.pkl"))
            logger.info("Loaded Optimized Tuned RandomForest & Preprocessing")
        except Exception as e:
            logger.warning(f"Failed to load Tuned Model: {e}")

    # Load Quantum Specialist (QSVC) for Rare Attacks
    q_path = os.path.join(MODELS_DIR, "qsvc_specialist.pkl")
    if os.path.exists(q_path):
        try:
            models["QSVM_Specialist"] = joblib.load(q_path)
            q_pre_dir = "models/saved/preprocessing/quantum"
            models["q_scaler"] = joblib.load(os.path.join(q_pre_dir, "scaler_specialist.pkl"))
            models["q_pca"] = joblib.load(os.path.join(q_pre_dir, "pca_specialist.pkl"))
            logger.info("Loaded Optimized Tuned Quantum Specialist (QSVC)")
        except Exception as e:
            logger.warning(f"Failed to load Tuned Quantum Specialist: {e}")

    # Load test dataset
    try:
        test_df = pd.read_csv(TEST_DATA_PATH)
        logger.info(f"Loaded {len(test_df)} rows for simulation")
    except Exception as e:
        logger.error(f"Could not load test dataset: {e}")

    # Initialize Sniffer but don't start yet
    if PacketSniffer:
        try:
            sniffer = PacketSniffer(callback=packet_callback)
            logger.info("Real-time Sniffer initialized.")
        except Exception as e:
            logger.warning(f"Sniffer initialization failed: {e}")

    yield
    
    if sniffer: sniffer.stop()
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

@app.get("/toggle-mode")
async def toggle_mode(mode: str):
    global sniff_mode, sniffer
    if mode not in ["simulation", "live"]:
        return {"error": "Invalid mode"}
    
    sniff_mode = mode
    if mode == "live" and sniffer:
        sniffer.start()
        logger.info("Switching to LIVE SNIFFING mode.")
    elif mode == "simulation" and sniffer:
        sniffer.stop()
        logger.info("Switching to SIMULATION mode.")
    
    return {"status": "Success", "current_mode": sniff_mode}

@app.websocket("/ws/traffic")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    logger.info(f"WebSocket established. Initial Mode: {sniff_mode}")
    
    try:
        while True:
            packet_data = {}
            X = None
            
            if sniff_mode == "live":
                if not live_packet_queue.empty():
                    packet_data = live_packet_queue.get()
                    X = packet_data.pop("raw_features")
                else:
                    await asyncio.sleep(0.1)
                    continue
            else:
                # Simulation Mode
                if test_df is None: break
                sample = test_df.sample(1)
                X = sample.drop("label", axis=1)
                true_y = int(sample["label"].values[0])
                packet_data = {
                    "timestamp": pd.Timestamp.now().isoformat(),
                    "src_bytes": int(sample["src_bytes"].values[0]) if "src_bytes" in sample.columns else 0,
                    "dst_bytes": int(sample["dst_bytes"].values[0]) if "dst_bytes" in sample.columns else 0,
                    "protocol_type": str(sample["protocol_type"].values[0]) if "protocol_type" in sample.columns else "tcp",
                    "true_label": true_y
                }

            # Inference
            prediction_result = -1
            confidence = 0.0
            used_model = "None"
            start_t = time.time()
            
            if "RandomForest" in models:
                used_model = "RandomForest"
                
                # 1. Classical Preprocessing & Prediction
                # If simulation, X might need scaling/pca
                X_scaled = models["scaler_unified"].transform(X)
                X_pca = models["pca_unified"].transform(X_scaled)
                rf_pred = int(models['RandomForest'].predict(X_pca)[0])
                prediction_result = rf_pred
                confidence = float(models['RandomForest'].predict_proba(X_pca)[0].max() * 100)
                
                # Update Session Classical Stats (In simulation we have true_y)
                if sniff_mode == "simulation" and "true_label" in packet_data:
                    session_stats["classical_total"] += 1
                    if rf_pred == packet_data["true_label"]:
                        session_stats["classical_correct"] += 1

                # 2. Quantum Specialist Audit (Hybrid Logic)
                is_quantum_catch = False
                if "QSVM_Specialist" in models:
                    try:
                        # Only audit if RF is uncertain or it's a high-priority packet
                        # For this dashboard implementation, we audit all for visibility
                        q_feat = X[['duration', 'src_bytes', 'dst_bytes', 'count', 'srv_count']]
                        q_scaled = models["q_scaler"].transform(q_feat)
                        q_pca = models["q_pca"].transform(q_scaled)
                        q_pred = int(models["QSVM_Specialist"].predict(q_pca)[0])
                        
                        # Is it a rare attack? (Based on ground truth in simulation)
                        is_rare = False
                        if sniff_mode == "simulation" and "true_label" in packet_data:
                            # Get the actual label string to check for rarity
                            actual_label = str(sample["label"].values[0])
                            is_rare = any(r in actual_label for r in ["Infiltration", "Sql Injection", "Heartbleed"])
                            
                            if is_rare:
                                session_stats["quantum_total"] += 1
                                if q_pred == 1:
                                    session_stats["quantum_correct"] += 1
                                
                        # HYBRID DECISION: Priority to Quantum Specialist for detected threats
                        if q_pred == 1:
                            prediction_result = 1
                            used_model = "Hybrid (QF+RF)"
                            if rf_pred == 0:
                                is_quantum_catch = True
                                session_stats["quantum_catches"] += 1
                                logger.info("QUANTUM CATCH! Specialist detected a threat missed by RF.")
                    except Exception as qe:
                        logger.debug(f"Quantum Audit Skipped: {qe}")
                
                # Update Final Hybrid Stats
                if sniff_mode == "simulation" and "true_label" in packet_data:
                    session_stats["hybrid_total"] += 1
                    if prediction_result == packet_data["true_label"]:
                        session_stats["hybrid_correct"] += 1
            elif "SVM" in models:
                used_model = "SVM"
                pred = models['SVM'].predict(X)
                prediction_result = int(pred[0])
                try:
                    probs = models['SVM'].predict_proba(X)
                    confidence = float(probs[0].max() * 100)
                except:
                    confidence = 85.0 # fallback if probability=False
            
            latency = (time.time() - start_t) * 1000
            
            # Calculate Session Metrics for JSON
            c_acc = (session_stats["classical_correct"] / session_stats["classical_total"] * 100) if session_stats["classical_total"] > 0 else 0
            q_acc = (session_stats["quantum_correct"] / session_stats["quantum_total"] * 100) if session_stats["quantum_total"] > 0 else 0
            h_acc = (session_stats["hybrid_correct"] / session_stats["hybrid_total"] * 100) if session_stats["hybrid_total"] > 0 else 0

            packet_data.update({
                "predicted": prediction_result,
                "confidence_percent": round(confidence, 2),
                "model_used": used_model,
                "latency_ms": round(latency, 2),
                "mode": sniff_mode,
                "is_quantum_catch": is_quantum_catch,
                "session_classical_acc": round(c_acc, 2),
                "session_quantum_acc": round(q_acc, 2),
                "session_hybrid_acc": round(h_acc, 2),
                "quantum_catch_count": session_stats["quantum_catches"]
            })
            
            await websocket.send_text(json.dumps(packet_data))
            
            if sniff_mode == "simulation":
                await asyncio.sleep(random.uniform(1.0, 2.0))
                
    except WebSocketDisconnect:
        logger.info("Client disconnected.")
    except Exception as e:
        logger.error(f"WS error: {e}")
