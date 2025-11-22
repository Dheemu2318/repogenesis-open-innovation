"""
Vehicle Predictive Maintenance - FastAPI Backend
Integrates ML model with REST API endpoints
Compatible with Python 3.10 / 3.11+
"""

import os
import time
import sqlite3
import joblib
import uvicorn
import numpy as np
import pandas as pd
from datetime import datetime
from typing import List, Dict, Optional
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# Scikit-learn imports
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

# --- Configuration ---
DB_PATH = 'vehicle_diagnostics.db'
MODEL_PATH = 'engine_predictor.pkl'
RANDOM_SEED = 42
RPM_MIN = 800
RPM_MAX = 6000
TEMP_BASE = 75
TEMP_COEFFICIENT = 0.008

# Global storage for ML components
ml_models = {}

# --- Pydantic Models ---
class PredictionRequest(BaseModel):
    rpm: list[float] = Field(..., min_length=5, description="List of RPM readings (min 5)")
    oil_temp: list[float] = Field(..., min_length=5, description="List of Oil Temp readings (min 5)")

class PredictionResponse(BaseModel):
    success: bool
    prediction: str
    confidence: float
    anomaly_score: float
    risk_level: str
    severity: str
    explanation: str
    recommendation: str
    timestamp: str

class TripResponse(BaseModel):
    success: bool
    trip_id: str
    duration_min: float
    samples: int
    avg_rpm: float
    avg_temp: float
    max_temp: float
    message: str

class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    database_connected: bool
    timestamp: str

# --- Database Class ---
class DiagnosticsDB:
    def __init__(self, db_path=DB_PATH):
        self.db_path = db_path
        self._init_schema()
    
    def _init_schema(self):
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS trips (
                    trip_id TEXT PRIMARY KEY,
                    start_time TEXT NOT NULL,
                    end_time TEXT NOT NULL,
                    duration_min REAL,
                    avg_rpm REAL,
                    avg_temp REAL,
                    max_temp REAL,
                    samples INTEGER,
                    status TEXT DEFAULT 'Recorded'
                )
            ''')
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS readings (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    trip_id TEXT NOT NULL,
                    timestamp TEXT NOT NULL,
                    rpm REAL NOT NULL,
                    oil_temp REAL NOT NULL,
                    FOREIGN KEY(trip_id) REFERENCES trips(trip_id)
                )
            ''')
            conn.commit()
    
    def save_trip(self, trip_id: str, readings_df: pd.DataFrame) -> Dict:
        start_time = readings_df['timestamp'].iloc[0]
        end_time = readings_df['timestamp'].iloc[-1]
        duration = (pd.to_datetime(end_time) - pd.to_datetime(start_time)).seconds / 60
        
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute('''
                INSERT OR REPLACE INTO trips 
                (trip_id, start_time, end_time, duration_min, avg_rpm, avg_temp, max_temp, samples, status)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                trip_id, start_time, end_time, round(duration, 2),
                round(float(readings_df['rpm'].mean()), 1),
                round(float(readings_df['oil_temp'].mean()), 1),
                round(float(readings_df['oil_temp'].max()), 1),
                len(readings_df), 'Recorded'
            ))
            
            # Batch insert is faster
            readings_data = [
                (trip_id, row['timestamp'], row['rpm'], row['oil_temp'])
                for _, row in readings_df.iterrows()
            ]
            cursor.executemany('''
                INSERT INTO readings (trip_id, timestamp, rpm, oil_temp)
                VALUES (?, ?, ?, ?)
            ''', readings_data)
            conn.commit()
        
        return {
            "trip_id": trip_id,
            "duration_min": round(duration, 2),
            "samples": len(readings_df),
            "avg_rpm": round(float(readings_df['rpm'].mean()), 1),
            "avg_temp": round(float(readings_df['oil_temp'].mean()), 1),
            "max_temp": round(float(readings_df['oil_temp'].max()), 1)
        }
    
    def get_all_trips(self) -> list[dict]:
        with sqlite3.connect(self.db_path) as conn:
            df = pd.read_sql_query('SELECT * FROM trips ORDER BY start_time DESC', conn)
        return df.to_dict('records')

# --- Trip Simulator ---
class TripSimulator:
    def generate_trip(self, duration_min: int, interval_sec: int = 5, inject_fault: bool = False) -> pd.DataFrame:
        samples = int((duration_min * 60) / interval_sec)
        readings = []
        
        base_rpm = np.random.choice([1500, 2000, 2500, 3000, 3500])
        rpm_variation = 800
        
        # Pre-generate progress to avoid loop calculation overhead
        start_time = datetime.now()
        
        for i in range(samples):
            progress = i / samples
            
            rpm = base_rpm + np.random.uniform(-rpm_variation, rpm_variation)
            rpm = np.clip(rpm, RPM_MIN, RPM_MAX)
            
            expected_temp = TEMP_BASE + (rpm - RPM_MIN) * TEMP_COEFFICIENT
            expected_temp += progress * 15
            oil_temp = expected_temp + np.random.normal(0, 3)
            
            if inject_fault and 0.4 < progress < 0.6:
                if np.random.random() < 0.3:
                    oil_temp += np.random.uniform(20, 40)
            
            readings.append({
                'timestamp': (start_time + pd.Timedelta(seconds=i*interval_sec)).isoformat(),
                'rpm': round(float(rpm), 1),
                'oil_temp': round(float(oil_temp), 1)
            })
            
        return pd.DataFrame(readings)

# --- ML Model ---
class EngineFailurePredictor:
    def __init__(self):
        self.model = None
        self.scaler = StandardScaler()
        self.metrics = {}
    
    def prepare_features(self, df: pd.DataFrame) -> np.ndarray:
        features = pd.DataFrame()
        
        # Helper to safely handle single rows or empty logic
        if len(df) < 2:
            # Fallback for very short sequences
            rate = 0.0
        else:
            rate = (df['oil_temp'].iloc[-1] - df['oil_temp'].iloc[0]) / len(df)

        features['rpm_mean'] = [df['rpm'].mean()]
        features['rpm_std'] = [df['rpm'].std() if len(df) > 1 else 0]
        features['rpm_max'] = [df['rpm'].max()]
        features['rpm_min'] = [df['rpm'].min()]
        
        features['temp_mean'] = [df['oil_temp'].mean()]
        features['temp_std'] = [df['oil_temp'].std() if len(df) > 1 else 0]
        features['temp_max'] = [df['oil_temp'].max()]
        features['temp_min'] = [df['oil_temp'].min()]
        
        features['temp_rate'] = [rate]
        features['rpm_rate'] = [(df['rpm'].iloc[-1] - df['rpm'].iloc[0]) / len(df) if len(df) > 0 else 0]
        
        features['temp_rpm_ratio'] = [df['oil_temp'].mean() / (df['rpm'].mean() + 1)]
        features['temp_rpm_corr'] = [df['rpm'].corr(df['oil_temp']) if len(df) > 1 else 0]
        features = features.fillna(0) # Handle NaN from correlation
        
        features['temp_over_110'] = [(df['oil_temp'] > 110).sum()]
        features['temp_over_120'] = [(df['oil_temp'] > 120).sum()]
        features['rpm_over_5000'] = [(df['rpm'] > 5000).sum()]
        
        features['temp_range'] = [df['oil_temp'].max() - df['oil_temp'].min()]
        features['rpm_range'] = [df['rpm'].max() - df['rpm'].min()]
        
        return features.values
    
    def train(self, normal_trips: list[pd.DataFrame], faulty_trips: list[pd.DataFrame]):
        X_list = []
        y_list = []
        
        for trip_df in normal_trips:
            X_list.append(self.prepare_features(trip_df))
            y_list.append(0)
        
        for trip_df in faulty_trips:
            X_list.append(self.prepare_features(trip_df))
            y_list.append(1)
        
        X = np.vstack(X_list)
        y = np.array(y_list)
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=RANDOM_SEED, stratify=y
        )
        
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        self.model = RandomForestClassifier(
            n_estimators=100, # Reduced for faster startup in dev
            max_depth=10,
            random_state=RANDOM_SEED,
            n_jobs=-1
        )
        
        self.model.fit(X_train_scaled, y_train)
        
        test_score = self.model.score(X_test_scaled, y_test)
        
        # Handle case where only 1 class exists in test set
        try:
            y_pred_proba = self.model.predict_proba(X_test_scaled)[:, 1]
            auc = roc_auc_score(y_test, y_pred_proba)
        except ValueError:
            auc = 0.0

        self.metrics = {
            'test_accuracy': round(test_score, 4),
            'roc_auc': round(auc, 4)
        }
        return self.metrics
    
    def predict(self, df: pd.DataFrame) -> tuple:
        if self.model is None:
            raise ValueError("Model not trained")
        
        features = self.prepare_features(df)
        features_scaled = self.scaler.transform(features)
        
        prediction = self.model.predict(features_scaled)[0]
        probability = self.model.predict_proba(features_scaled)[0]
        
        # Return Python native types for JSON serialization
        return int(prediction), float(round(probability[1], 4))
    
    def save_model(self, path: str = MODEL_PATH):
        joblib.dump({
            'model': self.model,
            'scaler': self.scaler,
            'metrics': self.metrics,
            'timestamp': datetime.now().isoformat()
        }, path)
    
    def load_model(self, path: str = MODEL_PATH):
        if not os.path.exists(path):
            return False
        data = joblib.load(path)
        self.model = data['model']
        self.scaler = data['scaler']
        self.metrics = data['metrics']
        return True

def train_model_if_needed():
    print("Checking model status...")
    predictor = EngineFailurePredictor()
    
    # Try loading existing model
    if predictor.load_model():
        print(f"✅ Loaded cached model. Accuracy: {predictor.metrics.get('test_accuracy', 0):.2%}")
        ml_models['predictor'] = predictor
        return

    print("⚠️ No model found. Training new model (this may take a moment)...")
    simulator = TripSimulator()
    
    # Generate smaller dataset for quicker startup
    normal_trips = [simulator.generate_trip(5, 5, False) for _ in range(50)]
    faulty_trips = [simulator.generate_trip(5, 5, True) for _ in range(50)]
    
    metrics = predictor.train(normal_trips, faulty_trips)
    predictor.save_model()
    
    ml_models['predictor'] = predictor
    print(f"✅ Training complete. Accuracy: {metrics['test_accuracy']:.2%}")

# --- FastAPI Lifespan ---
@asynccontextmanager
async def lifespan(app: FastAPI):
    ml_models['db'] = DiagnosticsDB()
    ml_models['simulator'] = TripSimulator()
    
    # Run training/loading in a separate thread to not block startup if it was heavy
    # For simplicity here, we run it directly, but verify it's fast enough
    train_model_if_needed()
    
    yield
    
    ml_models.clear()
    print("🛑 Cleanup complete")

app = FastAPI(lifespan=lifespan, title="Vehicle PdM API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Endpoints ---
# NOTE: We use 'def' (not 'async def') for endpoints doing heavy CPU/DB work.
# FastAPI runs 'def' endpoints in a threadpool, preventing the event loop from blocking.

@app.get("/")
def root():
    return {"status": "online", "docs": "/docs"}

@app.get("/health", response_model=HealthResponse)
def health_check():
    return {
        "status": "healthy",
        "model_loaded": ml_models.get('predictor') is not None,
        "database_connected": ml_models.get('db') is not None,
        "timestamp": datetime.now().isoformat()
    }

@app.post("/predict", response_model=PredictionResponse)
def predict_anomaly(request: PredictionRequest):
    try:
        # Create DataFrame directly
        df = pd.DataFrame({'rpm': request.rpm, 'oil_temp': request.oil_temp})
        
        predictor = ml_models.get('predictor')
        if not predictor:
            raise HTTPException(503, "Model not ready")

        prediction, probability = predictor.predict(df)
        is_anomaly = (prediction == 1)
        
        if is_anomaly:
            if probability > 0.8:
                severity, risk = "Critical", "High"
                rec = "Immediate engine shutdown & inspection"
            else:
                severity, risk = "Warning", "Medium"
                rec = "Schedule maintenance soon"
        else:
            severity, risk, rec = "Normal", "Low", "Continue monitoring"
            
        return {
            "success": True,
            "prediction": "fault" if is_anomaly else "normal",
            "confidence": probability,
            "anomaly_score": probability * 100,
            "risk_level": risk,
            "severity": severity,
            "explanation": f"{severity} status detected based on sensor patterns",
            "recommendation": rec,
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        import traceback
        traceback.print_exc() # Print error to VS Code terminal
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/trip/record", response_model=TripResponse)
def record_trip(duration: int = 10, inject_fault: bool = False):
    try:
        trip_id = f"TRIP_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        simulator = ml_models['simulator']
        
        # Generate
        trip_data = simulator.generate_trip(duration, 5, inject_fault)
        
        # Save
        db = ml_models['db']
        trip_info = db.save_trip(trip_id, trip_data)
        
        return {"success": True, "message": "Recorded", **trip_info}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/trips")
def get_all_trips():
    db = ml_models['db']
    return db.get_all_trips()

if __name__ == "__main__":
    # Use 'app' object directly - prevents filename errors in VS Code
    uvicorn.run(app, host="0.0.0.0", port=8000)