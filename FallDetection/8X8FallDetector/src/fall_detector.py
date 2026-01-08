#!/usr/bin/env python3
"""
Echtzeit-Sturzerkennung mit MLX90640 + ESP32 + Random Forest
Basierend auf Master-Thesis: Panasonic Grid-EYE Fall Detection
"""

import json
import time
import serial
import numpy as np
from pathlib import Path
from collections import deque
from datetime import datetime
import joblib
from scipy.ndimage import zoom

# ========== KONFIGURATION ==========
SERIAL_PORT = "/dev/ttyUSB0"  # Linux - anpassen für Windows: "COM3"
BAUD_RATE = 1000000           # Muss mit ESP32 übereinstimmen
FRAME_BUFFER_SIZE = 8         # Paper: 8 Frames pro Sequenz (bestätigt: 8×64=512 Features)
DETECTION_THRESHOLD = 0.7     # Confidence für Alarm
MODEL_PATH = Path(__file__).parent.parent / "models" / "random_forest_model.joblib"

# Modell-Info (aus Original extrahiert):
# - 100 Bäume (n_estimators)
# - 512 Features = 8 Frames × 8×8 Pixel
# - Gini-Kriterium
# - Unbegrenzte Tiefe

# ========== PREPROCESSING (Paper-konform) ==========

def downsample_frame(frame_32x24: np.ndarray) -> np.ndarray:
    """
    Reduziert 32x24 auf 8x8 (wie im Original-Paper)
    Verwendet Area-Interpolation für beste Ergebnisse
    """
    frame_2d = frame_32x24.reshape(24, 32)
    # Zoom-Faktor: 8/24 = 0.333 für Höhe, 8/32 = 0.25 für Breite
    downsampled = zoom(frame_2d, (8/24, 8/32), order=1)
    return downsampled.flatten()


def normalize_frame(frame: np.ndarray, ambient_temp: float) -> np.ndarray:
    """
    Paper Section 2.2: Normalisierung relativ zur Umgebungstemperatur
    T_norm = T - T_ambient
    """
    return frame - ambient_temp


def extract_features(frames: list[np.ndarray]) -> np.ndarray:
    """
    Extrahiert Features aus 8 aufeinanderfolgenden Frames
    Paper: Flattened sequence von 8 x 64 = 512 Features
    """
    # Stack alle Frames zu einem Feature-Vektor
    features = np.concatenate(frames)
    return features


# ========== HAUPTKLASSE ==========

class FallDetector:
    def __init__(self, model_path: Path, port: str, baudrate: int):
        self.frame_buffer = deque(maxlen=FRAME_BUFFER_SIZE)
        self.model = None
        self.serial_conn = None
        self.running = False
        
        # Model laden
        self.load_model(model_path)
        
        # Serial verbinden
        self.connect_serial(port, baudrate)
    
    def load_model(self, model_path: Path):
        """Lädt das Random Forest Modell"""
        if model_path.exists():
            print(f"[INFO] Lade Modell: {model_path}")
            self.model = joblib.load(model_path)
            print(f"[OK] Modell geladen: {type(self.model).__name__}")
        else:
            print(f"[WARN] Modell nicht gefunden: {model_path}")
            print("[WARN] Starte im DEMO-Modus (nur Visualisierung)")
            self.model = None
    
    def connect_serial(self, port: str, baudrate: int):
        """Verbindet mit ESP32 über Serial"""
        try:
            self.serial_conn = serial.Serial(
                port=port,
                baudrate=baudrate,
                timeout=1.0
            )
            time.sleep(2)  # ESP32 Reset abwarten
            print(f"[OK] Verbunden mit {port} @ {baudrate} baud")
        except serial.SerialException as e:
            print(f"[ERROR] Serial-Verbindung fehlgeschlagen: {e}")
            print("[INFO] Verfügbare Ports:")
            import serial.tools.list_ports
            for p in serial.tools.list_ports.comports():
                print(f"       - {p.device}: {p.description}")
            raise
    
    def read_frame(self) -> tuple[np.ndarray, float] | None:
        """Liest einen Frame vom ESP32"""
        try:
            line = self.serial_conn.readline().decode('utf-8').strip()
            if not line or line.startswith("{\"error"):
                return None
            
            data = json.loads(line)
            
            if "temperature" in data:
                frame = np.array(data["temperature"], dtype=np.float32)
                ambient = data.get("at", 25.0)
                return frame, ambient
                
        except (json.JSONDecodeError, UnicodeDecodeError) as e:
            pass  # Ignoriere fehlerhafte Frames
        
        return None
    
    def predict(self, features: np.ndarray) -> tuple[int, float]:
        """
        Führt Prediction durch
        Returns: (class, confidence)
            class: 0 = Normal, 1 = Fall
        """
        if self.model is None:
            return 0, 0.0
        
        # Reshape für sklearn
        X = features.reshape(1, -1)
        
        # Prediction
        prediction = self.model.predict(X)[0]
        
        # Confidence (Probability)
        if hasattr(self.model, 'predict_proba'):
            proba = self.model.predict_proba(X)[0]
            confidence = proba[1] if prediction == 1 else proba[0]
        else:
            confidence = 1.0
        
        return int(prediction), float(confidence)
    
    def run(self, callback=None):
        """
        Hauptloop für Echtzeit-Erkennung
        callback(frame, prediction, confidence) wird bei jedem Frame aufgerufen
        """
        self.running = True
        print("\n" + "="*50)
        print("  STURZERKENNUNG AKTIV")
        print("  Drücke Ctrl+C zum Beenden")
        print("="*50 + "\n")
        
        frame_count = 0
        last_prediction = 0
        last_confidence = 0.0
        
        try:
            while self.running:
                result = self.read_frame()
                
                if result is None:
                    continue
                
                frame_raw, ambient_temp = result
                frame_count += 1
                
                # Preprocessing
                frame_downsampled = downsample_frame(frame_raw)
                frame_normalized = normalize_frame(frame_downsampled, ambient_temp)
                
                # Buffer füllen
                self.frame_buffer.append(frame_normalized)
                
                # Prediction wenn Buffer voll
                if len(self.frame_buffer) == FRAME_BUFFER_SIZE:
                    features = extract_features(list(self.frame_buffer))
                    prediction, confidence = self.predict(features)
                    
                    last_prediction = prediction
                    last_confidence = confidence
                    
                    # Alarm bei Sturz
                    if prediction == 1 and confidence >= DETECTION_THRESHOLD:
                        timestamp = datetime.now().strftime("%H:%M:%S")
                        print(f"\n{'!'*50}")
                        print(f"  ⚠️  STURZ ERKANNT! [{timestamp}]")
                        print(f"  Confidence: {confidence:.1%}")
                        print(f"{'!'*50}\n")
                
                # Callback für UI
                if callback:
                    callback(
                        frame=frame_raw.reshape(24, 32),
                        prediction=last_prediction,
                        confidence=last_confidence,
                        ambient=ambient_temp
                    )
                
                # Status alle 50 Frames
                if frame_count % 50 == 0:
                    status = "STURZ" if last_prediction == 1 else "Normal"
                    print(f"[Frame {frame_count}] Status: {status} ({last_confidence:.1%}) | Ambient: {ambient_temp:.1f}°C")
        
        except KeyboardInterrupt:
            print("\n[INFO] Beende...")
        
        finally:
            self.running = False
            if self.serial_conn:
                self.serial_conn.close()
                print("[OK] Serial-Verbindung geschlossen")


# ========== CLI ==========

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Echtzeit-Sturzerkennung")
    parser.add_argument("--port", default=SERIAL_PORT, help="Serial Port (z.B. /dev/ttyUSB0 oder COM3)")
    parser.add_argument("--baud", type=int, default=BAUD_RATE, help="Baudrate")
    parser.add_argument("--model", default=str(MODEL_PATH), help="Pfad zum Modell")
    parser.add_argument("--threshold", type=float, default=DETECTION_THRESHOLD, help="Detection Threshold")
    
    args = parser.parse_args()
    
    detector = FallDetector(
        model_path=Path(args.model),
        port=args.port,
        baudrate=args.baud
    )
    
    detector.run()
