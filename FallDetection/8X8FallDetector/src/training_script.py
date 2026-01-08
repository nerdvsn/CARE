#!/usr/bin/env python3
"""
Trainings-Skript für Sturzerkennung
===================================
Sammle eigene Daten und trainiere ein neues Random Forest Modell.

ANLEITUNG:
1. Starte im Aufnahme-Modus: python training_script.py --record
2. Führe normale Aktivitäten aus (Gehen, Sitzen, Hocken) → Taste 'n'
3. Simuliere Stürze → Taste 'f'
4. Beende mit 'q'
5. Trainiere: python training_script.py --train
"""

import json
import time
import serial
import numpy as np
from pathlib import Path
from collections import deque
from datetime import datetime
import argparse
import sys
from scipy.ndimage import zoom

# Konfiguration (muss mit fall_detector.py übereinstimmen!)
SERIAL_PORT = "/dev/ttyUSB0"
BAUD_RATE = 1000000
FRAME_BUFFER_SIZE = 8
DATA_DIR = Path(__file__).parent.parent / "data"


def downsample_frame(frame_32x24: np.ndarray) -> np.ndarray:
    """32x24 → 8x8"""
    frame_2d = frame_32x24.reshape(24, 32)
    downsampled = zoom(frame_2d, (8/24, 8/32), order=1)
    return downsampled.flatten()


def normalize_frame(frame: np.ndarray, ambient_temp: float) -> np.ndarray:
    """Normalisierung relativ zur Umgebungstemperatur"""
    return frame - ambient_temp


class DataRecorder:
    """Nimmt Trainings-Daten auf"""
    
    def __init__(self, port: str, baudrate: int):
        self.frame_buffer = deque(maxlen=FRAME_BUFFER_SIZE)
        self.samples = []  # Liste von (features, label)
        self.serial_conn = None
        
        # Serial verbinden
        self.serial_conn = serial.Serial(port, baudrate, timeout=1.0)
        time.sleep(2)
        print(f"[OK] Verbunden mit {port}")
    
    def read_frame(self) -> tuple | None:
        try:
            line = self.serial_conn.readline().decode('utf-8').strip()
            if not line or "error" in line.lower():
                return None
            
            data = json.loads(line)
            if "temperature" in data:
                frame = np.array(data["temperature"], dtype=np.float32)
                ambient = data.get("at", 25.0)
                return frame, ambient
        except:
            pass
        return None
    
    def record_sample(self, label: int):
        """Speichert aktuelle 8 Frames als Sample"""
        if len(self.frame_buffer) < FRAME_BUFFER_SIZE:
            print(f"  ⚠️ Buffer nicht voll ({len(self.frame_buffer)}/{FRAME_BUFFER_SIZE})")
            return False
        
        features = np.concatenate(list(self.frame_buffer))
        self.samples.append((features, label))
        
        label_str = "STURZ" if label == 1 else "NORMAL"
        print(f"  ✓ Sample #{len(self.samples)} aufgenommen: {label_str}")
        
        # Buffer leeren für nächstes Sample
        self.frame_buffer.clear()
        return True
    
    def run(self):
        """Interaktiver Aufnahme-Modus"""
        print("\n" + "="*50)
        print("  TRAININGS-DATEN AUFNAHME")
        print("="*50)
        print("  Tasten:")
        print("    n = Normale Aktivität aufnehmen")
        print("    f = Sturz aufnehmen")
        print("    s = Speichern und Beenden")
        print("    q = Beenden ohne Speichern")
        print("="*50 + "\n")
        
        # Non-blocking keyboard input
        import threading
        from queue import Queue, Empty
        
        key_queue = Queue()
        
        def key_listener():
            while True:
                try:
                    key = input()
                    key_queue.put(key.lower().strip())
                except:
                    break
        
        listener_thread = threading.Thread(target=key_listener, daemon=True)
        listener_thread.start()
        
        frame_count = 0
        running = True
        
        try:
            while running:
                # Frame lesen
                result = self.read_frame()
                if result:
                    frame_raw, ambient = result
                    frame_down = downsample_frame(frame_raw)
                    frame_norm = normalize_frame(frame_down, ambient)
                    self.frame_buffer.append(frame_norm)
                    frame_count += 1
                    
                    if frame_count % 50 == 0:
                        print(f"[Frame {frame_count}] Buffer: {len(self.frame_buffer)}/{FRAME_BUFFER_SIZE} | Samples: {len(self.samples)}")
                
                # Tastatur prüfen
                try:
                    key = key_queue.get_nowait()
                    
                    if key == 'n':
                        self.record_sample(label=0)  # Normal
                    elif key == 'f':
                        self.record_sample(label=1)  # Fall
                    elif key == 's':
                        self.save_samples()
                        running = False
                    elif key == 'q':
                        print("\n[INFO] Beende ohne Speichern...")
                        running = False
                        
                except Empty:
                    pass
                    
        except KeyboardInterrupt:
            print("\n[INFO] Unterbrochen")
        
        finally:
            if self.serial_conn:
                self.serial_conn.close()
    
    def save_samples(self):
        """Speichert Samples als .npz Datei"""
        if not self.samples:
            print("[WARN] Keine Samples zum Speichern!")
            return
        
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        
        X = np.array([s[0] for s in self.samples])
        y = np.array([s[1] for s in self.samples])
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = DATA_DIR / f"training_data_{timestamp}.npz"
        
        np.savez(filename, X=X, y=y)
        
        print(f"\n[OK] Gespeichert: {filename}")
        print(f"     Samples: {len(self.samples)}")
        print(f"     Normal: {np.sum(y == 0)}, Stürze: {np.sum(y == 1)}")


def train_model():
    """Trainiert Random Forest mit gesammelten Daten"""
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import train_test_split, cross_val_score
    from sklearn.metrics import classification_report, confusion_matrix
    import joblib
    
    print("\n" + "="*50)
    print("  MODELL TRAINING")
    print("="*50)
    
    # Alle .npz Dateien laden
    data_files = list(DATA_DIR.glob("training_data_*.npz"))
    
    if not data_files:
        print(f"[ERROR] Keine Trainingsdaten gefunden in {DATA_DIR}")
        print("        Führe zuerst --record aus!")
        return
    
    print(f"[INFO] Gefundene Datensätze: {len(data_files)}")
    
    X_all, y_all = [], []
    for f in data_files:
        data = np.load(f)
        X_all.append(data['X'])
        y_all.append(data['y'])
        print(f"       - {f.name}: {len(data['X'])} Samples")
    
    X = np.vstack(X_all)
    y = np.concatenate(y_all)
    
    print(f"\n[INFO] Gesamt: {len(X)} Samples")
    print(f"       Normal: {np.sum(y == 0)}, Stürze: {np.sum(y == 1)}")
    
    # Train/Test Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Modell erstellen (Parameter wie Original)
    print("\n[INFO] Trainiere Random Forest...")
    model = RandomForestClassifier(
        n_estimators=100,
        criterion='gini',
        max_depth=None,
        random_state=42,
        n_jobs=-1
    )
    
    model.fit(X_train, y_train)
    
    # Evaluation
    print("\n[INFO] Evaluation:")
    y_pred = model.predict(X_test)
    
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=['Normal', 'Sturz']))
    
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))
    
    # Cross-Validation
    cv_scores = cross_val_score(model, X, y, cv=5)
    print(f"\n5-Fold CV Score: {cv_scores.mean():.3f} (+/- {cv_scores.std()*2:.3f})")
    
    # Speichern
    model_path = Path(__file__).parent.parent / "models" / "random_forest_model.joblib"
    model_path.parent.mkdir(parents=True, exist_ok=True)
    
    joblib.dump(model, model_path)
    print(f"\n[OK] Modell gespeichert: {model_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Training für Sturzerkennung")
    parser.add_argument("--record", action="store_true", help="Aufnahme-Modus starten")
    parser.add_argument("--train", action="store_true", help="Modell trainieren")
    parser.add_argument("--port", default=SERIAL_PORT, help="Serial Port")
    parser.add_argument("--baud", type=int, default=BAUD_RATE, help="Baudrate")
    
    args = parser.parse_args()
    
    if args.record:
        recorder = DataRecorder(args.port, args.baud)
        recorder.run()
    elif args.train:
        train_model()
    else:
        print("Verwende --record oder --train")
        print("Beispiel: python training_script.py --record --port /dev/ttyUSB0")
