#!/usr/bin/env python3
"""
STURZERKENNUNG - TRAINING & ERKENNUNG
=====================================
Da das Original-Modell defekt ist, trainieren wir ein neues!

ANLEITUNG:
1. Daten sammeln:  python train_and_detect.py --record
2. Trainieren:     python train_and_detect.py --train
3. Testen:         python train_and_detect.py --detect
"""

import numpy as np
import sys
import os
import json
import time
import argparse
from datetime import datetime
from collections import deque
from pathlib import Path

# === KONFIGURATION ===
SERIAL_PORT = '/dev/ttyUSB0'
BAUD_RATE = 1000000
DATA_DIR = Path("training_data")
MODEL_FILE = "my_fall_model.pkl"

WINDOW_SIZE = 8        # 8 Frames pro Sequenz
TARGET_FPS = 8         # 8 Hz
FRAME_INTERVAL = 1.0 / TARGET_FPS

# === HILFSFUNKTIONEN ===

def downsample_to_8x8(flat_768):
    """32×24 → 8×8"""
    matrix = np.array(flat_768, dtype=np.float32).reshape(24, 32)
    blocks = matrix.reshape(8, 3, 8, 4)
    return blocks.mean(axis=(1, 3))


# ============================================================
#                    DATEN AUFNEHMEN
# ============================================================

def record_data():
    """Interaktive Datenaufnahme"""
    import serial
    import cv2
    
    print("="*60)
    print("  TRAININGS-DATEN AUFNAHME")
    print("="*60)
    print("""
  STEUERUNG:
    [n] = Aktuelle Sequenz als NORMAL speichern
    [f] = Aktuelle Sequenz als STURZ speichern
    [q] = Beenden
    
  TIPPS:
    - Für NORMAL: Gehen, Stehen, Sitzen, Hinsetzen
    - Für STURZ: Verschiedene Fallrichtungen aufnehmen
    - Mindestens 30 Samples pro Klasse sammeln
    """)
    print("="*60)
    
    # Serial verbinden
    try:
        ser = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=1)
        ser.flushInput()
        time.sleep(1)
        print(f"✅ Verbunden mit {SERIAL_PORT}")
    except Exception as e:
        print(f"❌ Serial-Fehler: {e}")
        return
    
    # Daten-Ordner erstellen
    DATA_DIR.mkdir(exist_ok=True)
    
    # Bestehende Daten zählen
    existing_normal = len(list(DATA_DIR.glob("normal_*.npy")))
    existing_fall = len(list(DATA_DIR.glob("fall_*.npy")))
    print(f"📊 Bestehende Daten: {existing_normal} Normal, {existing_fall} Sturz")
    
    frame_buffer = deque(maxlen=WINDOW_SIZE)
    last_sample_time = 0
    
    normal_count = existing_normal
    fall_count = existing_fall
    
    print("\n🎬 Starte Aufnahme...\n")
    
    try:
        while True:
            # Frame lesen
            if ser.in_waiting > 0:
                line = ser.readline().decode('utf-8', errors='ignore').strip()
                try:
                    data = json.loads(line)
                    if 'temperature' not in data:
                        continue
                    
                    current_time = time.time()
                    if current_time - last_sample_time < FRAME_INTERVAL:
                        continue
                    last_sample_time = current_time
                    
                    raw = data['temperature']
                    ambient = data['at']
                    
                    # Preprocessing
                    frame_8x8 = downsample_to_8x8(raw)
                    normalized = frame_8x8 - ambient
                    frame_buffer.append(normalized.flatten())
                    
                    # Visualisierung
                    disp = np.clip(normalized, -2, 12)
                    disp = ((disp + 2) / 14 * 255).astype(np.uint8)
                    img = cv2.resize(disp, (400, 400), interpolation=cv2.INTER_NEAREST)
                    img = cv2.applyColorMap(img, cv2.COLORMAP_JET)
                    
                    # Status
                    buffer_full = len(frame_buffer) == WINDOW_SIZE
                    status = "BEREIT" if buffer_full else f"Sammle... {len(frame_buffer)}/8"
                    color = (0, 255, 0) if buffer_full else (0, 255, 255)
                    
                    cv2.putText(img, status, (10, 30), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
                    cv2.putText(img, f"Normal: {normal_count} | Sturz: {fall_count}", 
                                (10, 380), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 1)
                    cv2.putText(img, "[N]=Normal  [F]=Sturz  [Q]=Quit", 
                                (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200,200,200), 1)
                    
                    cv2.imshow('Training Data Collection', img)
                    
                except json.JSONDecodeError:
                    continue
            
            # Tastatur
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('n') and len(frame_buffer) == WINDOW_SIZE:
                # Normal speichern
                features = np.array(list(frame_buffer)).flatten()
                filename = DATA_DIR / f"normal_{normal_count:04d}.npy"
                np.save(filename, features)
                normal_count += 1
                print(f"✅ Normal #{normal_count} gespeichert")
                frame_buffer.clear()
                
            elif key == ord('f') and len(frame_buffer) == WINDOW_SIZE:
                # Sturz speichern
                features = np.array(list(frame_buffer)).flatten()
                filename = DATA_DIR / f"fall_{fall_count:04d}.npy"
                np.save(filename, features)
                fall_count += 1
                print(f"✅ Sturz #{fall_count} gespeichert")
                frame_buffer.clear()
                
            elif key == ord('q'):
                break
    
    except KeyboardInterrupt:
        pass
    finally:
        ser.close()
        cv2.destroyAllWindows()
    
    print(f"\n📊 Gespeichert: {normal_count} Normal, {fall_count} Sturz")
    print(f"   Ordner: {DATA_DIR.absolute()}")


# ============================================================
#                    MODELL TRAINIEREN
# ============================================================

def train_model():
    """Trainiert Random Forest mit gesammelten Daten"""
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import train_test_split, cross_val_score
    from sklearn.metrics import classification_report, confusion_matrix
    import joblib
    
    print("="*60)
    print("  MODELL TRAINING")
    print("="*60)
    
    # Daten laden
    normal_files = list(DATA_DIR.glob("normal_*.npy"))
    fall_files = list(DATA_DIR.glob("fall_*.npy"))
    
    print(f"\n📊 Gefundene Daten:")
    print(f"   Normal: {len(normal_files)} Samples")
    print(f"   Sturz:  {len(fall_files)} Samples")
    
    if len(normal_files) < 10 or len(fall_files) < 10:
        print("\n⚠️  Zu wenig Daten! Mindestens 10 Samples pro Klasse benötigt.")
        print("   Führe zuerst --record aus.")
        return
    
    # Daten laden
    X = []
    y = []
    
    for f in normal_files:
        X.append(np.load(f))
        y.append(0)
    
    for f in fall_files:
        X.append(np.load(f))
        y.append(1)
    
    X = np.array(X)
    y = np.array(y)
    
    print(f"\n   Feature-Shape: {X.shape}")
    
    # Train/Test Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"   Training: {len(X_train)} Samples")
    print(f"   Test: {len(X_test)} Samples")
    
    # Modell trainieren
    print("\n🔧 Trainiere Random Forest...")
    
    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=None,
        min_samples_split=2,
        random_state=42,
        n_jobs=-1
    )
    
    model.fit(X_train, y_train)
    
    # Evaluation
    print("\n📈 Evaluation:")
    
    y_pred = model.predict(X_test)
    
    print("\n   Classification Report:")
    print(classification_report(y_test, y_pred, target_names=['Normal', 'Sturz']))
    
    print("   Confusion Matrix:")
    cm = confusion_matrix(y_test, y_pred)
    print(f"   [[TN={cm[0,0]:3d}  FP={cm[0,1]:3d}]")
    print(f"    [FN={cm[1,0]:3d}  TP={cm[1,1]:3d}]]")
    
    # Cross-Validation
    cv_scores = cross_val_score(model, X, y, cv=5)
    print(f"\n   5-Fold CV: {cv_scores.mean():.1%} (+/- {cv_scores.std()*2:.1%})")
    
    # Funktionstest
    print("\n🧪 Funktionstest:")
    test_inputs = [
        ("Nullen", np.zeros((1, 512))),
        ("Einsen", np.ones((1, 512)) * 5),
        ("Zufall", np.random.randn(1, 512)),
    ]
    
    probs = []
    for name, data in test_inputs:
        p = model.predict_proba(data)[0][1]
        probs.append(p)
        print(f"   {name}: P(Fall) = {p:.3f}")
    
    if len(set([round(p, 2) for p in probs])) > 1:
        print("   ✅ Modell reagiert unterschiedlich!")
    else:
        print("   ⚠️  Modell gibt gleiche Werte - mehr Daten sammeln!")
    
    # Speichern
    print(f"\n💾 Speichere als: {MODEL_FILE}")
    joblib.dump(model, MODEL_FILE)
    print("✅ Fertig!")
    
    print("\n" + "="*60)
    print(f"  Teste mit: python {sys.argv[0]} --detect")
    print("="*60)


# ============================================================
#                    ERKENNUNG
# ============================================================

def detect():
    """Echtzeit-Sturzerkennung mit trainiertem Modell"""
    import serial
    import cv2
    import joblib
    
    print("="*60)
    print("  ECHTZEIT-STURZERKENNUNG")
    print("="*60)
    
    # Modell laden
    if not os.path.exists(MODEL_FILE):
        print(f"❌ Modell nicht gefunden: {MODEL_FILE}")
        print(f"   Trainiere zuerst mit: python {sys.argv[0]} --train")
        return
    
    model = joblib.load(MODEL_FILE)
    print(f"✅ Modell geladen: {MODEL_FILE}")
    
    # Serial
    try:
        ser = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=1)
        ser.flushInput()
        time.sleep(1)
        print(f"✅ Verbunden mit {SERIAL_PORT}")
    except Exception as e:
        print(f"❌ Serial-Fehler: {e}")
        return
    
    frame_buffer = deque(maxlen=WINDOW_SIZE)
    last_sample_time = 0
    detection_count = 0
    
    print("\n🔍 Starte Erkennung... [Q] zum Beenden\n")
    
    try:
        while True:
            if ser.in_waiting > 0:
                line = ser.readline().decode('utf-8', errors='ignore').strip()
                try:
                    data = json.loads(line)
                    if 'temperature' not in data:
                        continue
                    
                    current_time = time.time()
                    if current_time - last_sample_time < FRAME_INTERVAL:
                        continue
                    last_sample_time = current_time
                    
                    raw = data['temperature']
                    ambient = data['at']
                    
                    frame_8x8 = downsample_to_8x8(raw)
                    normalized = frame_8x8 - ambient
                    frame_buffer.append(normalized.flatten())
                    
                    # Prediction
                    prediction = 0
                    fall_prob = 0.0
                    
                    if len(frame_buffer) == WINDOW_SIZE:
                        features = np.array(list(frame_buffer)).flatten().reshape(1, -1)
                        prediction = model.predict(features)[0]
                        fall_prob = model.predict_proba(features)[0][1]
                    
                    # Visualisierung
                    disp = np.clip(normalized, -2, 12)
                    disp = ((disp + 2) / 14 * 255).astype(np.uint8)
                    img = cv2.resize(disp, (400, 400), interpolation=cv2.INTER_NEAREST)
                    img = cv2.applyColorMap(img, cv2.COLORMAP_JET)
                    
                    # Status
                    if prediction == 1 and fall_prob > 0.5:
                        status = "STURZ ERKANNT!"
                        color = (0, 0, 255)
                        cv2.rectangle(img, (0,0), (399,399), (0,0,255), 8)
                        detection_count += 1
                        print(f"⚠️  STURZ! [{datetime.now().strftime('%H:%M:%S')}] P={fall_prob:.1%}")
                    else:
                        status = "Normal"
                        color = (0, 255, 0)
                    
                    cv2.putText(img, status, (10, 40), 
                                cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 3)
                    cv2.putText(img, f"P(Fall): {fall_prob:.1%}", (10, 80),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
                    
                    # Probability bar
                    bar_w = int(fall_prob * 380)
                    cv2.rectangle(img, (10, 360), (390, 385), (50,50,50), -1)
                    if bar_w > 0:
                        bar_color = (0,255,0) if fall_prob < 0.5 else (0,0,255)
                        cv2.rectangle(img, (10, 360), (10+bar_w, 385), bar_color, -1)
                    
                    cv2.imshow('Fall Detection', img)
                    
                except json.JSONDecodeError:
                    continue
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    
    except KeyboardInterrupt:
        pass
    finally:
        ser.close()
        cv2.destroyAllWindows()
        print(f"\n📊 Erkennungen: {detection_count}")


# ============================================================
#                    MAIN
# ============================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Sturzerkennung Training & Erkennung")
    parser.add_argument("--record", action="store_true", help="Trainings-Daten aufnehmen")
    parser.add_argument("--train", action="store_true", help="Modell trainieren")
    parser.add_argument("--detect", action="store_true", help="Echtzeit-Erkennung starten")
    parser.add_argument("--port", default=SERIAL_PORT, help="Serial Port")
    
    args = parser.parse_args()
    
    if args.port:
        SERIAL_PORT = args.port
    
    if args.record:
        record_data()
    elif args.train:
        train_model()
    elif args.detect:
        detect()
    else:
        print("Sturzerkennung - Training & Erkennung")
        print("="*40)
        print("\nBenutzung:")
        print(f"  1. Daten sammeln:  python {sys.argv[0]} --record")
        print(f"  2. Trainieren:     python {sys.argv[0]} --train")
        print(f"  3. Erkennen:       python {sys.argv[0]} --detect")
        print("\nOptionen:")
        print(f"  --port PORT   Serial Port (Standard: {SERIAL_PORT})")