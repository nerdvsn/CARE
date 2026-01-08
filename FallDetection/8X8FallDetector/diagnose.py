#!/usr/bin/env python3
"""
DIAGNOSE-SKRIPT
Findet heraus warum die Probability immer 32% ist
"""

import numpy as np
import sys

if not hasattr(np, "float"):
    np.float = float
if not hasattr(np, "bool"):
    np.bool = bool
if not hasattr(np, "int"):
    np.int = int
if not hasattr(np, "typeDict"):
    np.typeDict = {}

import serial
import json
import joblib
import time
from collections import deque

SERIAL_PORT = '/dev/ttyUSB0'
BAUD_RATE = 1000000
MODEL_PATH = 'model__random_forest.pkl'

print("="*60)
print("  DIAGNOSE")
print("="*60)

# 1. MODELL TESTEN
print("\n[1] MODELL-TEST")
print("-"*40)

try:
    model = joblib.load(MODEL_PATH)
    print(f"✅ Modell geladen")
    
    # Test mit Zufallsdaten
    print("\n   Test mit verschiedenen Inputs:")
    
    # Test 1: Nullen
    test_zeros = np.zeros((1, 512))
    pred = model.predict(test_zeros)[0]
    prob = model.predict_proba(test_zeros)[0]
    print(f"   - Nur Nullen:        Pred={pred}, Prob=[{prob[0]:.3f}, {prob[1]:.3f}]")
    
    # Test 2: Kleine Werte (normale Temp-Differenz)
    test_small = np.ones((1, 512)) * 2.0
    pred = model.predict(test_small)[0]
    prob = model.predict_proba(test_small)[0]
    print(f"   - Kleine Werte (2):  Pred={pred}, Prob=[{prob[0]:.3f}, {prob[1]:.3f}]")
    
    # Test 3: Große Werte (heiß)
    test_big = np.ones((1, 512)) * 10.0
    pred = model.predict(test_big)[0]
    prob = model.predict_proba(test_big)[0]
    print(f"   - Große Werte (10):  Pred={pred}, Prob=[{prob[0]:.3f}, {prob[1]:.3f}]")
    
    # Test 4: Zufallswerte
    test_rand = np.random.randn(1, 512) * 5
    pred = model.predict(test_rand)[0]
    prob = model.predict_proba(test_rand)[0]
    print(f"   - Zufallswerte:      Pred={pred}, Prob=[{prob[0]:.3f}, {prob[1]:.3f}]")
    
    # Test 5: Simulierter "Sturz" - große Änderung in der Mitte
    test_fall = np.zeros((1, 512))
    # Frames 1-4: Person steht (mittlerer Bereich warm)
    for i in range(4):
        test_fall[0, i*64 + 27:i*64 + 37] = 8.0  # Mitte warm
    # Frames 5-8: Person liegt (unterer Bereich warm)  
    for i in range(4, 8):
        test_fall[0, i*64 + 56:i*64 + 64] = 8.0  # Unten warm
    pred = model.predict(test_fall)[0]
    prob = model.predict_proba(test_fall)[0]
    print(f"   - Simulierter Sturz: Pred={pred}, Prob=[{prob[0]:.3f}, {prob[1]:.3f}]")
    
except Exception as e:
    print(f"❌ Modell-Fehler: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# 2. SERIAL-DATEN TESTEN
print("\n[2] SERIAL-DATEN TEST")
print("-"*40)

try:
    ser = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=2)
    ser.flushInput()
    print(f"✅ Serial verbunden")
    
    time.sleep(1)
    
    # Sammle 10 Frames
    frames = []
    print("   Sammle 10 Frames...")
    
    for i in range(50):  # Max 50 Versuche
        if ser.in_waiting > 0:
            line = ser.readline().decode('utf-8', errors='ignore').strip()
            try:
                data = json.loads(line)
                if 'temperature' in data:
                    frames.append({
                        'temp': np.array(data['temperature']),
                        'ambient': data['at']
                    })
                    if len(frames) >= 10:
                        break
            except:
                pass
        time.sleep(0.1)
    
    if len(frames) < 10:
        print(f"⚠️  Nur {len(frames)} Frames empfangen")
    else:
        print(f"✅ {len(frames)} Frames empfangen")
    
    # Analysiere Frames
    print("\n   Frame-Analyse:")
    for i, f in enumerate(frames[:5]):
        temp = f['temp']
        amb = f['ambient']
        delta = temp - amb
        print(f"   Frame {i+1}: Ambient={amb:.1f}°C, "
              f"Temp Range=[{temp.min():.1f}, {temp.max():.1f}], "
              f"Delta Range=[{delta.min():.1f}, {delta.max():.1f}]")
    
    # Prüfe ob sich Daten ändern
    if len(frames) >= 2:
        diff = np.abs(frames[0]['temp'] - frames[-1]['temp']).sum()
        print(f"\n   Differenz Frame1 vs Frame{len(frames)}: {diff:.2f}")
        if diff < 1.0:
            print("   ⚠️  WARNUNG: Frames ändern sich kaum!")
    
    ser.close()
    
except Exception as e:
    print(f"❌ Serial-Fehler: {e}")

# 3. BUFFER-TEST
print("\n[3] BUFFER & FEATURE-TEST")
print("-"*40)

def downsample_to_8x8(flat_768):
    matrix = np.array(flat_768, dtype=np.float32).reshape(24, 32)
    blocks = matrix.reshape(8, 3, 8, 4)
    return blocks.mean(axis=(1, 3))

if len(frames) >= 8:
    print("   Erstelle Feature-Vektor aus 8 Frames...")
    
    buffer = []
    for f in frames[:8]:
        ds = downsample_to_8x8(f['temp'])
        norm = ds - f['ambient']
        buffer.append(norm.flatten())
    
    features = np.array(buffer).flatten().reshape(1, -1)
    
    print(f"   Feature-Shape: {features.shape}")
    print(f"   Feature-Range: [{features.min():.2f}, {features.max():.2f}]")
    print(f"   Feature-Mean: {features.mean():.2f}")
    print(f"   Feature-Std: {features.std():.2f}")
    
    # Prediction mit echten Daten
    pred = model.predict(features)[0]
    prob = model.predict_proba(features)[0]
    print(f"\n   Prediction mit echten Daten:")
    print(f"   → Pred={pred}, Prob=[{prob[0]:.3f}, {prob[1]:.3f}]")
    print(f"   → P(Fall) = {prob[1]*100:.1f}%")
    
    # Jetzt OHNE Normalisierung testen
    print("\n   Test OHNE Ambient-Subtraktion:")
    buffer_raw = []
    for f in frames[:8]:
        ds = downsample_to_8x8(f['temp'])
        buffer_raw.append(ds.flatten())  # KEINE Subtraktion!
    
    features_raw = np.array(buffer_raw).flatten().reshape(1, -1)
    pred = model.predict(features_raw)[0]
    prob = model.predict_proba(features_raw)[0]
    print(f"   → Pred={pred}, Prob=[{prob[0]:.3f}, {prob[1]:.3f}]")
    print(f"   → P(Fall) = {prob[1]*100:.1f}%")

print("\n" + "="*60)
print("  DIAGNOSE ABGESCHLOSSEN")
print("="*60)