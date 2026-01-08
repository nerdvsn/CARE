#!/usr/bin/env python3
"""
MODELL-KONVERTER v2 - mit detaillierter Fehleranalyse
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
    
import sys
import pickle
import joblib


INPUT_FILE = "model__random_forest.pkl"
OUTPUT_FILE = "model_random_forest_CONVERTED.pkl"

print("="*60)
print("  MODELL-KONVERTIERUNG v2")
print("="*60)

# sklearn Version
try:
    import sklearn
    print(f"sklearn Version: {sklearn.__version__}")
except:
    print("sklearn nicht gefunden!")
    sys.exit(1)

print(f"\n[1] Versuche {INPUT_FILE} zu laden...")
print("-"*40)

model = None
errors = []

# Methode 1: joblib.load
print("\n   Methode 1: joblib.load()")
try:
    model = joblib.load(INPUT_FILE)
    print("   ✅ Erfolg!")
except Exception as e:
    print(f"   ❌ Fehler: {type(e).__name__}: {e}")
    errors.append(("joblib.load", str(e)))

# Methode 2: pickle.load
if model is None:
    print("\n   Methode 2: pickle.load()")
    try:
        with open(INPUT_FILE, 'rb') as f:
            model = pickle.load(f)
        print("   ✅ Erfolg!")
    except Exception as e:
        print(f"   ❌ Fehler: {type(e).__name__}: {e}")
        errors.append(("pickle.load", str(e)))

# Methode 3: pickle mit encoding
if model is None:
    print("\n   Methode 3: pickle.load(encoding='latin1')")
    try:
        with open(INPUT_FILE, 'rb') as f:
            model = pickle.load(f, encoding='latin1')
        print("   ✅ Erfolg!")
    except Exception as e:
        print(f"   ❌ Fehler: {type(e).__name__}: {e}")
        errors.append(("pickle latin1", str(e)))

# Methode 4: pickle mit bytes
if model is None:
    print("\n   Methode 4: pickle.load(encoding='bytes')")
    try:
        with open(INPUT_FILE, 'rb') as f:
            model = pickle.load(f, encoding='bytes')
        print("   ✅ Erfolg!")
    except Exception as e:
        print(f"   ❌ Fehler: {type(e).__name__}: {e}")
        errors.append(("pickle bytes", str(e)))

# Methode 5: Datei-Header prüfen
print("\n   Methode 5: Datei-Header analysieren")
try:
    with open(INPUT_FILE, 'rb') as f:
        header = f.read(100)
        print(f"   Erste 50 Bytes (hex): {header[:50].hex()}")
        print(f"   Erste 50 Bytes (raw): {header[:50]}")
        
        # Prüfe auf bekannte Formate
        if header[:2] == b'\x1f\x8b':
            print("   → GZIP komprimiert!")
        elif header[:2] == b'BZ':
            print("   → BZIP2 komprimiert!")
        elif header[:2] == b'\x80\x03':
            print("   → Pickle Protocol 3")
        elif header[:2] == b'\x80\x04':
            print("   → Pickle Protocol 4")
        elif header[:2] == b'\x80\x05':
            print("   → Pickle Protocol 5")
        elif b'sklearn' in header:
            print("   → Enthält 'sklearn' String")
            
except Exception as e:
    print(f"   ❌ Fehler: {e}")

# Methode 6: joblib mit verschiedenen Kompressionen
if model is None:
    print("\n   Methode 6: Datei entpacken falls komprimiert")
    
    # Prüfe ob gzip
    try:
        import gzip
        with gzip.open(INPUT_FILE, 'rb') as f:
            model = joblib.load(f)
        print("   ✅ Erfolg mit gzip!")
    except Exception as e:
        errors.append(("gzip+joblib", str(e)[:50]))
    
    # Prüfe ob die Datei eigentlich mehrere Objekte enthält
    if model is None:
        try:
            with open(INPUT_FILE, 'rb') as f:
                objects = []
                while True:
                    try:
                        obj = pickle.load(f)
                        objects.append(obj)
                        print(f"   Objekt gefunden: {type(obj)}")
                    except EOFError:
                        break
                if objects:
                    model = objects[0]
                    print(f"   ✅ {len(objects)} Objekte gefunden!")
        except Exception as e:
            errors.append(("multi-pickle", str(e)[:50]))

# Ergebnis
print("\n" + "="*60)
if model is not None:
    print("✅ MODELL ERFOLGREICH GELADEN!")
    print(f"   Typ: {type(model)}")
    
    # Test
    if hasattr(model, 'predict'):
        print("\n   Funktionstest:")
        try:
            # Features ermitteln
            n_feat = getattr(model, 'n_features_in_', 
                           getattr(model, 'n_features_', 512))
            
            test1 = np.zeros((1, n_feat))
            test2 = np.ones((1, n_feat)) * 5
            test3 = np.random.randn(1, n_feat)
            
            p1 = model.predict_proba(test1)[0][1]
            p2 = model.predict_proba(test2)[0][1]
            p3 = model.predict_proba(test3)[0][1]
            
            print(f"   Nullen:  P(Fall) = {p1:.3f}")
            print(f"   Fünfen:  P(Fall) = {p2:.3f}")
            print(f"   Zufall:  P(Fall) = {p3:.3f}")
            
            if abs(p1 - p2) > 0.01 or abs(p2 - p3) > 0.01:
                print("   ✅ Modell reagiert unterschiedlich!")
            else:
                print("   ⚠️  Modell gibt immer gleiche Werte!")
        except Exception as e:
            print(f"   Test-Fehler: {e}")
    
    # Speichern
    print(f"\n   Speichere als: {OUTPUT_FILE}")
    try:
        joblib.dump(model, OUTPUT_FILE)
        print("   ✅ Gespeichert!")
    except Exception as e:
        print(f"   ❌ Speichern fehlgeschlagen: {e}")
        
else:
    print("❌ MODELL KONNTE NICHT GELADEN WERDEN!")
    print("\n   Alle Fehler:")
    for method, error in errors:
        print(f"   - {method}: {error[:60]}")
    
    print("\n   MÖGLICHE URSACHEN:")
    print("   1. Datei ist beschädigt")
    print("   2. Datei wurde mit einer anderen Python-Version erstellt")
    print("   3. Datei ist kein sklearn-Modell")
    print("\n   LÖSUNG:")
    print("   → Kontaktiere den Autor der Thesis für eine neue Modell-Datei")
    print("   → ODER trainiere ein neues Modell mit eigenen Daten")

print("="*60)