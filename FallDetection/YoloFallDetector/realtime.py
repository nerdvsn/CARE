from ultralytics import YOLO
import cv2
import time
import sys

# ====================================================================
# 1. KONFIGURATION FALL DETECTION
# ====================================================================
MODEL_PATH = 'YOLOv8_Fall_Detection/finetune_v1/weights/best.pt'
CAMERA_SOURCE = 0
CONFIDENCE_THRESHOLD = 0.50
IOU_THRESHOLD = 0.70
ALARM_COOLDOWN = 2  # Sekunden zwischen Alarmen
# ====================================================================

# 2. MODELL LADEN
try:
    model = YOLO(MODEL_PATH)
    print(f"✅ Fall Detection Modell erfolgreich geladen.")
    print(f"📊 Modellklassen: {model.names}")
except Exception as e:
    print(f"❌ Fehler beim Laden des Modells: {e}")
    sys.exit(1)

# 3. KAMERA INITIALISIEREN
cap = cv2.VideoCapture(CAMERA_SOURCE)
if not cap.isOpened():
    print(f"❌ Fehler: Konnte Kamera {CAMERA_SOURCE} nicht öffnen.")
    sys.exit(1)

# Kameraeigenschaften setzen für bessere Performance
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
cap.set(cv2.CAP_PROP_FPS, 30)

print(f"▶️ Starte Fall Detection auf Kamera {CAMERA_SOURCE}")
print("🚨 Drücken Sie 'q', um zu beenden.")
print("🔊 Fall-Erkennungen werden als ALARM angezeigt")

# Variablen für Alarm-Handling
last_alarm_time = 0
fall_count = 0
frame_count = 0
start_time = time.time()

# 4. HAUPTSCHLEIFE
try:
    while True:
        ret, frame = cap.read()
        if not ret:
            print("❌ Fehler beim Lesen des Kameraframes.")
            break
            
        frame_count += 1
        
        # INFERENZ DURCHFÜHREN
        results = model.predict(
            source=frame, 
            conf=CONFIDENCE_THRESHOLD,
            iou=IOU_THRESHOLD,
            verbose=False
        )
        
        # ERGEBNISSE VERARBEITEN
        for r in results:
            annotated_frame = r.plot()
            
            # FALL DETECTION LOGIK
            current_time = time.time()
            boxes = r.boxes
            
            if len(boxes) > 0:
                for box in boxes:
                    confidence = float(box.conf[0])
                    class_id = int(box.cls[0])
                    class_name = model.names[class_id]
                    
                    # Alarm nur auslösen wenn Cooldown vorbei
                    if current_time - last_alarm_time > ALARM_COOLDOWN:
                        fall_count += 1
                        last_alarm_time = current_time
                        print(f"🚨 ALARM! {class_name} erkannt - Konfidenz: {confidence:.2f} - Fall #{fall_count}")
            
            # STATISTIKEN UND INFO ZUM FRAME HINZUFÜGEN
            fps = frame_count / (current_time - start_time)
            
            # Hintergrund für Text
            cv2.rectangle(annotated_frame, (5, 5), (300, 120), (0, 0, 0), -1)
            
            # FPS anzeigen
            cv2.putText(annotated_frame, f"FPS: {fps:.1f}", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            # Fall Count anzeigen
            cv2.putText(annotated_frame, f"Faelle: {fall_count}", (10, 60), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            
            # Aktuelle Detektionen anzeigen
            if len(boxes) > 0:
                status_text = f"ERKENNT: {len(boxes)} Fall(s)"
                cv2.putText(annotated_frame, status_text, (10, 90), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                
                # Roten Rahmen um gesamtes Bild bei Erkennung
                cv2.rectangle(annotated_frame, (0, 0), 
                             (annotated_frame.shape[1], annotated_frame.shape[0]), 
                             (0, 0, 255), 10)
            else:
                status_text = "Bereit - Keine Faelle"
                cv2.putText(annotated_frame, status_text, (10, 90), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            # Fenstertitel mit Status
            window_title = f"Fall Detection - {len(boxes)} Fall(s) erkannt" if len(boxes) > 0 else "Fall Detection - Bereit"
            
            # Frame anzeigen
            cv2.imshow(window_title, annotated_frame)
        
        # BEENDEN BEI 'q' DRUCK
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
            
except KeyboardInterrupt:
    print("\n⏹️  Programm durch Benutzer beendet")
except Exception as e:
    print(f"❌ Unerwarteter Fehler: {e}")

# 5. RESSOURCEN FREIGEBEN UND STATISTIKEN
cap.release()
cv2.destroyAllWindows()

# ZUSAMMENFASSUNG
total_time = time.time() - start_time
print("\n📊 ZUSAMMENFASSUNG:")
print(f"   Gesamtzeit: {total_time:.1f} Sekunden")
print(f"   Verarbeitete Frames: {frame_count}")
print(f"   Durchschnittliche FPS: {frame_count/total_time:.1f}")
print(f"   Erkannte Faelle: {fall_count}")
print("✅ Sitzung beendet.")