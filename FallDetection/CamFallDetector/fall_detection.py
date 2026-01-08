#!/usr/bin/env python3
"""
STURZERKENNUNG MIT MEDIAPIPE
============================
Verbesserte Version mit:
- Besserer Fall-Algorithmus (Geschwindigkeit + Position)
- Visuelle Anzeige des Status
- Konfigurierbare Parameter
- Optional: Ohne Gesichtserkennung für einfacheren Start

Installation:
    pip install opencv-python mediapipe numpy

Ausführen:
    python fall_detection_mediapipe.py
"""

import cv2
import numpy as np
import mediapipe as mp
from time import time
from collections import deque
from datetime import datetime

# === KONFIGURATION ===
CAMERA_INDEX = 0                    # Webcam Index (0 = Standard)
FALL_THRESHOLD_RATIO = 1.3          # Schultern fallen um 30% der Körperhöhe
FALL_VELOCITY_THRESHOLD = 100       # Pixel pro Sekunde
CHECK_INTERVAL = 0.1                # Prüfe alle 100ms
HISTORY_SIZE = 20                   # Speichere letzte 20 Positionen
ALARM_COOLDOWN = 5                  # Sekunden zwischen Alarmen

# === MEDIAPIPE SETUP ===
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

# Landmark-Indizes
LEFT_SHOULDER = 11
RIGHT_SHOULDER = 12
LEFT_HIP = 23
RIGHT_HIP = 24
NOSE = 0


class FallDetector:
    def __init__(self):
        self.pose = mp_pose.Pose(
            static_image_mode=False,
            model_complexity=1,          # 0=lite, 1=full, 2=heavy
            smooth_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        
        # Tracking-Variablen
        self.shoulder_history = deque(maxlen=HISTORY_SIZE)
        self.time_history = deque(maxlen=HISTORY_SIZE)
        self.initial_shoulder_height = None
        self.initial_body_height = None
        self.last_alarm_time = 0
        self.fall_detected = False
        self.standing_reference = None
        
    def get_landmarks(self, frame):
        """Extrahiert Pose-Landmarks aus dem Frame"""
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.pose.process(rgb_frame)
        
        if not results.pose_landmarks:
            return None, None
        
        h, w, _ = frame.shape
        landmarks = []
        
        for lm in results.pose_landmarks.landmark:
            landmarks.append({
                'x': int(lm.x * w),
                'y': int(lm.y * h),
                'z': lm.z,
                'visibility': lm.visibility
            })
        
        return landmarks, results.pose_landmarks
    
    def calculate_metrics(self, landmarks):
        """Berechnet relevante Metriken aus den Landmarks"""
        if landmarks is None:
            return None
        
        # Schulterhöhe (Durchschnitt links/rechts)
        left_shoulder = landmarks[LEFT_SHOULDER]
        right_shoulder = landmarks[RIGHT_SHOULDER]
        avg_shoulder_y = (left_shoulder['y'] + right_shoulder['y']) / 2
        
        # Hüfthöhe
        left_hip = landmarks[LEFT_HIP]
        right_hip = landmarks[RIGHT_HIP]
        avg_hip_y = (left_hip['y'] + right_hip['y']) / 2
        
        # Körperhöhe (Schulter zu Hüfte)
        body_height = abs(avg_hip_y - avg_shoulder_y)
        
        # Kopfposition
        nose_y = landmarks[NOSE]['y']
        
        return {
            'shoulder_y': avg_shoulder_y,
            'hip_y': avg_hip_y,
            'nose_y': nose_y,
            'body_height': body_height,
            'timestamp': time()
        }
    
    def detect_fall(self, metrics):
        """
        Erkennt einen Sturz basierend auf:
        1. Schnelle Abwärtsbewegung der Schultern
        2. Schultern fallen unter einen Schwellwert
        """
        if metrics is None:
            return False, "Keine Person erkannt"
        
        current_time = metrics['timestamp']
        shoulder_y = metrics['shoulder_y']
        
        # Speichere Historie
        self.shoulder_history.append(shoulder_y)
        self.time_history.append(current_time)
        
        # Brauchen mindestens 5 Datenpunkte
        if len(self.shoulder_history) < 5:
            return False, "Sammle Daten..."
        
        # Setze Referenz wenn Person steht
        if self.standing_reference is None:
            # Nehme Durchschnitt der ersten Messungen als Referenz
            self.standing_reference = np.mean(list(self.shoulder_history)[:5])
            self.initial_body_height = metrics['body_height']
            return False, "Referenz gesetzt"
        
        # === FALL-ERKENNUNG ===
        
        # 1. Geschwindigkeit berechnen (Pixel pro Sekunde)
        if len(self.shoulder_history) >= 3:
            dy = self.shoulder_history[-1] - self.shoulder_history[-3]
            dt = self.time_history[-1] - self.time_history[-3]
            velocity = dy / dt if dt > 0 else 0
        else:
            velocity = 0
        
        # 2. Position relativ zur Referenz
        height_drop = shoulder_y - self.standing_reference
        drop_ratio = height_drop / self.initial_body_height if self.initial_body_height > 0 else 0
        
        # 3. Alarm-Cooldown prüfen
        time_since_alarm = current_time - self.last_alarm_time
        
        # === FALL-KRITERIEN ===
        # Sturz erkannt wenn:
        # - Schultern sind signifikant gefallen (>30% der Körperhöhe)
        # - ODER schnelle Abwärtsbewegung (>100 px/s)
        
        is_fallen = drop_ratio > FALL_THRESHOLD_RATIO
        is_falling_fast = velocity > FALL_VELOCITY_THRESHOLD
        
        status = f"Drop: {drop_ratio:.1%} | Vel: {velocity:.0f}px/s"
        
        if (is_fallen or is_falling_fast) and time_since_alarm > ALARM_COOLDOWN:
            self.last_alarm_time = current_time
            self.fall_detected = True
            return True, f"STURZ! {status}"
        
        # Reset wenn Person wieder steht
        if drop_ratio < 0.2:
            self.fall_detected = False
        
        return False, status
    
    def reset_reference(self):
        """Setzt die Referenz zurück (z.B. wenn Person aufsteht)"""
        self.standing_reference = None
        self.shoulder_history.clear()
        self.time_history.clear()
        self.fall_detected = False
        print("Referenz zurückgesetzt")
    
    def draw_visualization(self, frame, landmarks_raw, metrics, fall_status, status_text):
        """Zeichnet Visualisierung auf das Frame"""
        h, w, _ = frame.shape
        
        # Pose-Landmarks zeichnen
        if landmarks_raw:
            mp_drawing.draw_landmarks(
                frame,
                landmarks_raw,
                mp_pose.POSE_CONNECTIONS,
                landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style()
            )
        
        # Status-Box oben
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (w, 100), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.5, frame, 0.5, 0, frame)
        
        # Status-Text
        if fall_status:
            status_color = (0, 0, 255)  # Rot
            status_label = "STURZ ERKANNT!"
            cv2.rectangle(frame, (0, 0), (w, 100), (0, 0, 255), 5)
        else:
            status_color = (0, 255, 0)  # Grün
            status_label = "Normal"
        
        cv2.putText(frame, status_label, (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, status_color, 3)
        cv2.putText(frame, status_text, (20, 80),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        
        # Referenzlinie zeichnen
        if self.standing_reference:
            ref_y = int(self.standing_reference)
            cv2.line(frame, (0, ref_y), (w, ref_y), (255, 255, 0), 2)
            cv2.putText(frame, "Referenz", (10, ref_y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
            
            # Threshold-Linie
            if self.initial_body_height:
                thresh_y = int(self.standing_reference + self.initial_body_height * FALL_THRESHOLD_RATIO)
                cv2.line(frame, (0, thresh_y), (w, thresh_y), (0, 0, 255), 2)
                cv2.putText(frame, "Fall-Threshold", (10, thresh_y - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
        
        # Aktuelle Schulterhöhe
        if metrics:
            shoulder_y = int(metrics['shoulder_y'])
            cv2.circle(frame, (w // 2, shoulder_y), 10, (0, 255, 255), -1)
        
        # Anleitung unten
        cv2.putText(frame, "[R] Reset Referenz | [Q] Beenden", (10, h - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        
        return frame


def main():
    print("="*60)
    print("  STURZERKENNUNG MIT MEDIAPIPE")
    print("="*60)
    print("""
  Anleitung:
  1. Stell dich vor die Kamera (Oberkörper sichtbar)
  2. Warte bis "Referenz gesetzt" erscheint
  3. Die gelbe Linie zeigt deine normale Schulterhöhe
  4. Die rote Linie ist der Fall-Threshold
  
  Tasten:
    [R] = Referenz neu setzen (wenn du aufstehst)
    [Q] = Beenden
    """)
    print("="*60)
    
    # Kamera öffnen
    cap = cv2.VideoCapture(CAMERA_INDEX)
    if not cap.isOpened():
        print(f"Kamera {CAMERA_INDEX} konnte nicht geöffnet werden!")
        return
    
    # Auflösung setzen
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    
    print(f"Kamera geöffnet")
    
    detector = FallDetector()
    last_check = 0
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Frame-Fehler")
                continue
            
            # Spiegeln für natürlicheres Gefühl
            frame = cv2.flip(frame, 1)
            
            # Landmarks erkennen
            landmarks, landmarks_raw = detector.get_landmarks(frame)
            
            # Metriken berechnen
            metrics = detector.calculate_metrics(landmarks)
            
            # Fall-Erkennung (mit Intervall)
            current_time = time()
            fall_status = False
            status_text = "Warte auf Person..."
            
            if current_time - last_check >= CHECK_INTERVAL:
                fall_status, status_text = detector.detect_fall(metrics)
                last_check = current_time
                
                if fall_status:
                    timestamp = datetime.now().strftime("%H:%M:%S")
                    print(f"\n{'!'*50}")
                    print(f"STURZ ERKANNT! [{timestamp}]")
                    print(f"{'!'*50}\n")
            
            # Visualisierung
            frame = detector.draw_visualization(
                frame, landmarks_raw, metrics, 
                detector.fall_detected, status_text
            )
            
            cv2.imshow('Fall Detection', frame)
            
            # Tastatur
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q') or key == 27:  # Q oder ESC
                break
            elif key == ord('r'):
                detector.reset_reference()
    
    except KeyboardInterrupt:
        print("\nBeendet")
    
    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()