#!/usr/bin/env python3
"""
STURZERKENNUNG MIT MEDIAPIPE + TELEGRAM ALARM
==============================================
Bei Sturzerkennung wird eine Telegram-Nachricht gesendet.

Installation:
    pip install opencv-python mediapipe numpy requests

Ausführen:
    python fall_detection_telegram.py
"""

import cv2
import numpy as np
import mediapipe as mp
import requests
from time import time
from collections import deque
from datetime import datetime
import threading

# ============================================================
#                    TELEGRAM KONFIGURATION
# ============================================================
# HIER DEINE DATEN EINTRAGEN:

TELEGRAM_BOT_TOKEN = "8506000918:AAE0hgUR4W7YOrNZUYn_YvogN0syOKXwfRQ"
TELEGRAM_CHAT_ID = "5393970914"

# ============================================================

# === FALL DETECTION KONFIGURATION ===
CAMERA_INDEX = 0
FALL_THRESHOLD_RATIO = 1.3
FALL_VELOCITY_THRESHOLD = 100
CHECK_INTERVAL = 0.1
HISTORY_SIZE = 20
ALARM_COOLDOWN = 30  # 30 Sekunden zwischen Telegram-Nachrichten

# === MEDIAPIPE SETUP ===
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

LEFT_SHOULDER = 11
RIGHT_SHOULDER = 12
LEFT_HIP = 23
RIGHT_HIP = 24
NOSE = 0


def send_telegram_message(message):
    """Sendet eine Nachricht über Telegram (nicht-blockierend)"""
    def send():
        try:
            url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
            data = {
                "chat_id": TELEGRAM_CHAT_ID,
                "text": message,
                "parse_mode": "HTML"
            }
            response = requests.post(url, data=data, timeout=10)
            if response.status_code == 200:
                print("✅ Telegram-Nachricht gesendet!")
            else:
                print(f"❌ Telegram-Fehler: {response.text}")
        except Exception as e:
            print(f"❌ Telegram-Fehler: {e}")
    
    # Sende in separatem Thread um Video nicht zu blockieren
    thread = threading.Thread(target=send)
    thread.start()


def test_telegram():
    """Testet die Telegram-Verbindung"""
    print("📱 Teste Telegram-Verbindung...")
    try:
        url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
        data = {
            "chat_id": TELEGRAM_CHAT_ID,
            "text": "🔔 <b>Sturzerkennung aktiviert!</b>\n\nDas System ist jetzt aktiv und überwacht.",
            "parse_mode": "HTML"
        }
        response = requests.post(url, data=data, timeout=10)
        if response.status_code == 200:
            print("✅ Telegram funktioniert!")
            return True
        else:
            print(f"❌ Telegram-Fehler: {response.json()}")
            return False
    except Exception as e:
        print(f"❌ Telegram-Fehler: {e}")
        return False


class FallDetector:
    def __init__(self):
        self.pose = mp_pose.Pose(
            static_image_mode=False,
            model_complexity=1,
            smooth_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        
        self.shoulder_history = deque(maxlen=HISTORY_SIZE)
        self.time_history = deque(maxlen=HISTORY_SIZE)
        self.initial_shoulder_height = None
        self.initial_body_height = None
        self.last_alarm_time = 0
        self.fall_detected = False
        self.standing_reference = None
        
    def get_landmarks(self, frame):
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
        if landmarks is None:
            return None
        
        left_shoulder = landmarks[LEFT_SHOULDER]
        right_shoulder = landmarks[RIGHT_SHOULDER]
        avg_shoulder_y = (left_shoulder['y'] + right_shoulder['y']) / 2
        
        left_hip = landmarks[LEFT_HIP]
        right_hip = landmarks[RIGHT_HIP]
        avg_hip_y = (left_hip['y'] + right_hip['y']) / 2
        
        body_height = abs(avg_hip_y - avg_shoulder_y)
        nose_y = landmarks[NOSE]['y']
        
        return {
            'shoulder_y': avg_shoulder_y,
            'hip_y': avg_hip_y,
            'nose_y': nose_y,
            'body_height': body_height,
            'timestamp': time()
        }
    
    def detect_fall(self, metrics):
        if metrics is None:
            return False, "Keine Person erkannt"
        
        current_time = metrics['timestamp']
        shoulder_y = metrics['shoulder_y']
        
        self.shoulder_history.append(shoulder_y)
        self.time_history.append(current_time)
        
        if len(self.shoulder_history) < 5:
            return False, "Sammle Daten..."
        
        if self.standing_reference is None:
            self.standing_reference = np.mean(list(self.shoulder_history)[:5])
            self.initial_body_height = metrics['body_height']
            return False, "Referenz gesetzt"
        
        # Geschwindigkeit berechnen
        if len(self.shoulder_history) >= 3:
            dy = self.shoulder_history[-1] - self.shoulder_history[-3]
            dt = self.time_history[-1] - self.time_history[-3]
            velocity = dy / dt if dt > 0 else 0
        else:
            velocity = 0
        
        # Position relativ zur Referenz
        height_drop = shoulder_y - self.standing_reference
        drop_ratio = height_drop / self.initial_body_height if self.initial_body_height > 0 else 0
        
        time_since_alarm = current_time - self.last_alarm_time
        
        is_fallen = drop_ratio > FALL_THRESHOLD_RATIO
        is_falling_fast = velocity > FALL_VELOCITY_THRESHOLD
        
        status = f"Drop: {drop_ratio:.1%} | Vel: {velocity:.0f}px/s"
        
        if (is_fallen or is_falling_fast) and time_since_alarm > ALARM_COOLDOWN:
            self.last_alarm_time = current_time
            self.fall_detected = True
            
            # === TELEGRAM ALARM ===
            timestamp = datetime.now().strftime("%d.%m.%Y %H:%M:%S")
            message = (
                f"🚨 <b>STURZ ERKANNT!</b> 🚨\n\n"
                f"⏰ Zeit: {timestamp}\n"
                f"📊 Drop: {drop_ratio:.1%}\n"
                f"💨 Geschwindigkeit: {velocity:.0f} px/s\n\n"
                f"Bitte überprüfen Sie die Person!"
            )
            send_telegram_message(message)
            
            return True, f"STURZ! {status}"
        
        if drop_ratio < 0.2:
            self.fall_detected = False
        
        return False, status
    
    def reset_reference(self):
        self.standing_reference = None
        self.shoulder_history.clear()
        self.time_history.clear()
        self.fall_detected = False
        print("Referenz zurückgesetzt")
    
    def draw_visualization(self, frame, landmarks_raw, metrics, fall_status, status_text):
        h, w, _ = frame.shape
        
        if landmarks_raw:
            mp_drawing.draw_landmarks(
                frame,
                landmarks_raw,
                mp_pose.POSE_CONNECTIONS,
                landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style()
            )
        
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (w, 100), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.5, frame, 0.5, 0, frame)
        
        if fall_status:
            status_color = (0, 0, 255)
            status_label = "STURZ ERKANNT! (Telegram gesendet)"
            cv2.rectangle(frame, (0, 0), (w, 100), (0, 0, 255), 5)
        else:
            status_color = (0, 255, 0)
            status_label = "Normal"
        
        cv2.putText(frame, status_label, (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, status_color, 3)
        cv2.putText(frame, status_text, (20, 80),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        
        if self.standing_reference:
            ref_y = int(self.standing_reference)
            cv2.line(frame, (0, ref_y), (w, ref_y), (255, 255, 0), 2)
            cv2.putText(frame, "Referenz", (10, ref_y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
            
            if self.initial_body_height:
                thresh_y = int(self.standing_reference + self.initial_body_height * FALL_THRESHOLD_RATIO)
                cv2.line(frame, (0, thresh_y), (w, thresh_y), (0, 0, 255), 2)
                cv2.putText(frame, "Fall-Threshold", (10, thresh_y - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
        
        if metrics:
            shoulder_y = int(metrics['shoulder_y'])
            cv2.circle(frame, (w // 2, shoulder_y), 10, (0, 255, 255), -1)
        
        cv2.putText(frame, "[R] Reset | [T] Test Telegram | [Q] Beenden", (10, h - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        
        return frame


def main():
    print("="*60)
    print("  STURZERKENNUNG MIT TELEGRAM ALARM")
    print("="*60)
    
    # Prüfe Telegram-Konfiguration
    if "DEIN_BOT_TOKEN_HIER" in TELEGRAM_BOT_TOKEN:
        print("\n❌ FEHLER: Bitte trage deinen Bot Token ein!")
        print("   Öffne die Datei und ersetze TELEGRAM_BOT_TOKEN")
        return
    
    if "DEINE_CHAT_ID_HIER" in TELEGRAM_CHAT_ID:
        print("\n❌ FEHLER: Bitte trage deine Chat ID ein!")
        print("   Öffne die Datei und ersetze TELEGRAM_CHAT_ID")
        return
    
    # Teste Telegram
    if not test_telegram():
        print("\n⚠️  Telegram-Test fehlgeschlagen! Prüfe Token und Chat ID.")
        response = input("Trotzdem fortfahren? (j/n): ")
        if response.lower() != 'j':
            return
    
    print("""
  Tasten:
    [R] = Referenz neu setzen
    [T] = Telegram Test-Nachricht
    [Q] = Beenden
    """)
    print("="*60)
    
    cap = cv2.VideoCapture(CAMERA_INDEX)
    if not cap.isOpened():
        print(f"❌ Kamera {CAMERA_INDEX} konnte nicht geöffnet werden!")
        return
    
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    
    print("✅ Kamera geöffnet")
    
    detector = FallDetector()
    last_check = 0
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                continue
            
            frame = cv2.flip(frame, 1)
            
            landmarks, landmarks_raw = detector.get_landmarks(frame)
            metrics = detector.calculate_metrics(landmarks)
            
            current_time = time()
            fall_status = False
            status_text = "Warte auf Person..."
            
            if current_time - last_check >= CHECK_INTERVAL:
                fall_status, status_text = detector.detect_fall(metrics)
                last_check = current_time
                
                if fall_status:
                    timestamp = datetime.now().strftime("%H:%M:%S")
                    print(f"\n{'!'*50}")
                    print(f"  ⚠️  STURZ ERKANNT! [{timestamp}]")
                    print(f"  📱 Telegram-Nachricht gesendet!")
                    print(f"{'!'*50}\n")
            
            frame = detector.draw_visualization(
                frame, landmarks_raw, metrics,
                detector.fall_detected, status_text
            )
            
            cv2.imshow('Fall Detection + Telegram', frame)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q') or key == 27:
                break
            elif key == ord('r'):
                detector.reset_reference()
            elif key == ord('t'):
                send_telegram_message("🔔 <b>Test-Nachricht</b>\n\nDie Sturzerkennung funktioniert!")
    
    except KeyboardInterrupt:
        print("\nBeendet")
    
    finally:
        # Abschlussnachricht
        send_telegram_message("🔴 Sturzerkennung wurde beendet.")
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()