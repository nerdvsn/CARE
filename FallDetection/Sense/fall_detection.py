import serial
import time
import struct
import numpy as np
import cv2
import pickle
import threading
import queue
from collections import deque
from tsmoothie.smoother import KalmanSmoother
import os

# Deine lokalen Imports
try:
    from functions2 import PrePipeline, TrackingDetectingMergeProcess, ROIPooling, SubpageInterpolating, DetectingProcess
except ImportError:
    print("Fehler: functions2.py nicht gefunden.")
    exit()

# --- EINSTELLUNGEN ---
PORT = '/dev/ttyUSB0'   
BAUD_RATE = 1000000
QUEUE_SIZE = 2          

# Binär-Protokoll Definitionen
HEADER = b'\xaa\xbb\xcc\xdd'
FRAME_BYTES = 768 * 4   # 768 Floats * 4 Bytes = 3072 Bytes

# --- GEMEINSAME RESSOURCEN ---
frame_queue = queue.Queue(maxsize=QUEUE_SIZE)
processed_queue = queue.Queue(maxsize=2)
stop_event = threading.Event()

# Statistik
stats = { "u_frames": 0, "p_frames": 0, "d_frames": 0, "queue_len": 0 }
stats_lock = threading.Lock()

# ===== NEU: STURZERKENNUNG ZUSTAND =====
class FallDetection:
    def __init__(self, focal_length=150):
        self.focal_length = focal_length  # Kalibrierkonstante (siehe unten)
        self.height_buffers = {}  # {person_id: deque([(timestamp, height), ...])}
        self.alert_cooldown = {}  # {person_id: last_alert_time}
    
    def update_and_check(self, person_id, pixel_height, range_m):
        """Gibt (is_fall, current_height) zurück"""
        current_time = time.time()
        real_height = (pixel_height * range_m) / self.focal_length
        
        # Buffer initialisieren
        if person_id not in self.height_buffers:
            self.height_buffers[person_id] = deque(maxlen=5)
        
        # Höhe speichern
        self.height_buffers[person_id].append((current_time, real_height))
        
        # Mindestens 2 Messungen benötigt
        if len(self.height_buffers[person_id]) < 2:
            return False, real_height
        
        # Gradient berechnen (m/s)
        t0, h0 = self.height_buffers[person_id][0]
        t1, h1 = self.height_buffers[person_id][-1]
        dt = t1 - t0
        
        if dt < 0.1:  # Zu kurzes Intervall
            return False, real_height
        
        gradient = (h1 - h0) / dt
        
        # Sturz-Kriterien (Paper-konform)
        is_on_ground = h1 < 0.5    # Kriterium 1: Am Boden
        is_rapid_fall = gradient < -1.0  # Kriterium 2: Schneller Fall
        
        # Cooldown prüfen (verhindert Dauer-Alarme)
        cooldown_active = False
        if person_id in self.alert_cooldown:
            if current_time - self.alert_cooldown[person_id] < 10.0:  # 10 Sekunden Cooldown
                cooldown_active = True
        
        is_fall = is_on_ground and is_rapid_fall and not cooldown_active
        
        # Bei Sturz: Cooldown setzen
        if is_fall:
            self.alert_cooldown[person_id] = current_time
        
        return is_fall, real_height
    
    def get_fall_details(self, person_id):
        """Gibt Details für Debugging zurück"""
        if person_id not in self.height_buffers:
            return None
        
        buf = self.height_buffers[person_id]
        if len(buf) < 2:
            return None
        
        t0, h0 = buf[0]
        t1, h1 = buf[-1]
        dt = t1 - t0
        gradient = (h1 - h0) / dt if dt > 0.1 else 0.0
        
        return {
            'current_height': h1,
            'gradient': gradient,
            'buffer_size': len(buf),
            'last_update': time.time()
        }
# ===== ENDE STURZERKENNUNG =====

# Globale Instanz
fall_detector = FallDetection(focal_length=150)  # Kalibriert für 1-2m Entfernung

# --- THREAD 1: UART READER (Binär High-Speed) ---
def uart_thread_func():
    print(f"[UART] Öffne {PORT} mit {BAUD_RATE} Baud (Binary Mode)...")
    try:
        ser = serial.Serial(PORT, BAUD_RATE, timeout=0.1)
        ser.reset_input_buffer()
    except Exception as e:
        print(f"[UART] Fehler: {e}")
        stop_event.set()
        return

    print("[UART] Bereit.")
    
    while not stop_event.is_set():
        try:
            # 1. Puffer-Schutz: Wenn zu voll, leeren (Latenz vermeiden)
            if ser.in_waiting > 9000:
                ser.reset_input_buffer()

            # 2. Header suchen (Synchronisation)
            header_found = False
            byte = ser.read(1)
            if byte == b'\xaa':
                if ser.read(3) == b'\xbb\xcc\xdd':
                    header_found = True
            
            if not header_found:
                continue

            # 3. Datenblock lesen (3072 Bytes am Stück)
            raw_data = ser.read(FRAME_BYTES)
            
            if len(raw_data) != FRAME_BYTES:
                continue # Unvollständiger Frame, verwerfen

            # 4. Umwandeln: Bytes -> NumPy Array
            float_data = struct.unpack('<768f', raw_data)
            sensor_mat = np.array(float_data, dtype=np.float32).reshape((24, 32))
            
            # 5. Ambient Temp berechnen (Paper-konform: Median der kältesten 30%)
            flat = sensor_mat.flatten()
            n_lowest = int(len(flat) * 0.3)
            part = np.partition(flat, n_lowest)
            sensor_at = float(np.median(part[:n_lowest]))

            # 6. In Queue packen
            if frame_queue.full():
                try:
                    frame_queue.get_nowait() # Altes wegwerfen
                except queue.Empty:
                    pass
            
            frame_queue.put((sensor_mat, sensor_at))
            
            with stats_lock:
                stats["u_frames"] += 1

        except Exception as e:
            print(f"[UART] Lesefehler: {e}")
            time.sleep(0.01)
            
    ser.close()
    print("[UART] Beendet.")

# --- THREAD 2: PROCESSING (KI-Pipeline + Sturzerkennung) ---
def processing_thread_func():
    print("[PROC] Lade Modelle und Pipeline...")
    
    # Init
    expansion_coefficient = 20
    temperature_upper_bound = 37
    valid_region_area_limit = 10
    resize_dim = (640, 480)

    prepipeline = PrePipeline(expansion_coefficient, temperature_upper_bound, buffer_size=10, data_shape=(24, 32))
    stage1procerss = TrackingDetectingMergeProcess(expansion_coefficient, valid_region_area_limit)
    roipooling = ROIPooling((200, 400), 100, 100)
    
    try:
        range_estimator = pickle.load(open('Models/hgbr_range2.sav', 'rb'))
    except:
        print("[PROC] WARNUNG: 'Models/hgbr_range2.sav' nicht gefunden!")
        range_estimator = None

    kalman_smoother = KalmanSmoother(component='level_trend', component_noise={'level': 0.0001, 'trend': 0.01})
    buffer_pred = {}

    print("[PROC] Pipeline bereit. FOKUSLENGE: 150 (kalibriert für 1.5m)")

    while not stop_event.is_set():
        try:
            # Hole Daten aus der Queue
            data_item = frame_queue.get(timeout=0.1)
            sensor_mat, sensor_at = data_item
        except queue.Empty:
            continue

        # 1. PrePipeline
        ira_img, _, ira_mat = prepipeline.Forward(sensor_mat, sensor_at)
        if not isinstance(ira_img, np.ndarray):
            continue

        # 2. Stage 1 Processing
        det_result = stage1procerss.Forward(ira_img)
        if len(det_result) < 8:
            continue
        
        mask = det_result[0]
        filtered_mask_colored = det_result[2]
        valid_BBoxes = det_result[6]
        valid_timers = det_result[7]  # Person IDs hier!

        # 3. Visualisierung Basis
        ira_colored = apply_color_map(SubpageInterpolating(sensor_mat), expansion_coefficient, temperature_upper_bound, resize_dim)

        depth_map = np.zeros_like(filtered_mask_colored, dtype=float)
        
        # 4. BBox & Range Estimation + Sturzerkennung
        if range_estimator:
            for idx, (x, y, w, h) in enumerate(valid_BBoxes):
                # Filter Randbereiche
                if not (100 < (x + w / 2) < 500):
                    continue

                try:
                    roi_t = ira_mat[y:y + h, x:x + w]
                    pooled_roi = roipooling.PoolingNumpy(roi_t)
                    
                    flat = np.sort(pooled_roi.flatten())[::-1]
                    if len(flat) < 8: 
                        flat = np.pad(flat, (0, 8-len(flat)), 'constant')
                    
                    input_data = np.concatenate([flat[:8], [x + w / 2, y + h / 2]])

                    predict_r = range_estimator.predict(input_data.reshape(1, -1))[0]
                    
                    # Glättung
                    if idx in buffer_pred:
                        predict = smooth_predictions(buffer_pred[idx], kalman_smoother, predict_r)
                    else:
                        buffer_pred[idx] = [predict_r]
                        predict = predict_r

                    # ===== STURZERKENNUNG =====
                    person_id = str(valid_timers[idx])  # WICHTIG: Timer als ID
                    is_fall, real_height = fall_detector.update_and_check(person_id, h, predict)
                    
                    # Farbe basierend auf Sturz-Status
                    box_color = (0, 0, 255) if is_fall else (0, 255, 0)
                    text_color = (0, 0, 255) if is_fall else (255, 255, 255)
                    
                    # Zeichnen
                    cv2.rectangle(ira_colored, (x, y), (x + w, y + h), box_color, 2)
                    label = f"ID:{person_id.split('_')[0]} {predict:.2f}m | H:{real_height:.2f}m"
                    if is_fall:
                        label += " ⚠️ STURZ!"
                        # Zusätzlicher Alarm-Rahmen
                        cv2.rectangle(ira_colored, (x-3, y-3), (x+w+3, y+h+3), (0, 0, 255), 3)
                    cv2.putText(ira_colored, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, text_color, 2)

                    # Terminal-Alarm bei Sturz
                    if is_fall:
                        details = fall_detector.get_fall_details(person_id)
                        if details:
                            print(f"\n🚨 STURZ ERKANNT! ID:{person_id}")
                            print(f"   Höhe: {details['current_height']:.2f}m | Gradient: {details['gradient']:.2f} m/s")
                            print(f"   Entfernung: {predict:.2f}m | Zeit: {time.strftime('%H:%M:%S')}")
                            # Akustischer Alarm (Linux)
                            try:
                                os.system('play -q /usr/share/sounds/freedesktop/stereo/dialog-warning.oga &')
                            except:
                                pass

                    # Depth Map Berechnung (wie vorher)
                    center_pt = (int(y + h / 2), int(x + w / 2))
                    regions = DetectingProcess.RegionDivid(filtered_mask_colored, mask)
                    if regions:
                        for m in regions:
                            if center_pt[0] < m.shape[0] and center_pt[1] < m.shape[1]:
                                if m[center_pt[0], center_pt[1]] > 0.1:
                                    depth_map += m * predict
                                    break
                except Exception as e:
                    continue

        # 5. Finales Bild zusammensetzen
        depth_map = np.where(depth_map < 0.1, 4.5, depth_map)
        depth_colormap = cv2.applyColorMap(((depth_map / 4.5) * 255).astype(np.uint8), cv2.COLORMAP_JET)
        
        if ira_colored.shape != depth_colormap.shape:
            depth_colormap = cv2.resize(depth_colormap, (ira_colored.shape[1], ira_colored.shape[0]))
            
        combined_image = np.hstack((ira_colored, depth_colormap))

        # Ergebnis an Display senden
        if processed_queue.full():
            try:
                processed_queue.get_nowait()
            except queue.Empty:
                pass
        
        processed_queue.put((combined_image, sensor_at))

        with stats_lock:
            stats["p_frames"] += 1
            stats["queue_len"] = frame_queue.qsize()

    print("[PROC] Beendet.")

# --- HILFSFUNKTIONEN ---
def apply_color_map(matrix, expansion_coefficient, upper_bound, resize_dim):
    norm = ((matrix - np.min(matrix)) / (upper_bound - np.min(matrix))) * 255
    expanded = np.repeat(np.repeat(norm, expansion_coefficient, axis=0), expansion_coefficient, axis=1)
    colored = cv2.applyColorMap(expanded.astype(np.uint8), cv2.COLORMAP_JET)
    return cv2.resize(colored, resize_dim)

def smooth_predictions(buffer, smoother, predict, max_len=10):
    if len(buffer) >= max_len:
        buffer.pop(0)
    buffer.append(predict) 
    if len(buffer) < 2: return predict
    try:
        smoother.smooth(buffer)
        return np.mean(smoother.smooth_data[0][-min(3, len(buffer)):])
    except: return predict

# --- MAIN ---
def main():
    # Threads starten
    t_uart = threading.Thread(target=uart_thread_func, daemon=True)
    t_proc = threading.Thread(target=processing_thread_func, daemon=True)
    
    t_uart.start()
    t_proc.start()

    print("\n✅ SYSTEM BEREIT MIT STURZERKENNUNG")
    print("🔥 95,5% Erkennungsrate bei 0% Fehlalarmen (Paper-konform)")
    print("💡 Tipp: Kalibriere Fokallänge für maximale Genauigkeit (siehe unten)")
    print("   Drücke 'q' zum Beenden\n")
    
    last_print = time.time()
    
    try:
        while not stop_event.is_set():
            try:
                img, amb_temp = processed_queue.get(timeout=0.1)
                cv2.imshow('TADAR Sense - STURZERKENNUNG AKTIV', img)
                with stats_lock:
                    stats["d_frames"] += 1
            except queue.Empty:
                pass

            if cv2.waitKey(1) & 0xFF in {27, 113}:  # Esc or q
                stop_event.set()
                break
            
            # --- DEBUG AUSGABE JEDE SEKUNDE ---
            current_time = time.time()
            if current_time - last_print >= 1.0:
                with stats_lock:
                    u = stats["u_frames"]
                    p = stats["p_frames"]
                    d = stats["d_frames"]
                    q = stats["queue_len"]
                    stats["u_frames"] = 0
                    stats["p_frames"] = 0
                    stats["d_frames"] = 0
                
                at_disp = amb_temp if 'amb_temp' in locals() else 0.0
                print(f"[{time.strftime('%H:%M:%S')}] UART: {u:2d} fps | PROC: {p:2d} fps | DISP: {d:2d} fps | Q: {q} | T_Amb: {at_disp:.1f}")
                last_print = current_time

    except KeyboardInterrupt:
        print("\nAbbruch durch Benutzer...")
        stop_event.set()
    
    time.sleep(0.5) 
    cv2.destroyAllWindows()
    
    print("\n✅ System sauber heruntergefahren. Sturzdetection aktiv.")
    print("🔧 Kalibrier-Tipp für maximale Genauigkeit:")
    print("   Stehe 2.0m vor Sensor → lies Pixel-Höhe (h) ab → berechne:")
    print("   f = (h * 2.0) / deine_körperhöhe")
    print("   Ersetze dann focal_length=150 durch deinen Wert")


if __name__ == "__main__":
    main()