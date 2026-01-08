import serial
import time
import json  # Viel schneller als ast für deine Daten!
import numpy as np
import cv2
import pickle
import threading
import queue
from tsmoothie.smoother import KalmanSmoother

# Deine lokalen Imports (müssen im selben Ordner liegen)
try:
    from functions2 import PrePipeline, TrackingDetectingMergeProcess, ROIPooling, SubpageInterpolating, DetectingProcess
except ImportError:
    print("Fehler: functions2.py nicht gefunden.")
    exit()

# --- EINSTELLUNGEN ---
PORT = '/dev/ttyUSB0'   # Ggf. anpassen (z.B. COM3 auf Windows)
BAUD_RATE = 1000000     # Bei JSON-Daten ist das Limit hier ca. 18 FPS
QUEUE_SIZE = 2          # Klein halten für Echtzeit!

# --- GEMEINSAME RESSOURCEN ---
frame_queue = queue.Queue(maxsize=QUEUE_SIZE)
processed_queue = queue.Queue(maxsize=2)
stop_event = threading.Event()

# Statistik-Variablen (Thread-safe via Lock nicht zwingend bei simplen Ints, aber sauberer)
stats = {
    "u_frames": 0,
    "p_frames": 0,
    "d_frames": 0,
    "queue_len": 0
}
stats_lock = threading.Lock()

# --- THREAD 1: UART READER (Holen der Daten) ---
def uart_thread_func():
    print(f"[UART] Öffne {PORT} mit {BAUD_RATE} Baud...")
    try:
        ser = serial.Serial(PORT, BAUD_RATE, timeout=1)
    except Exception as e:
        print(f"[UART] Fehler: {e}")
        stop_event.set()
        return

    print("[UART] Bereit.")
    
    while not stop_event.is_set():
        try:
            # Wir lesen eine ganze Zeile (dein JSON endet mit \n vom ESP32?)
            # Falls der ESP32 kein \n sendet, blockiert das hier bis zum Timeout!
            line = ser.readline()
            
            if not line:
                continue

            try:
                # Decodieren
                data_str = line.decode('utf-8').strip()
            except UnicodeDecodeError:
                continue

            # Einfacher Check: Beginnt es wie dein JSON?
            if not data_str.startswith("{") or not "temperature" in data_str:
                continue

            # Queue-Management: Wenn voll, altes wegwerfen (Drop Oldest)
            if frame_queue.full():
                try:
                    frame_queue.get_nowait()
                except queue.Empty:
                    pass
            
            frame_queue.put(data_str)
            
            with stats_lock:
                stats["u_frames"] += 1

        except Exception as e:
            print(f"[UART] Lesefehler: {e}")
            time.sleep(0.1)
            
    ser.close()
    print("[UART] Beendet.")

# --- THREAD 2: PROCESSING (Deine KI-Pipeline) ---
def processing_thread_func():
    print("[PROC] Lade Modelle und Pipeline...")
    
    # Init
    expansion_coefficient = 20
    temperature_upper_bound = 37
    valid_region_area_limit = 10
    data_shape = (24, 32)
    resize_dim = (640, 480)

    prepipeline = PrePipeline(expansion_coefficient, temperature_upper_bound, buffer_size=10, data_shape=data_shape)
    stage1procerss = TrackingDetectingMergeProcess(expansion_coefficient, valid_region_area_limit)
    roipooling = ROIPooling((200, 400), 100, 100)
    
    try:
        range_estimator = pickle.load(open('Models/hgbr_range2.sav', 'rb'))
    except:
        print("[PROC] WARNUNG: 'Models/hgbr_range2.sav' nicht gefunden!")
        range_estimator = None

    kalman_smoother = KalmanSmoother(component='level_trend', component_noise={'level': 0.0001, 'trend': 0.01})
    buffer_pred = {}

    print("[PROC] Pipeline bereit.")

    while not stop_event.is_set():
        try:
            # Hole den String aus der Queue
            msg_str = frame_queue.get(timeout=0.1)
        except queue.Empty:
            continue

        # --- Parsing (Hier jetzt json statt ast -> Schneller) ---
        try:
            # Dein JSON Format: {"temperature": [...], "at": 20.5}
            dict_data = json.loads(msg_str) 
            
            sensor_mat = np.array(dict_data["temperature"]).reshape(data_shape)
            sensor_at = dict_data["at"]
        except Exception:
            # Falls JSON kaputt ist (passiert bei Serial manchmal)
            continue

        # --- Pipeline Logik ---
        # 1. PrePipeline
        ira_img, _, ira_mat = prepipeline.Forward(np.flip(sensor_mat, 0), sensor_at)
        if not isinstance(ira_img, np.ndarray):
            continue

        # 2. Stage 1 Processing
        mask, _, filtered_mask_colored, _, _, _, valid_BBoxes, valid_timers = stage1procerss.Forward(ira_img)
        
        # 3. Visualisierung Basis
        ira_colored = apply_color_map(SubpageInterpolating(np.flip(sensor_mat, 0)), expansion_coefficient, temperature_upper_bound, resize_dim)

        depth_map = np.zeros_like(filtered_mask_colored, dtype=float)
        
        # 4. BBox & Range Estimation
        if range_estimator:
            for idx, (x, y, w, h) in enumerate(valid_BBoxes):
                # Filter Randbereiche
                if not (100 < (x + w / 2) < 500):
                    continue

                roi_t = ira_mat[y:y + h, x:x + w]
                pooled_roi = roipooling.PoolingNumpy(roi_t)
                # Input für Model vorbereiten
                input_data = np.concatenate([np.sort(pooled_roi.flatten())[::-1][:8], [x + w / 2, y + h / 2]])

                predict_r = range_estimator.predict(input_data.reshape(1, -1))[0]
                
                # Glättung
                if idx in buffer_pred:
                    predict = smooth_predictions(buffer_pred[idx], kalman_smoother, predict_r)
                else:
                    buffer_pred[idx] = [predict_r]
                    predict = predict_r

                # Zeichnen
                cv2.rectangle(ira_colored, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.putText(ira_colored, f"{predict:.2f}m", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)

                # Depth Map Berechnung
                center_pt = (int(y + h / 2), int(x + w / 2))
                for m in DetectingProcess.RegionDivid(filtered_mask_colored, mask):
                    if m[center_pt[0], center_pt[1]] > 0.1:
                        depth_map += m * predict
                        break

        # 5. Finales Bild zusammensetzen
        depth_map = np.where(depth_map < 0.1, 4.5, depth_map)
        depth_colormap = cv2.applyColorMap(((depth_map / 4.5) * 255).astype(np.uint8), cv2.COLORMAP_JET)
        combined_image = np.hstack((ira_colored, depth_colormap))

        # Ergebnis an Display senden
        if processed_queue.full():
            try:
                processed_queue.get_nowait()
            except queue.Empty:
                pass
        processed_queue.put((combined_image, sensor_at, np.mean(sensor_mat)))

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
    smoother.smooth(buffer)
    # Sicherstellen, dass Daten da sind
    if smoother.smooth_data is not None and len(smoother.smooth_data) > 0:
        smoothed_pred = smoother.smooth_data[0]
        return np.mean(smoothed_pred[-min(max_len, len(smoothed_pred)):])
    return predict

# --- MAIN ---
def main():
    # Threads starten
    t_uart = threading.Thread(target=uart_thread_func, daemon=True)
    t_proc = threading.Thread(target=processing_thread_func, daemon=True)
    
    t_uart.start()
    t_proc.start()

    print("System läuft. Warte auf Daten... (Drücke 'q' zum Beenden)")
    
    last_print = time.time()
    
    try:
        while not stop_event.is_set():
            # Versuche Bild zu holen
            try:
                img, amb_temp, mean_temp = processed_queue.get(timeout=0.1)
                
                # Bild anzeigen
                cv2.imshow('Thermal AI', img)
                
                with stats_lock:
                    stats["d_frames"] += 1
                
            except queue.Empty:
                # Kein neues Bild -> loop weiter
                pass

            # Tastenabfrage (muss regelmäßig aufgerufen werden für cv2 Events)
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
                    
                    # Reset Counters
                    stats["u_frames"] = 0
                    stats["p_frames"] = 0
                    stats["d_frames"] = 0
                
                # Wenn wir noch keine Daten haben, zeigen wir Dummy Werte
                at_disp = amb_temp if 'amb_temp' in locals() else 0.0
                mt_disp = mean_temp if 'mean_temp' in locals() else 0.0
                
                print(f"[{time.strftime('%H:%M:%S')}] UART: {u:2d} fps | PROC: {p:2d} fps | DISP: {d:2d} fps | Q: {q} | T_Amb: {at_disp:.1f}")
                last_print = current_time

    except KeyboardInterrupt:
        print("\nAbbruch durch Benutzer...")
        stop_event.set()
    
    # Aufräumen
    stop_event.set()
    # Kurzer Wait damit Threads sich beenden können
    time.sleep(0.5) 
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()