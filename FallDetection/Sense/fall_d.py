import serial
import time
import ast
import numpy as np
import cv2
import pickle
from pathlib import Path
from collections import defaultdict
from tsmoothie.smoother import KalmanSmoother
from functions2 import PrePipeline, TrackingDetectingMergeProcess, ROIPooling, SubpageInterpolating


# =================================================================
# CHESSBOARD HELPER FUNCTIONS
# =================================================================

def GetChessboard(shape=(24, 32)):
    """Generiert Schachbrettmuster für Subpage-Visualisierung."""
    chessboard = np.indices(shape).sum(axis=0) % 2
    chessboard_inverse = np.where((chessboard == 0) | (chessboard == 1), chessboard ^ 1, chessboard)
    return chessboard, chessboard_inverse


# Globale Schachbrett-Masken
CHESSBOARD_MASK, CHESSBOARD_MASK_INV = GetChessboard()


# =================================================================
# FALL DETECTOR CLASS
# =================================================================

class FallDetector:
    """
    Sturzerkennung basierend auf TADAR Paper.
    
    Formel: H = (h × r) / f
    - h = Pixel-Höhe der BBox
    - r = Entfernung in Metern
    - f = Kalibrierte Brennweite (Default: 160)
    - H = Echte Körperhöhe in Metern
    
    Sturz wird erkannt wenn:
    1. H < height_threshold (Person am Boden)
    2. Gradient < gradient_threshold (schneller Fall)
    3. H_vorher > standing_threshold (war vorher stehend)
    """
    
    def __init__(
        self,
        focal_length=160,           # f - Kalibrierte Brennweite (ANPASSEN!)
        height_threshold=0.5,       # H0 - Schwellwert für "am Boden" (Meter)
        gradient_threshold=-1.0,    # G0 - Schwellwert für "schneller Fall" (m/s)
        standing_threshold=1.0,     # Mindesthöhe um als "stehend" zu gelten (Meter)
        history_size=5,             # Anzahl der gespeicherten Messungen
        cooldown_time=10.0,         # Sekunden nach Alarm ohne neuen Alarm
        min_measurements=3          # Mindestanzahl Messungen vor Erkennung
    ):
        self.focal_length = focal_length
        self.height_threshold = height_threshold
        self.gradient_threshold = gradient_threshold
        self.standing_threshold = standing_threshold
        self.history_size = history_size
        self.cooldown_time = cooldown_time
        self.min_measurements = min_measurements
        
        # Historie pro Person: {person_key: [(H, timestamp), ...]}
        self.height_history = defaultdict(list)
        
        # Letzter Alarm-Zeitpunkt pro Person
        self.last_alarm_time = defaultdict(lambda: 0.0)
        
        # Aktueller Alarm-Status pro Person
        self.alarm_active = defaultdict(lambda: False)
        
        # Statistiken
        self.total_falls_detected = 0
    
    def get_person_key(self, x, y, w, h, grid_size=80):
        """Generiert stabilen Key für Person basierend auf Position."""
        center_x = x + w / 2
        center_y = y + h / 2
        grid_x = int(center_x / grid_size)
        grid_y = int(center_y / grid_size)
        return (grid_x, grid_y)
    
    def calculate_real_height(self, pixel_height, range_meters):
        """
        Berechnet echte Körperhöhe aus Pixel-Höhe und Entfernung.
        
        H = (h × r) / f
        """
        if self.focal_length <= 0 or range_meters <= 0:
            return 0.0
        return (pixel_height * range_meters) / self.focal_length
    
    def update(self, person_key, pixel_height, range_meters, timestamp=None):
        """
        Aktualisiert Historie und prüft auf Sturz.
        
        Returns:
            dict: {
                'is_fall': bool,
                'real_height': float,
                'gradient': float or None,
                'status': str
            }
        """
        if timestamp is None:
            timestamp = time.time()
        
        # Berechne echte Höhe
        real_height = self.calculate_real_height(pixel_height, range_meters)
        
        # Füge zur Historie hinzu
        history = self.height_history[person_key]
        history.append((real_height, timestamp))
        
        # Begrenze Historie
        if len(history) > self.history_size:
            history.pop(0)
        
        # Initialisiere Ergebnis
        result = {
            'is_fall': False,
            'real_height': real_height,
            'gradient': None,
            'status': 'MONITORING'
        }
        
        # Brauchen mindestens min_measurements für Gradient
        if len(history) < self.min_measurements:
            result['status'] = f'COLLECTING ({len(history)}/{self.min_measurements})'
            return result
        
        # Berechne Gradient (Änderungsrate)
        H_current = history[-1][0]
        H_previous = history[-2][0]
        t_current = history[-1][1]
        t_previous = history[-2][1]
        
        dt = t_current - t_previous
        if dt <= 0:
            dt = 0.1  # Fallback
        
        gradient = (H_current - H_previous) / dt
        result['gradient'] = gradient
        
        # Finde maximale Höhe in der Historie (war Person stehend?)
        max_height_in_history = max(h[0] for h in history)
        
        # Prüfe Cooldown
        time_since_last_alarm = timestamp - self.last_alarm_time[person_key]
        in_cooldown = time_since_last_alarm < self.cooldown_time
        
        if in_cooldown:
            result['status'] = f'COOLDOWN ({self.cooldown_time - time_since_last_alarm:.1f}s)'
            return result
        
        # ============================================================
        # STURZ-ERKENNUNG: Alle 3 Bedingungen müssen erfüllt sein
        # ============================================================
        
        condition_1 = real_height < self.height_threshold      # Am Boden
        condition_2 = gradient < self.gradient_threshold       # Schneller Fall
        condition_3 = max_height_in_history > self.standing_threshold  # War stehend

        # Konsolen-Output zur Diagnose
        if real_height < self.height_threshold: # Nur loggen wenn man niedrig ist
            print(f"DEBUG: H={real_height:.2f}/{self.height_threshold} ({condition_1}) | "
                f"G={gradient if gradient else 0:.2f}/{self.gradient_threshold} ({condition_2}) | "
                f"MaxH={max_height_in_history:.2f}/{self.standing_threshold} ({condition_3})")
        
        if condition_1 and condition_2 and condition_3:
            # STURZ ERKANNT!
            result['is_fall'] = True
            result['status'] = '⚠️ FALL DETECTED!'
            self.alarm_active[person_key] = True
            self.last_alarm_time[person_key] = timestamp
            self.total_falls_detected += 1
            
            # Historie zurücksetzen nach Alarm
            self.height_history[person_key] = [(real_height, timestamp)]
        else:
            # Kein Sturz - Status anzeigen
            if real_height < self.height_threshold:
                result['status'] = 'LOW (lying/sitting)'
            elif gradient < -0.5:
                result['status'] = 'MOVING DOWN'
            elif real_height > self.standing_threshold:
                result['status'] = 'STANDING'
            else:
                result['status'] = 'MONITORING'
            
            self.alarm_active[person_key] = False
        
        return result
    
    def reset(self, person_key=None):
        """Setzt Historie zurück (eine Person oder alle)."""
        if person_key is None:
            self.height_history.clear()
            self.last_alarm_time.clear()
            self.alarm_active.clear()
            print("🔄 Fall Detector: Alle Historien zurückgesetzt")
        else:
            self.height_history[person_key] = []
            self.last_alarm_time[person_key] = 0.0
            self.alarm_active[person_key] = False
    
    def get_info_string(self):
        """Gibt Info-String für Overlay zurück."""
        return f"f={self.focal_length} | H0={self.height_threshold}m | G0={self.gradient_threshold}m/s"


# =================================================================
# UART & DATA FUNCTIONS
# =================================================================

def initialize_uart(port='/dev/ttyUSB0', baud_rate=1000000):
    ser = serial.Serial(port, baud_rate, timeout=1)
    if not ser.is_open:
        raise RuntimeError(f"Failed to open serial port {port}")
    print(f"Reading data from {port} at {baud_rate} baud")
    return ser


def preprocess_temperature_data(data_str):
    try:
        dict_data = ast.literal_eval(data_str)
        temperature = np.array(dict_data["temperature"]).reshape((24, 32))
        ambient_temp = dict_data["at"]
        return temperature, ambient_temp
    except (ValueError, KeyError, SyntaxError):
        return None, None


def apply_inferno_colormap_with_chessboard(matrix, subpage_type, expansion_coefficient, temp_min=15.0, temp_max=45.0):
    """Wendet Inferno-Colormap an mit Schachbrettmuster."""
    rows, cols = matrix.shape
    
    norm = (matrix - temp_min) / (temp_max - temp_min)
    norm = np.clip(norm, 0, 1) * 255
    norm = norm.astype(np.uint8)
    
    colored_small = cv2.applyColorMap(norm, cv2.COLORMAP_INFERNO)
    
    if subpage_type == 0:
        mask = CHESSBOARD_MASK_INV
    else:
        mask = CHESSBOARD_MASK
    
    colored_small[mask == 1] = [0, 0, 0]
    
    expanded = cv2.resize(
        colored_small, 
        (cols * expansion_coefficient, rows * expansion_coefficient),
        interpolation=cv2.INTER_NEAREST
    )
    
    return expanded


def apply_inferno_colormap_full(matrix, expansion_coefficient, temp_min=15.0, temp_max=45.0):
    """Wendet Inferno-Colormap an ohne Schachbrettmuster."""
    rows, cols = matrix.shape
    
    norm = (matrix - temp_min) / (temp_max - temp_min)
    norm = np.clip(norm, 0, 1) * 255
    norm = norm.astype(np.uint8)
    
    colored_small = cv2.applyColorMap(norm, cv2.COLORMAP_INFERNO)
    
    expanded = cv2.resize(
        colored_small, 
        (cols * expansion_coefficient, rows * expansion_coefficient),
        interpolation=cv2.INTER_LINEAR
    )
    
    return expanded


def smooth_predictions(buffer, smoother, predict, max_len=10):
    if len(buffer) >= max_len:
        buffer.pop(0)
    buffer.append(predict)
    smoother.smooth(buffer)
    smoothed_pred = smoother.smooth_data[0]
    return np.mean(smoothed_pred[-min(max_len, len(smoothed_pred)):])


# =================================================================
# MAIN
# =================================================================

def main():
    ser = initialize_uart()
    
    # =============================================================
    # KONFIGURATION
    # =============================================================
    expansion_coefficient = 20
    temperature_upper_bound = 37
    valid_region_area_limit = 10
    data_shape = (24, 32)
    resize_dim = (640, 480)

    
    TEMP_MIN = 15.0
    TEMP_MAX = 45.0
    
    # =============================================================
    # FALL DETECTION KONFIGURATION
    # =============================================================
    RANGE_SCALE = 0.593   # 2.0m / 3.43m
    FOCAL_LENGTH = 327    # (278px * 2.0m) / 1.7m
    # FOCAL_LENGTH = 160          # ← HIER ANPASSEN nach Kalibrierung!
    HEIGHT_THRESHOLD = 0.5      # Meter - unter diesem Wert = "am Boden"
    GRADIENT_THRESHOLD = -1.0   # m/s - schneller als das = "Sturz"
    STANDING_THRESHOLD = 1.0    # Meter - über diesem Wert = "stehend"
    
    # =============================================================
    # PIPELINE INITIALISIERUNG
    # =============================================================
    prepipeline = PrePipeline(expansion_coefficient, temperature_upper_bound, buffer_size=10, data_shape=data_shape)
    stage1_process = TrackingDetectingMergeProcess(expansion_coefficient, valid_region_area_limit)
    roi_pooling = ROIPooling((200, 400), 100, 100)
    
    # Model laden
    script_dir = Path(__file__).parent.resolve()
    model_path = script_dir / 'Models' / 'hgbr_range2.sav'
    
    try:
        range_estimator = pickle.load(open(model_path, 'rb'))
    except FileNotFoundError:
        print(f"Warning: Model not found at {model_path}, trying relative path...")
        range_estimator = pickle.load(open('Models/hgbr_range2.sav', 'rb'))
    
    kalman_smoother = KalmanSmoother(component='level_trend', component_noise={'level': 0.0001, 'trend': 0.01})
    buffer_pred = {}
    
    # Fall Detector initialisieren
    fall_detector = FallDetector(
        focal_length=FOCAL_LENGTH,
        height_threshold=HEIGHT_THRESHOLD,
        gradient_threshold=GRADIENT_THRESHOLD,
        standing_threshold=STANDING_THRESHOLD
    )
    
    # Visualisierungs-Modi
    show_chessboard = True
    show_bbox = True
    show_fall_detection = True
    
    print("=" * 60)
    print("  TADAR Live - Fall Detection System")
    print("=" * 60)
    print(f"  Focal Length (f):      {FOCAL_LENGTH} px")
    print(f"  Height Threshold:      {HEIGHT_THRESHOLD} m")
    print(f"  Gradient Threshold:    {GRADIENT_THRESHOLD} m/s")
    print(f"  Standing Threshold:    {STANDING_THRESHOLD} m")
    print("=" * 60)
    print("  Tasten:")
    print("    'c' = Chessboard ein/aus")
    print("    'b' = BBox ein/aus")
    print("    'f' = Fall Detection ein/aus")
    print("    'r' = Reset Fall Detector")
    print("    'q' / ESC = Beenden")
    print("=" * 60)
    
    try:
        while True:
            try:
                data = ser.readline().strip()
            except:
                continue
            
            if not data:
                continue
            
            try:
                msg_str = data.decode('utf-8')
            except UnicodeDecodeError:
                continue
            
            sensor_mat, sensor_at = preprocess_temperature_data(msg_str)
            if sensor_mat is None:
                continue
            
            # Preprocessing
            flipped_mat = np.flip(sensor_mat, 0)
            ira_img, subpage_type, ira_mat = prepipeline.Forward(flipped_mat, sensor_at)
            
            if not isinstance(ira_img, np.ndarray):
                continue
            
            # Detection
            mask, _, filtered_mask_colored, _, _, _, valid_bboxes, valid_timers = stage1_process.Forward(ira_img)
            
            # Visualisierung
            interpolated_mat = SubpageInterpolating(flipped_mat)
            
            if show_chessboard:
                ira_colored = apply_inferno_colormap_with_chessboard(
                    interpolated_mat, subpage_type, expansion_coefficient, TEMP_MIN, TEMP_MAX
                )
            else:
                ira_colored = apply_inferno_colormap_full(
                    interpolated_mat, expansion_coefficient, TEMP_MIN, TEMP_MAX
                )
            
            ira_colored = cv2.resize(ira_colored, resize_dim)
            
            # Depth map
            depth_map = np.zeros_like(filtered_mask_colored, dtype=float)
            
            # Fall Detection Status für Overlay
            fall_status_texts = []
            any_fall_detected = False
            
            # =============================================================
            # BBOX PROCESSING + FALL DETECTION
            # =============================================================
            for idx, (x, y, w, h) in enumerate(valid_bboxes):
                if not (100 < (x + w / 2) < 500):
                    continue
                
                roi_t = ira_mat[y:y + h, x:x + w]
                
                if roi_t.size == 0:
                    continue
                
                # Range Prediction
                pooled_roi = roi_pooling.PoolingNumpy(roi_t)
                input_data = np.concatenate([np.sort(pooled_roi.flatten())[::-1][:8], [x + w / 2, y + h / 2]])
                
                predict_r = range_estimator.predict(input_data.reshape(1, -1))[0]
                predict_r = predict_r * RANGE_SCALE
                
                if idx in buffer_pred:
                    predict = smooth_predictions(buffer_pred[idx], kalman_smoother, predict_r)
                else:
                    buffer_pred[idx] = [predict_r]
                    predict = predict_r
                
                # =============================================================
                # FALL DETECTION
                # =============================================================
                fall_result = None
                if show_fall_detection:
                    person_key = fall_detector.get_person_key(x, y, w, h)
                    fall_result = fall_detector.update(
                        person_key=person_key,
                        pixel_height=h,
                        range_meters=predict
                    )
                    
                    if fall_result['is_fall']:
                        any_fall_detected = True
                        print(f"\n🚨 STURZ ERKANNT! H={fall_result['real_height']:.2f}m, Gradient={fall_result['gradient']:.2f}m/s")
                
                # =============================================================
                # BBOX ZEICHNEN
                # =============================================================
                if show_bbox:
                    # Farbe basierend auf Fall-Status
                    if fall_result and fall_result['is_fall']:
                        box_color = (0, 0, 255)      # ROT bei Sturz
                        text_color = (0, 0, 255)
                    elif fall_result and fall_result['real_height'] < HEIGHT_THRESHOLD:
                        box_color = (0, 165, 255)    # ORANGE wenn niedrig
                        text_color = (0, 165, 255)
                    else:
                        box_color = (0, 255, 0)      # GRÜN normal
                        text_color = (0, 255, 255)
                    
                    cv2.rectangle(ira_colored, (x, y), (x + w, y + h), box_color, 2)
                    
                    # Text: Range + echte Höhe
                    if show_fall_detection and fall_result:
                        text = f"R:{predict:.1f}m H:{fall_result['real_height']:.2f}m"
                    else:
                        text = f"{predict:.2f}m"
                    
                    text_x, text_y = x, y - 10
                    
                    (text_w, text_h), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                    cv2.rectangle(ira_colored, (text_x - 2, text_y - text_h - 5), (text_x + text_w + 2, text_y + 5), (0, 0, 0), -1)
                    cv2.putText(ira_colored, text, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.6, color=text_color, thickness=2)
                    
                    # Status unter der Box
                    if show_fall_detection and fall_result:
                        status_text = fall_result['status']
                        cv2.putText(ira_colored, status_text, (x, y + h + 20), cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.5, color=text_color, thickness=1)
                # ... (vorheriger Code in der Schleife)

                # Neue Ausgabe für Pixelhöhe
                # print(f"BBox #{idx}: Pixelhöhe h = {h} Pixel | Position: x={x}, y={y}, w={w} | Distanz-Schätzung: {predict:.2f}m")

                # ... (restlicher Code in der Schleife)
                # Depth map
                center_pt = (int(y + h / 2), int(x + w / 2))
                masks = stage1_process.detector.RegionDivid(mask)
                
                for m in masks:
                    if center_pt[0] < m.shape[0] and center_pt[1] < m.shape[1]:
                        if m[center_pt[0], center_pt[1]] > 0.1:
                            depth_map += m * predict
                            break
            
            # =============================================================
            # ALARM OVERLAY
            # =============================================================
            if any_fall_detected:
                # Großer Alarm-Text
                cv2.putText(
                    ira_colored, 
                    "!!! FALL DETECTED !!!", 
                    (resize_dim[0]//2 - 180, resize_dim[1]//2), 
                    cv2.FONT_HERSHEY_SIMPLEX, 
                    1.2, 
                    (0, 0, 255), 
                    3
                )
                # Roter Rahmen
                cv2.rectangle(ira_colored, (5, 5), (resize_dim[0]-5, resize_dim[1]-5), (0, 0, 255), 4)
            
            # Terminal-Ausgabe
            print(f"\r📊 Persons: {len(valid_bboxes)} | Falls detected: {fall_detector.total_falls_detected}", end="", flush=True)
            
            # Depth colormap
            depth_map = np.where(depth_map < 0.1, 4.5, depth_map)
            depth_norm = ((depth_map / 4.5) * 255).astype(np.uint8)
            depth_colormap = cv2.applyColorMap(depth_norm, cv2.COLORMAP_INFERNO)
            
            # Info-Overlay
            info_line1 = f"Chess:{'ON' if show_chessboard else 'OFF'} | BBox:{'ON' if show_bbox else 'OFF'} | Fall:{'ON' if show_fall_detection else 'OFF'} | Persons:{len(valid_bboxes)}"
            cv2.putText(ira_colored, info_line1, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            if show_fall_detection:
                info_line2 = f"f={FOCAL_LENGTH} | H<{HEIGHT_THRESHOLD}m | G<{GRADIENT_THRESHOLD}m/s | Total Falls:{fall_detector.total_falls_detected}"
                cv2.putText(ira_colored, info_line2, (10, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
            
            # Kombiniertes Bild
            combined_image = np.hstack((ira_colored, depth_colormap))
            
            cv2.imshow('TADAR Live - Fall Detection', combined_image)
            
            # =============================================================
            # TASTENEINGABE
            # =============================================================
            key = cv2.waitKey(1) & 0xFF
            if key in {27, ord('q')}:
                break
            elif key == ord('c'):
                show_chessboard = not show_chessboard
                print(f"\nChessboard: {'ON' if show_chessboard else 'OFF'}")
            elif key == ord('b'):
                show_bbox = not show_bbox
                print(f"\nBBox: {'ON' if show_bbox else 'OFF'}")
            elif key == ord('f'):
                show_fall_detection = not show_fall_detection
                print(f"\nFall Detection: {'ON' if show_fall_detection else 'OFF'}")
            elif key == ord('r'):
                fall_detector.reset()
                print("\n🔄 Fall Detector zurückgesetzt")
    
    finally:
        ser.close()
        cv2.destroyAllWindows()
        print(f"\n\n✅ Cleanup complete. Total falls detected: {fall_detector.total_falls_detected}")


if __name__ == "__main__":
    main()