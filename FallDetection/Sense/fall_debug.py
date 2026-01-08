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
    Vereinfachte Sturzerkennung - trackt die größte Person im Bild.
    
    Formel: H = (h × r) / f
    
    Sturz wird erkannt wenn:
    1. Aktuelle max H < height_threshold (Person am Boden)
    2. In Historie gab es H > standing_threshold (war stehend)
    3. Der Übergang war schnell genug
    """
    
    def __init__(
        self,
        focal_length=160,
        height_threshold=2.0,       # H unter diesem Wert = "am Boden"
        standing_threshold=4.0,     # H über diesem Wert = "stehend"
        history_seconds=3.0,        # Sekunden Historie
        cooldown_time=10.0,
        min_fall_speed=1.5          # Meter pro Sekunde (H-Änderung)
    ):
        self.focal_length = focal_length
        self.height_threshold = height_threshold
        self.standing_threshold = standing_threshold
        self.history_seconds = history_seconds
        self.cooldown_time = cooldown_time
        self.min_fall_speed = min_fall_speed
        
        # GLOBALE Historie (nicht pro Person!)
        # Format: [(max_H_in_frame, timestamp), ...]
        self.height_history = []
        self.last_alarm_time = 0.0
        self.total_falls_detected = 0
        
    def calculate_real_height(self, pixel_height, range_meters):
        """H = (h × r) / f"""
        if self.focal_length <= 0 or range_meters <= 0:
            return 0.0
        return (pixel_height * range_meters) / self.focal_length
    
    def update(self, detections, timestamp=None):
        """
        Aktualisiert mit allen Detektionen eines Frames.
        
        Args:
            detections: Liste von (pixel_height, range_meters) Tupeln
            timestamp: Zeitstempel (optional)
        
        Returns:
            dict mit Ergebnis
        """
        if timestamp is None:
            timestamp = time.time()
        
        # Finde die größte H in diesem Frame
        current_max_H = 0.0
        for pixel_height, range_meters in detections:
            H = self.calculate_real_height(pixel_height, range_meters)
            if H > current_max_H:
                current_max_H = H
        
        # Zur Historie hinzufügen
        self.height_history.append((current_max_H, timestamp))
        
        # Alte Einträge entfernen (älter als history_seconds)
        cutoff_time = timestamp - self.history_seconds
        self.height_history = [(h, t) for h, t in self.height_history if t >= cutoff_time]
        
        # Initialisiere Ergebnis
        result = {
            'is_fall': False,
            'current_max_H': current_max_H,
            'history_max_H': current_max_H,
            'history_min_H': current_max_H,
            'fall_speed': 0.0,
            'status': 'MONITORING'
        }
        
        if len(self.height_history) < 3:
            result['status'] = f'COLLECTING ({len(self.height_history)}/3)'
            return result
        
        # Analysiere Historie
        heights = [h for h, t in self.height_history]
        times = [t for h, t in self.height_history]
        
        history_max_H = max(heights)
        history_min_H = min(heights)
        
        result['history_max_H'] = history_max_H
        result['history_min_H'] = history_min_H
        
        # Berechne durchschnittliche Fallgeschwindigkeit
        if len(heights) >= 2:
            total_time = times[-1] - times[0]
            if total_time > 0:
                # Von max zu aktuell
                fall_speed = (history_max_H - current_max_H) / total_time
                result['fall_speed'] = fall_speed
        
        # Prüfe Cooldown
        time_since_alarm = timestamp - self.last_alarm_time
        if time_since_alarm < self.cooldown_time:
            result['status'] = f'COOLDOWN ({self.cooldown_time - time_since_alarm:.1f}s)'
            return result
        
        # ============================================================
        # STURZ-ERKENNUNG (VEREINFACHT)
        # ============================================================
        
        # Bedingung 1: Aktuelle Höhe ist niedrig
        cond1 = current_max_H < self.height_threshold
        
        # Bedingung 2: War vorher stehend (in Historie)
        cond2 = history_max_H > self.standing_threshold
        
        # Bedingung 3: Signifikanter Höhenunterschied
        height_drop = history_max_H - current_max_H
        cond3 = height_drop > (self.standing_threshold - self.height_threshold) * 0.5
        
        if cond1 and cond2 and cond3:
            result['is_fall'] = True
            result['status'] = '⚠️ FALL DETECTED!'
            self.last_alarm_time = timestamp
            self.total_falls_detected += 1
            # Historie leeren nach Alarm
            self.height_history = [(current_max_H, timestamp)]
        else:
            if current_max_H < self.height_threshold:
                result['status'] = 'LOW'
            elif current_max_H > self.standing_threshold:
                result['status'] = 'STANDING'
            else:
                result['status'] = 'MONITORING'
        
        return result
    
    def reset(self):
        """Setzt alles zurück."""
        self.height_history = []
        self.last_alarm_time = 0.0
        print("🔄 Fall Detector zurückgesetzt")
    
    def get_person_key(self, x, y, w, h, grid_size=80):
        """Dummy für Kompatibilität."""
        return (0, 0)


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
    # FALL DETECTION KONFIGURATION (KALIBRIERT)
    # =============================================================
    RANGE_SCALE = 0.593         # Kalibriert: 2.0m / 3.43m
    FOCAL_LENGTH = 327          # Kalibriert: (278px * 2.0m) / 1.7m
    
    # Mit kalibrierten Werten sollten H-Werte jetzt realistisch sein:
    # - Stehend: H ≈ 1.5-1.8m
    # - Am Boden: H ≈ 0.3-0.5m
    HEIGHT_THRESHOLD = 0.6      # Meter - unter diesem Wert = "am Boden"
    STANDING_THRESHOLD = 1.2    # Meter - über diesem Wert = "stehend"
    GRADIENT_THRESHOLD = -0.4   # m/s - schneller als das = "Sturz"
    
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
    
    # Fall Detector initialisieren (VEREINFACHTER GLOBALER ANSATZ)
    fall_detector = FallDetector(
        focal_length=FOCAL_LENGTH,
        height_threshold=HEIGHT_THRESHOLD,
        standing_threshold=STANDING_THRESHOLD,
        history_seconds=3.0,        # 3 Sekunden Historie
        cooldown_time=10.0
    )
    
    # Visualisierungs-Modi
    show_chessboard = True
    show_bbox = True
    show_fall_detection = True
    
    print("=" * 60)
    print("  TADAR Live - Fall Detection System (CALIBRATED)")
    print("=" * 60)
    print(f"  Range Scale:           {RANGE_SCALE}")
    print(f"  Focal Length (f):      {FOCAL_LENGTH} px")
    print(f"  Height Threshold:      {HEIGHT_THRESHOLD} m (below = on ground)")
    print(f"  Standing Threshold:    {STANDING_THRESHOLD} m (above = standing)")
    print(f"  History Window:        3.0 seconds")
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
            
            # Fall Detection: Sammle alle Detektionen
            all_detections = []
            fall_result = None
            any_fall_detected = False
            
            # =============================================================
            # BBOX PROCESSING
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
                predict_r = predict_r * RANGE_SCALE  # Kalibrierte Skalierung
                
                if idx in buffer_pred:
                    predict = smooth_predictions(buffer_pred[idx], kalman_smoother, predict_r)
                else:
                    buffer_pred[idx] = [predict_r]
                    predict = predict_r
                
                # Sammle für Fall Detection
                if show_fall_detection:
                    all_detections.append((h, predict))  # (pixel_height, range)
                
                # =============================================================
                # BBOX ZEICHNEN
                # =============================================================
                if show_bbox:
                    # Berechne H für diese Person
                    H_this = (h * predict) / FOCAL_LENGTH if FOCAL_LENGTH > 0 else 0
                    
                    # Farbe basierend auf Höhe
                    if H_this < HEIGHT_THRESHOLD:
                        box_color = (0, 165, 255)    # ORANGE wenn niedrig
                        text_color = (0, 165, 255)
                    elif H_this > STANDING_THRESHOLD:
                        box_color = (0, 255, 0)      # GRÜN wenn stehend
                        text_color = (0, 255, 255)
                    else:
                        box_color = (255, 255, 0)    # CYAN dazwischen
                        text_color = (255, 255, 255)
                    
                    cv2.rectangle(ira_colored, (x, y), (x + w, y + h), box_color, 2)
                    
                    # Text: Range + echte Höhe
                    text = f"R:{predict:.1f}m H:{H_this:.1f}m"
                    text_x, text_y = x, y - 10
                    
                    (text_w, text_h), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                    cv2.rectangle(ira_colored, (text_x - 2, text_y - text_h - 5), (text_x + text_w + 2, text_y + 5), (0, 0, 0), -1)
                    cv2.putText(ira_colored, text, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.6, color=text_color, thickness=2)
                
                # Depth map
                center_pt = (int(y + h / 2), int(x + w / 2))
                masks = stage1_process.detector.RegionDivid(mask)
                
                for m in masks:
                    if center_pt[0] < m.shape[0] and center_pt[1] < m.shape[1]:
                        if m[center_pt[0], center_pt[1]] > 0.1:
                            depth_map += m * predict
                            break
            
            # =============================================================
            # FALL DETECTION (nach allen BBoxen)
            # =============================================================
            if show_fall_detection and len(all_detections) > 0:
                fall_result = fall_detector.update(all_detections)
                
                # DEBUG Output
                print(f"\n📏 Detections: {len(all_detections)} | maxH={fall_result['history_max_H']:.2f}m | currH={fall_result['current_max_H']:.2f}m")
                
                cond1 = fall_result['current_max_H'] < HEIGHT_THRESHOLD
                cond2 = fall_result['history_max_H'] > STANDING_THRESHOLD
                height_drop = fall_result['history_max_H'] - fall_result['current_max_H']
                cond3 = height_drop > (STANDING_THRESHOLD - HEIGHT_THRESHOLD) * 0.5
                
                print(f"   ✓ currH<{HEIGHT_THRESHOLD}m: {cond1} ({fall_result['current_max_H']:.2f}m)")
                print(f"   ✓ maxH>{STANDING_THRESHOLD}m: {cond2} ({fall_result['history_max_H']:.2f}m)")
                print(f"   ✓ drop>{(STANDING_THRESHOLD - HEIGHT_THRESHOLD) * 0.5:.1f}m: {cond3} ({height_drop:.2f}m)")
                
                if fall_result['is_fall']:
                    any_fall_detected = True
                    print(f"🚨 STURZ ERKANNT!")
                    
                    # Alle Boxen rot färben
                    for (x, y, w, h) in valid_bboxes:
                        cv2.rectangle(ira_colored, (x, y), (x + w, y + h), (0, 0, 255), 3)
            
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
            
            cv2.imshow('Obnone Live - Fall Detection', combined_image)
            
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