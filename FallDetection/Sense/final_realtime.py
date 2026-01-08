import serial
import time
import ast
import numpy as np
import cv2
import pickle
from pathlib import Path
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
    """
    Wendet Inferno-Colormap an mit echtem Schachbrettmuster.
    Nicht-gelesene Pixel werden schwarz dargestellt.
    """
    rows, cols = matrix.shape
    
    # Normalisierung auf 0-255
    norm = (matrix - temp_min) / (temp_max - temp_min)
    norm = np.clip(norm, 0, 1) * 255
    norm = norm.astype(np.uint8)
    
    # Inferno Colormap anwenden (auf Original-Auflösung)
    colored_small = cv2.applyColorMap(norm, cv2.COLORMAP_INFERNO)
    
    # Schachbrettmuster: Nicht-gelesene Pixel schwarz setzen
    if subpage_type == 0:
        mask = CHESSBOARD_MASK_INV
    else:
        mask = CHESSBOARD_MASK
    
    # Schwarze Pixel für nicht-gelesene Stellen
    colored_small[mask == 1] = [0, 0, 0]
    
    # Expansion OHNE Interpolation (nearest neighbor) für scharfe Pixel
    expanded = cv2.resize(
        colored_small, 
        (cols * expansion_coefficient, rows * expansion_coefficient),
        interpolation=cv2.INTER_NEAREST
    )
    
    return expanded


def apply_inferno_colormap_full(matrix, expansion_coefficient, temp_min=15.0, temp_max=45.0):
    """
    Wendet Inferno-Colormap an ohne Schachbrettmuster (alle Pixel interpoliert).
    """
    rows, cols = matrix.shape
    
    # Normalisierung auf 0-255
    norm = (matrix - temp_min) / (temp_max - temp_min)
    norm = np.clip(norm, 0, 1) * 255
    norm = norm.astype(np.uint8)
    
    # Inferno Colormap anwenden
    colored_small = cv2.applyColorMap(norm, cv2.COLORMAP_INFERNO)
    
    # Expansion mit Interpolation für glatteres Bild
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
    # KONFIGURATION (aus deinem funktionierenden Code)
    # =============================================================
    expansion_coefficient = 20
    temperature_upper_bound = 37
    valid_region_area_limit = 6  # WICHTIG: 10 statt 5 für bessere Detektion
    data_shape = (24, 32)
    resize_dim = (640, 480)
    
    # Temperatur-Bereich für Colormap
    TEMP_MIN = 15.0
    TEMP_MAX = 45.0
    
    # =============================================================
    # PIPELINE (aus deinem funktionierenden Code)
    # =============================================================
    prepipeline = PrePipeline(expansion_coefficient, temperature_upper_bound, buffer_size=10, data_shape=data_shape)
    stage1_process = TrackingDetectingMergeProcess(expansion_coefficient, valid_region_area_limit)
    
    # ROI Pooling - WICHTIG: Original-Konfiguration aus deinem Code
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
    
    # Visualisierungs-Modus
    show_chessboard = True
    show_bbox = True
    
    print("=" * 50)
    print("TADAR Live - Inferno Chessboard Visualization")
    print("=" * 50)
    print("Tasten:")
    print("  'c' = Chessboard ein/aus")
    print("  'b' = BBox ein/aus")
    print("  'q' / ESC = Beenden")
    print("=" * 50)
    
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
            
            # =============================================================
            # PREPROCESSING (aus deinem funktionierenden Code)
            # =============================================================
            flipped_mat = np.flip(sensor_mat, 0)
            ira_img, subpage_type, ira_mat = prepipeline.Forward(flipped_mat, sensor_at)
            
            if not isinstance(ira_img, np.ndarray):
                continue
            
            # =============================================================
            # DETECTION (aus deinem funktionierenden Code)
            # =============================================================
            mask, _, filtered_mask_colored, _, _, _, valid_bboxes, valid_timers = stage1_process.Forward(ira_img)
            
            # =============================================================
            # VISUALISIERUNG (Inferno + Chessboard)
            # =============================================================
            interpolated_mat = SubpageInterpolating(flipped_mat)
            
            if show_chessboard:
                ira_colored = apply_inferno_colormap_with_chessboard(
                    interpolated_mat,
                    subpage_type,
                    expansion_coefficient,
                    TEMP_MIN,
                    TEMP_MAX
                )
            else:
                ira_colored = apply_inferno_colormap_full(
                    interpolated_mat,
                    expansion_coefficient,
                    TEMP_MIN,
                    TEMP_MAX
                )
            
            # Resize für konsistente Ausgabe
            ira_colored = cv2.resize(ira_colored, resize_dim)
            
            # Depth map initialisieren
            depth_map = np.zeros_like(filtered_mask_colored, dtype=float)
            
            # =============================================================
            # BBOX PROCESSING (Logik aus deinem funktionierenden Code)
            # =============================================================
            for idx, (x, y, w, h) in enumerate(valid_bboxes):
                # Filter wie in deinem Original-Code
                if not (100 < (x + w / 2) < 500):
                    continue
                
                # ROI extrahieren (Original-Methode)
                roi_t = ira_mat[y:y + h, x:x + w]
                
                if roi_t.size == 0:
                    continue
                
                # =============================================================
                # PREDICTION (exakt wie in deinem funktionierenden Code)
                # =============================================================
                pooled_roi = roi_pooling.PoolingNumpy(roi_t)
                input_data = np.concatenate([np.sort(pooled_roi.flatten())[::-1][:8], [x + w / 2, y + h / 2]])
                
                predict_r = range_estimator.predict(input_data.reshape(1, -1))[0]
                
                if idx in buffer_pred:
                    predict = smooth_predictions(buffer_pred[idx], kalman_smoother, predict_r)
                else:
                    buffer_pred[idx] = [predict_r]
                    predict = predict_r
                
                # =============================================================
                # BBOX ZEICHNEN (nur wenn aktiviert)
                # =============================================================
                if show_bbox:
                    cv2.rectangle(
                        ira_colored, 
                        (x, y), 
                        (x + w, y + h), 
                        (0, 255, 0),  # Grün
                        2
                    )
                    
                    # Text mit Hintergrund für bessere Lesbarkeit
                    text = f"{predict:.2f}m"
                    text_x, text_y = x, y - 10
                    
                    # Hintergrund für Text
                    (text_w, text_h), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)
                    cv2.rectangle(
                        ira_colored,
                        (text_x - 2, text_y - text_h - 5),
                        (text_x + text_w + 2, text_y + 5),
                        (0, 0, 0),
                        -1
                    )
                    
                    cv2.putText(
                        ira_colored, 
                        text,
                        (text_x, text_y), 
                        cv2.FONT_HERSHEY_SIMPLEX,
                        fontScale=0.8, 
                        color=(0, 255, 255),  # Cyan
                        thickness=2
                    )
                
                # Depth map aktualisieren
                center_pt = (int(y + h / 2), int(x + w / 2))
                masks = stage1_process.detector.RegionDivid(mask)
                
                for m in masks:
                    if center_pt[0] < m.shape[0] and center_pt[1] < m.shape[1]:
                        if m[center_pt[0], center_pt[1]] > 0.1:
                            depth_map += m * predict
                            break
            
            # Terminal-Ausgabe
            print(f"\r📊 Persons detected: {len(valid_bboxes)}", end="", flush=True)
            
            # Depth colormap erstellen (auch Inferno)
            depth_map = np.where(depth_map < 0.1, 4.5, depth_map)
            depth_norm = ((depth_map / 4.5) * 255).astype(np.uint8)
            depth_colormap = cv2.applyColorMap(depth_norm, cv2.COLORMAP_INFERNO)
            
            # Info-Overlay
            info_text = f"Chessboard: {'ON' if show_chessboard else 'OFF'} | BBox: {'ON' if show_bbox else 'OFF'} | Persons: {len(valid_bboxes)}"
            cv2.putText(
                ira_colored, 
                info_text, 
                (10, 25), 
                cv2.FONT_HERSHEY_SIMPLEX, 
                0.6, 
                (255, 255, 255), 
                2
            )
            
            # Kombiniertes Bild
            combined_image = np.hstack((ira_colored, depth_colormap))
            
            cv2.imshow('TADAR Live - Inferno', combined_image)
            
            key = cv2.waitKey(1) & 0xFF
            if key in {27, ord('q')}:
                break
            elif key == ord('c'):
                show_chessboard = not show_chessboard
                print(f"\nChessboard: {'ON' if show_chessboard else 'OFF'}")
            elif key == ord('b'):
                show_bbox = not show_bbox
                print(f"\nBBox: {'ON' if show_bbox else 'OFF'}")
    
    finally:
        ser.close()
        cv2.destroyAllWindows()
        print("\n✅ Cleanup complete")


if __name__ == "__main__":
    main()