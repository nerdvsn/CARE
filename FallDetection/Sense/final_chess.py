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


def SubpageType(mat, chessboard):
    """Bestimmt den Subpage-Typ (0 oder 1)."""
    subpage0 = mat * chessboard
    if np.sum(subpage0) < 1:
        return 1
    return 0


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


def apply_color_map(matrix, expansion_coefficient, upper_bound, resize_dim):
    min_val = np.min(matrix)
    if upper_bound - min_val == 0:
        norm = np.zeros_like(matrix)
    else:
        norm = ((matrix - min_val) / (upper_bound - min_val)) * 255
    norm = np.clip(norm, 0, 255)
    expanded = np.repeat(np.repeat(norm, expansion_coefficient, axis=0), expansion_coefficient, axis=1)
    colored = cv2.applyColorMap(expanded.astype(np.uint8), cv2.COLORMAP_HOT)
    return cv2.resize(colored, resize_dim)


def apply_color_map_with_chessboard(matrix, chessboard_mask, expansion_coefficient, upper_bound, resize_dim, min_temp=15.0):
    """
    Wendet Colormap an und visualisiert das Schachbrettmuster.
    Nicht-gelesene Pixel werden grau dargestellt.
    """
    # Normalisierung
    min_val = np.min(matrix)
    if upper_bound - min_val == 0:
        norm = np.zeros_like(matrix)
    else:
        norm = ((matrix - min_val) / (upper_bound - min_val)) * 255
    norm = np.clip(norm, 0, 255).astype(np.uint8)
    
    # Expansion
    expanded = np.repeat(np.repeat(norm, expansion_coefficient, axis=0), expansion_coefficient, axis=1)
    expanded_mask = np.repeat(np.repeat(chessboard_mask, expansion_coefficient, axis=0), expansion_coefficient, axis=1)
    
    # Colormap anwenden
    colored = cv2.applyColorMap(expanded, cv2.COLORMAP_HOT)
    
    # Nicht-gelesene Pixel grau markieren (wo mask == 1)
    gray_value = 50  # Dunkelgrau für nicht-gelesene Pixel
    colored[expanded_mask == 1] = [gray_value, gray_value, gray_value]
    
    return cv2.resize(colored, resize_dim)


def smooth_predictions(buffer, smoother, predict, max_len=10):
    if len(buffer) >= max_len:
        buffer.pop(0)
    buffer.append(predict)
    
    if len(buffer) < 2:
        return predict
    
    try:
        smoother.smooth(buffer)
        smoothed_pred = smoother.smooth_data[0]
        return np.mean(smoothed_pred[-min(max_len, len(smoothed_pred)):])
    except Exception:
        return predict


def get_bbox_key(x, y, w, h, grid_size=50):
    """Generate a stable key for bounding box based on center position."""
    center_x = x + w / 2
    center_y = y + h / 2
    grid_x = int(center_x / grid_size)
    grid_y = int(center_y / grid_size)
    return (grid_x, grid_y)


# =================================================================
# MAIN
# =================================================================

def main():
    ser = initialize_uart()
    
    # Konfiguration
    expansion_coefficient = 20
    temperature_upper_bound = 37
    valid_region_area_limit = 5
    data_shape = (24, 32)
    resize_dim = (640, 480)
    
    # Pipeline initialisieren
    prepipeline = PrePipeline(expansion_coefficient, temperature_upper_bound, buffer_size=10, data_shape=data_shape)
    stage1_process = TrackingDetectingMergeProcess(expansion_coefficient, valid_region_area_limit)
    
    # ROI Pooling
    roi_pooling_size = (2, 4)
    resize_shape = (100 * roi_pooling_size[0], 100 * roi_pooling_size[1])
    window_size = 100
    roi_pooling = ROIPooling(resize_shape, window_size, window_size)
    topk = roi_pooling_size[0] * roi_pooling_size[1]  # = 8
    
    # Model laden
    script_dir = Path(__file__).parent.resolve()
    model_path = script_dir / 'Models' / 'hgbr_range2.sav'
    
    try:
        range_estimator = pickle.load(open(model_path, 'rb'))
    except FileNotFoundError:
        print(f"Error: Model not found at {model_path}")
        # Fallback auf relativen Pfad
        range_estimator = pickle.load(open('Models/hgbr_range2.sav', 'rb'))
    
    kalman_smoother = KalmanSmoother(component='level_trend', component_noise={'level': 0.0001, 'trend': 0.01})
    buffer_pred = {}
    
    # Chessboard-Visualisierung an/aus
    show_chessboard = True  # Auf False setzen um Schachbrett auszublenden
    
    print("Start!!")
    print("Press 'c' to toggle chessboard visualization")
    print("Press 'q' or 'Esc' to quit")
    
    try:
        while True:
            try:
                data = ser.readline().strip()
            except serial.SerialException:
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
            
            # Interpolierte Matrix für Visualisierung
            interpolated_mat = SubpageInterpolating(flipped_mat)
            
            # Colormap mit oder ohne Chessboard
            if show_chessboard:
                # Bestimme welche Pixel aktuell gelesen wurden
                if subpage_type == 0:
                    # Subpage 0 gelesen -> Subpage 1 (inverse) ist nicht aktuell
                    current_mask = CHESSBOARD_MASK_INV
                else:
                    # Subpage 1 gelesen -> Subpage 0 ist nicht aktuell
                    current_mask = CHESSBOARD_MASK
                
                ira_colored = apply_color_map_with_chessboard(
                    interpolated_mat, 
                    current_mask,
                    expansion_coefficient, 
                    temperature_upper_bound, 
                    resize_dim
                )
            else:
                ira_colored = apply_color_map(
                    interpolated_mat, 
                    expansion_coefficient, 
                    temperature_upper_bound, 
                    resize_dim
                )
            
            # Depth map initialisieren
            depth_map = np.zeros_like(filtered_mask_colored, dtype=float)
            
            for idx, (x, y, w, h) in enumerate(valid_bboxes):
                # Skip ungültige Boxen
                if w <= 0 or h <= 0:
                    continue
                
                center_x = x + w / 2
                if not (100 < center_x < 500):
                    continue
                
                # ROI extrahieren mit Bounds-Check
                y1, y2 = int(y), int(y + h)
                x1, x2 = int(x), int(x + w)
                
                if y1 < 0 or x1 < 0 or y2 > ira_mat.shape[0] or x2 > ira_mat.shape[1]:
                    continue
                
                roi_t = ira_mat[y1:y2, x1:x2]
                
                if roi_t.size == 0:
                    continue
                
                # ROI Pooling & Prediction
                pooled_roi = roi_pooling.PoolingNumpy(roi_t)
                flat_data = np.reshape(np.array([pooled_roi]), (1, -1))
                sort_flat_data = np.sort(flat_data, axis=-1)[:, ::-1]
                
                center_point = (center_x, y + h / 2)
                center_data = np.reshape(np.array([center_point]), (1, -1))
                input_data = np.concatenate((sort_flat_data[:, :topk], center_data), axis=1)
                
                predict_r = range_estimator.predict(input_data)[0]
                
                # Smoothing mit stabilem Key
                bbox_key = get_bbox_key(x, y, w, h)
                if bbox_key in buffer_pred:
                    predict = smooth_predictions(buffer_pred[bbox_key], kalman_smoother, predict_r)
                else:
                    buffer_pred[bbox_key] = [predict_r]
                    predict = predict_r
                
                # Bounding Box zeichnen
                cv2.rectangle(ira_colored, (int(x), int(y)), (int(x + w), int(y + h)), (0, 255, 0), 2)
                cv2.putText(
                    ira_colored, 
                    f"{round(predict, 2)}m", 
                    (int(x), int(y) - 10), 
                    cv2.FONT_HERSHEY_SIMPLEX,
                    fontScale=1.2, 
                    color=(0, 0, 255), 
                    thickness=3
                )
                
                # Depth map aktualisieren
                center_pt = (int(y + h / 2), int(x + w / 2))
                
                # RegionDivid korrekt aufrufen (nur masks, keine areas)
                masks = stage1_process.detector.RegionDivid(mask)
                
                for m in masks:
                    if center_pt[0] < m.shape[0] and center_pt[1] < m.shape[1]:
                        if m[center_pt[0], center_pt[1]] > 0.1:
                            depth_map += m * predict
                            break
            
            # Depth colormap erstellen
            depth_map = np.where(depth_map < 0.1, 4.5, depth_map)
            depth_colormap = cv2.applyColorMap(((depth_map / 4.5) * 255).astype(np.uint8), cv2.COLORMAP_JET)
            
            # Info-Text für Chessboard-Status
            status_text = f"Chessboard: {'ON' if show_chessboard else 'OFF'} | Subpage: {subpage_type}"
            cv2.putText(ira_colored, status_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            # Kombiniertes Bild
            combined_image = np.hstack((ira_colored, depth_colormap))
            
            cv2.imshow('TADAR Live', combined_image)
            
            key = cv2.waitKey(1) & 0xFF
            if key in {27, ord('q')}:  # ESC oder 'q'
                break
            elif key == ord('c'):  # 'c' zum Umschalten
                show_chessboard = not show_chessboard
                print(f"Chessboard visualization: {'ON' if show_chessboard else 'OFF'}")
    
    finally:
        ser.close()
        cv2.destroyAllWindows()
        print("Cleanup complete")


if __name__ == "__main__":
    main()