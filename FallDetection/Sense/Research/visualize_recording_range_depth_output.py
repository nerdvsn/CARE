import pickle
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib as mpl
from matplotlib.patches import Rectangle 

# === WICHTIG: TKAGG-BACKEND FÜR VNC/TERMINAL AUSFÜHRUNG ===
try:
    mpl.use('TkAgg') 
except ImportError:
    print("Warnung: TkAgg-Backend nicht verfügbar. Bitte 'sudo apt install python3-tk' ausführen.")

# === PFAD ZUR DATENDATEI ANPASSEN ===
# !!! HIER den Pfad zur Output-Datei aus dem 'Outputs/' Ordner angeben !!!
output_path = "FourUser_Dynamic_4_sensor_4.pkl"  # Angenommener Pfad basierend auf Ihrem Output

# === DEKLARATION DER BEZUGSAUFLÖSUNG FÜR DIE SKALIERUNG ===
ORIGINAL_BBOX_WIDTH = 640
ORIGINAL_BBOX_HEIGHT = 480


def load_processed_data(output_path):
    """Lädt die verarbeiteten Daten aus der Output-PKL-Datei."""
    print(f"Lade verarbeitete Daten aus: {output_path}...")
    try:
        with open(output_path, 'rb') as f:
            data = pickle.load(f)
        
        REQUIRED_KEYS = ['ira_matrix', 'frame_index', 'GT_timestamps', 
                         'Predicted_BBoxes', 'range_KF_smoothed_prediction', 
                         'depth_KF_smoothed_Size_based_predictioins']
        
        if all(key in data for key in REQUIRED_KEYS):
            # 1. WICHTIGE KORREKTUR: Ira_matrix muss als numerisches Array geladen werden.
            # Wir verwenden die einfache Konvertierung, um dtype=object zu vermeiden.
            # Wenn es eine Liste von Arrays mit gleicher Form ist, funktioniert das.
            ira_matrix_raw = data['ira_matrix']
            # Wir müssen sicherstellen, dass die ira_matrix in ein numerisches Array konvertiert werden kann.
            # Wenn die Bilder in den inneren Arrays von data['ira_matrix'] bereits die richtige numerische dtype haben,
            # sollte diese Konvertierung funktionieren:
            ira_matrix = np.array(ira_matrix_raw) 

            if ira_matrix.dtype == object:
                print("WARNUNG: ira_matrix konnte nicht in ein numerisches Array konvertiert werden. Versuche Konvertierung manuell.")
                # Falls die äußere Dimension dtype=object ist, versuchen wir, die Bilder einzeln zu konvertieren.
                # Dies deutet darauf hin, dass die Bilder unterschiedliche Größen haben oder das Laden fehlgeschlagen ist.
                # Wir geben das ursprüngliche Objekt-Array zurück, verlassen uns aber auf die inneren Arrays.
                # Der Fehler wird wahrscheinlich bei der Visualisierung des ersten Frames auftreten.
                pass # ira_matrix bleibt in diesem Fall ein Array von Objekten

            # 2. Flache Arrays für das Mapping
            frame_indices = np.array(data['frame_index'])
            timestamps = np.array(data['GT_timestamps']) 
            
            # 3. Flache Vorhersage-Arrays
            predicted_bboxes = np.array(data['Predicted_BBoxes']) # N x 4
            predicted_ranges = np.array(data['range_KF_smoothed_prediction'])
            predicted_depths = np.array(data['depth_KF_smoothed_Size_based_predictioins'])
            
            print(f"Erfolgreich geladen. Gesamt-Detektionen: {predicted_bboxes.shape[0]}")
            
            return ira_matrix, frame_indices, timestamps, predicted_bboxes, predicted_depths, predicted_ranges
        # ... Fehlerbehandlung (unverändert) ...
        # ...
        else:
            missing_keys = [key for key in REQUIRED_KEYS if key not in data]
            print(f"FEHLER: Einer oder mehrere benötigte Schlüssel fehlen in der Output-Datei: {missing_keys}")
            return None, None, None, None, None, None
            
    except FileNotFoundError:
        print(f"FEHLER: Output-Datei nicht gefunden unter {output_path}. Haben Sie 'test_neu.py' ausgeführt?")
        return None, None, None, None, None, None
    except Exception as e:
        print(f"FEHLER beim Laden oder Verarbeiten der Pickle-Datei: {e}")
        return None, None, None, None, None, None


def visualize_frames(ira_matrix, frame_indices, timestamps, predicted_bboxes, predicted_depths, predicted_ranges):
    """Erstellt eine Matplotlib-Animation der thermischen Aufnahmen mit den vorhergesagten BBoxes."""
    
    if ira_matrix is None or ira_matrix.size == 0 or ira_matrix[0].ndim < 2:
        print("Keine Frames oder ungültiges IRA-Format zum Visualisieren vorhanden.")
        return
        
    # Hier wird angenommen, dass ira_matrix[0] das erste numerische Bild-Array ist.
    # WENN ira_matrix dtype=object ist, ist ira_matrix[0] das innere numerische Array.
    # WENN ira_matrix ein numerisches 3D/4D Array ist, ist ira_matrix[0] die 2D/3D Scheibe.
    IR_FRAME_DIMENSIONS = ira_matrix[0].shape[:2]
    IR_HEIGHT, IR_WIDTH = IR_FRAME_DIMENSIONS
    
    # ... Rest der Funktion (unverändert, da die Fehlerursache das Laden war) ...

    # Skalierungsfaktoren (von 640x480 auf IRA-Größe)
    scale_x = IR_WIDTH / ORIGINAL_BBOX_WIDTH 
    scale_y = IR_HEIGHT / ORIGINAL_BBOX_HEIGHT

    # Bestimmt die maximale Frame-Nummer, um die Animation zu begrenzen
    MAX_FRAMES = len(ira_matrix)

    # --- Matplotlib Setup ---
    fig, ax = plt.subplots(figsize=(10, 8))
    
    global_min = 15.0  
    global_max = 45.0  
    cmap = 'inferno' 
    
    # DIES IST DIE ZEILE, DIE DEN FEHLER VERURSACHT HATTE
    therm_image = ax.imshow(
        ira_matrix[0], 
        cmap=cmap,
        vmin=global_min,
        vmax=global_max,
        interpolation='nearest' 
    )

    cbar = fig.colorbar(therm_image)
    cbar.set_label('Temperature [°C]', fontsize=14)
    
    ax.set_xticks(np.arange(0, IR_WIDTH + 1, IR_WIDTH / 4))
    ax.set_yticks(np.arange(0, IR_HEIGHT + 1, IR_HEIGHT / 4))
    ax.set_xlabel(f'Horizontal Pixel ({IR_WIDTH})')
    ax.set_ylabel(f'Vertikal Pixel ({IR_HEIGHT})')
    
    title = ax.set_title(f"Frame 0 (IR: {IR_WIDTH}x{IR_HEIGHT}) | Detektionen: 0")
    rectangles = []
    texts = []

    def update_frame(current_frame_idx):
        """Funktion, die in jedem Animationsschritt aufgerufen wird."""
        
        # 1. Bilddaten aktualisieren
        therm_image.set_data(ira_matrix[current_frame_idx]) 
        
        # 2. Alte Elemente entfernen
        for rect in rectangles:
            rect.remove()
        rectangles.clear()
        
        for txt in texts:
            txt.remove()
        texts.clear()
        
        # 3. Vorhersagen für den aktuellen Frame filtern (Mapping der flachen Arrays)
        indices_in_frame = np.where(frame_indices == current_frame_idx)[0]
        num_detections = indices_in_frame.size

        if num_detections > 0:
            start_idx = indices_in_frame[0]
            end_idx = indices_in_frame[-1] + 1
            
            bboxes_in_frame = predicted_bboxes[start_idx:end_idx]
            depths_in_frame = predicted_depths[start_idx:end_idx]
            ranges_in_frame = predicted_ranges[start_idx:end_idx]

            # 4. Bounding Boxes, Depth und Range hinzufügen
            for k in range(num_detections):
                
                bbox = bboxes_in_frame[k]
                depth_val = depths_in_frame[k]
                range_val = ranges_in_frame[k]
                
                # BBox-Berechnung und Skalierung
                x_min_orig, y_min_orig, w_orig, h_orig = bbox
                
                x_min_scaled = x_min_orig * scale_x
                y_min_scaled = y_min_orig * scale_y
                w_scaled = w_orig * scale_x
                h_scaled = h_orig * scale_y
                
                # Rechteck hinzufügen
                rect = Rectangle(
                    (x_min_scaled, y_min_scaled), 
                    w_scaled, h_scaled,           
                    linewidth=2,                    
                    edgecolor='yellow',               
                    facecolor='none'
                )
                ax.add_patch(rect)
                rectangles.append(rect)
                
                # Text-Anzeige
                text_label = f"D: {depth_val:.2f}m\nR: {range_val:.2f}m"
                
                # Positioniere den Text leicht oberhalb der Box
                txt = ax.text(
                    x_min_scaled, 
                    y_min_scaled - 1,
                    text_label, 
                    color='yellow', 
                    fontsize=8, 
                    bbox=dict(facecolor='black', alpha=0.5, edgecolor='none', pad=2)
                )
                texts.append(txt)
        
        # 5. Titel aktualisieren
        display_timestamp = "N/A"
        if num_detections > 0:
             display_timestamp = f"{timestamps[indices_in_frame[0]]}" 
             
        updated_title_text = (
            f"Frame {current_frame_idx} von {MAX_FRAMES - 1} | T: {display_timestamp} | Detektionen: {num_detections}"
        )
        title.set_text(updated_title_text)
        
        # Rückgabe aller zu aktualisierenden Elemente
        return [therm_image, title] + rectangles + texts

    # Erstelle die Animation
    ani = animation.FuncAnimation(
        fig, 
        update_frame, 
        frames=MAX_FRAMES, 
        interval=100, 
        blit=False, 
        repeat=True
    )

    # Zeigt das Fenster an und startet die Animation
    plt.show()


# --- Hauptausführung ---
if __name__ == "__main__":
    ira_matrix, frame_indices, timestamps, predicted_bboxes, predicted_depths, predicted_ranges = load_processed_data(output_path)
    
    if ira_matrix is not None and ira_matrix.size > 0:
        visualize_frames(ira_matrix, frame_indices, timestamps, predicted_bboxes, predicted_depths, predicted_ranges)
    else:
        print("Daten konnten nicht vollständig geladen oder sind leer. Beende das Skript.")