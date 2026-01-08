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
file_path = "FiveUser_Static_4_sensor_4.pickle"  

# === DEKLARATION DER BEZUGSAUFLÖSUNG FÜR DIE SKALIERUNG ===
# Wir behalten die Annahme 640x480, da die BBox-Skalierung damit funktioniert.
ORIGINAL_RGB_WIDTH = 640
ORIGINAL_RGB_HEIGHT = 480


def load_data(file_path):
    """Lädt die IRA-Matrix, die Zeitstempel und die Bounding Boxes aus der Pickle-Datei."""
    print(f"Lade Daten aus: {file_path}...")
    try:
        with open(file_path, 'rb') as f:
            data = pickle.load(f)
        
        REQUIRED_KEYS = ['ira_temperature_matrix', 'timestamps', 'GT_bbox']
        
        if all(key in data for key in REQUIRED_KEYS):
            # ANNAHME: ira_temperature_matrix hat die Form (Anzahl Frames, Höhe, Breite)
            ira_matrix = np.array(data['ira_temperature_matrix'])
            time_stamps = np.array(data['timestamps']).flatten() 
            bbox_data = np.array(data['GT_bbox'], dtype=object)
            
            print(f"Erfolgreich geladen. Gesamt-Frames: {ira_matrix.shape[0]}")
            return ira_matrix, time_stamps, bbox_data
        else:
            # ... (Fehlerbehandlung) ...
            return None, None, None
            
    except FileNotFoundError:
        print(f"FEHLER: Datei nicht gefunden unter {file_path}. Überprüfen Sie den Pfad oder das Arbeitsverzeichnis.")
        return None, None, None
    except Exception as e:
        print(f"FEHLER beim Laden oder Verarbeiten der Pickle-Datei: {e}")
        return None, None, None


def visualize_frames(ira_matrix, time_stamps, bbox_data):
    """Erstellt eine Matplotlib-Animation der thermischen Aufnahmen mit Bounding Boxes."""
    if ira_matrix is None or ira_matrix.size == 0:
        print("Keine Frames zum Visualisieren vorhanden.")
        return
    
    # === KORREKTUR: Dimensionsbestimmung des einzelnen Frames ===
    # Nehmen Sie die letzten ZWEI Dimensionen der Matrix, um Höhe (H) und Breite (W) zu erhalten
    # HINWEIS: Bei IRA-Daten (MLX90640) ist die Reihenfolge oft (24, 32).
    # Wir nehmen an: ira_matrix.shape = (Frames, Höhe, Breite)
    IR_HEIGHT = ira_matrix.shape[1] # MUSS 24 sein
    IR_WIDTH = ira_matrix.shape[2]  # MUSS 32 sein

    # Skalierungsfaktoren neu berechnen (mit korrigierten IR-Dimensionen)
    scale_x = IR_WIDTH / ORIGINAL_RGB_WIDTH 
    scale_y = IR_HEIGHT / ORIGINAL_RGB_HEIGHT
    
    # --- DEBUG Skalierung (KORRIGIERT) ---
    print(f"\n--- DEBUG Skalierung (KORRIGIERT) ---")
    print(f"IR-Dimensionen: {IR_WIDTH}x{IR_HEIGHT}") # Wird 32x24 ausgeben
    print(f"Original-Dimensionen: {ORIGINAL_RGB_WIDTH}x{ORIGINAL_RGB_HEIGHT}")
    print(f"Skalierungsfaktoren: X={scale_x:.4f}, Y={scale_y:.4f}")

    # ... (Plot-Setup bleibt gleich) ...
    fig, ax = plt.subplots(figsize=(10, 8))
    
    global_min = 15.0  
    global_max = 45.0  
    cmap = 'inferno' 
    
    # Der erste Frame (Index 0) ist ira_matrix[0]
    therm_image = ax.imshow(
        ira_matrix[0], # Zeigt den 24x32 Frame an
        cmap=cmap,
        vmin=global_min,
        vmax=global_max,
        interpolation='nearest' 
    )

    cbar = fig.colorbar(therm_image)
    cbar.set_label('Temperature [°C]', fontsize=14)
    
    # Beschriftungen (korrigiert auf IR_WIDTH/IR_HEIGHT)
    ax.set_xticks(np.arange(0, IR_WIDTH, 4)) # 32
    ax.set_yticks(np.arange(0, IR_HEIGHT, 4)) # 24
    ax.set_xlabel('Horizontal Pixel (Original 32)')
    ax.set_ylabel('Vertikal Pixel (Original 24)')
    
    title = ax.set_title(f"Frame 0 / Timestamp: {time_stamps[0]:.2f}")
    
    rectangles = []

    def update_frame(i):
        """Funktion, die in jedem Animationsschritt aufgerufen wird."""
        
        # 1. Bilddaten aktualisieren
        therm_image.set_data(ira_matrix[i]) 
        
        # 2. Alte Rechtecke entfernen
        for rect in rectangles:
            rect.remove()
        rectangles.clear()
        
        # 3. Neue Rechtecke hinzufügen
        current_bboxes = bbox_data[i]
        
        if len(current_bboxes) > 0:
            for k, bbox in enumerate(current_bboxes):
                
                # BBox Format: [x_min, y_min, w, h] (in der Originalauflösung)
                x_min_orig, y_min_orig, w_orig, h_orig = bbox
                
                # Skalierung der BBox-Werte (Skalierungsfaktoren wurden korrigiert)
                x_min_scaled = x_min_orig * scale_x
                y_min_scaled = y_min_orig * scale_y
                w_scaled = w_orig * scale_x
                h_scaled = h_orig * scale_y
                
                # Debug-Ausgabe für den ersten Frame (KORRIGIERT)
                if i == 0 and k == 0:
                    print(f"Debug BBox (Frame {i}, Box {k}): Original=[{x_min_orig}, {y_min_orig}, {w_orig}, {h_orig}]")
                    print(f"Debug BBox (Frame {i}, Box {k}): Skaliert=[{x_min_scaled:.1f}, {y_min_scaled:.1f}, {w_scaled:.1f}, {h_scaled:.1f}] (X_Scale={scale_x:.4f}, Y_Scale={scale_y:.4f})")
                
                
                # Füge das Rechteck hinzu
                rect = Rectangle(
                    (x_min_scaled, y_min_scaled), # x, y der linken oberen Ecke
                    w_scaled, h_scaled,           # Breite und Höhe
                    linewidth=4,                    
                    edgecolor='lime',               
                    facecolor='none'
                )
                ax.add_patch(rect)
                rectangles.append(rect)
        
        # 4. Titel aktualisieren
        title.set_text(f"Frame {i+1} von {ira_matrix.shape[0]} (Timestamp: {time_stamps[i]:.2f}) | BBoxen: {len(current_bboxes)}")
        
        return [therm_image, title] + rectangles

    # Erstelle die Animation
    ani = animation.FuncAnimation(
        fig, 
        update_frame, 
        frames=ira_matrix.shape[0], 
        interval=100, 
        blit=False, 
        repeat=True
    )

    # Zeigt das Fenster an und startet die Animation
    plt.show()


# --- Hauptausführung ---
if __name__ == "__main__":
    ira_matrix, time_stamps, bbox_data = load_data(file_path)
    
    if ira_matrix is not None and bbox_data is not None:
        visualize_frames(ira_matrix, time_stamps, bbox_data)
    else:
        print("Daten konnten nicht vollständig geladen werden. Beende das Skript.")