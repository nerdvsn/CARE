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
file_path = "..Outputs/Corridor2_3_sensor_1.pickle"  

# === DEKLARATION DER BEZUGSAUFLÖSUNG FÜR DIE SKALIERUNG ===
ORIGINAL_RGB_WIDTH = 640
ORIGINAL_RGB_HEIGHT = 480


def load_data(file_path):
    """Lädt alle benötigten Daten (IRA, Time, BBoxes, Depth, Range)."""
    print(f"Lade Daten aus: {file_path}...")
    try:
        with open(file_path, 'rb') as f:
            data = pickle.load(f)
        
        REQUIRED_KEYS = ['ira_temperature_matrix', 'timestamps', 'GT_bbox', 'depth', 'range']
        
        if all(key in data for key in REQUIRED_KEYS):
            ira_matrix = np.array(data['ira_temperature_matrix'])
            time_stamps = np.array(data['timestamps']).flatten() 
            
            # Verwenden von dtype=object, um die Struktur verschachtelter Listen beizubehalten
            bbox_data = np.array(data['GT_bbox'], dtype=object)
            depth_data = np.array(data['depth'], dtype=object)
            range_data = np.array(data['range'], dtype=object)
            
            print(f"Erfolgreich geladen. Gesamt-Frames: {ira_matrix.shape[0]}")
            return ira_matrix, time_stamps, bbox_data, depth_data, range_data
        else:
            missing_keys = [key for key in REQUIRED_KEYS if key not in data]
            print(f"FEHLER: Einer oder mehrere benötigte Schlüssel fehlen in der Datei: {missing_keys}")
            return None, None, None, None, None
            
    except FileNotFoundError:
        print(f"FEHLER: Datei nicht gefunden unter {file_path}.")
        return None, None, None, None, None
    except Exception as e:
        print(f"FEHLER beim Laden oder Verarbeiten der Pickle-Datei: {e}")
        return None, None, None, None, None


def visualize_frames(ira_matrix, time_stamps, bbox_data, depth_data, range_data):
    """Erstellt eine Matplotlib-Animation der thermischen Aufnahmen mit dynamischen Bounding Boxes und Depth/Range-Text."""
    if ira_matrix is None or ira_matrix.size == 0:
        print("Keine Frames zum Visualisieren vorhanden.")
        return
    
    IR_HEIGHT = ira_matrix.shape[1] 
    IR_WIDTH = ira_matrix.shape[2] 

    scale_x = IR_WIDTH / ORIGINAL_RGB_WIDTH 
    scale_y = IR_HEIGHT / ORIGINAL_RGB_HEIGHT

    fig, ax = plt.subplots(figsize=(10, 8))
    
    global_min = 15.0  
    global_max = 45.0  
    cmap = 'inferno' 
    
    therm_image = ax.imshow(
        ira_matrix[0], 
        cmap=cmap,
        vmin=global_min,
        vmax=global_max,
        interpolation='nearest' 
    )

    cbar = fig.colorbar(therm_image)
    cbar.set_label('Temperature [°C]', fontsize=14)
    
    ax.set_xticks(np.arange(0, IR_WIDTH, 4))
    ax.set_yticks(np.arange(0, IR_HEIGHT, 4))
    ax.set_xlabel(f'Horizontal Pixel (Original {IR_WIDTH})')
    ax.set_ylabel(f'Vertikal Pixel (Original {IR_HEIGHT})')
    
    # Initialer Titel (dynamische Anzahl von BBoxes)
    initial_title_text = f"Frame 0 / T: {time_stamps[0]:.2f} | BBoxes: {len(bbox_data[0])}"
    title = ax.set_title(initial_title_text)
    
    rectangles = []
    texts = []

    def update_frame(i):
        """Funktion, die in jedem Animationsschritt aufgerufen wird."""
        
        # 1. Bilddaten aktualisieren
        therm_image.set_data(ira_matrix[i]) 
        
        # 2. Alte Elemente entfernen
        for rect in rectangles:
            rect.remove()
        rectangles.clear()
        
        for txt in texts:
            txt.remove()
        texts.clear()
        
        # 3. Neue Bounding Boxes, Depth und Range hinzufügen
        current_bboxes = bbox_data[i]
        current_depths = depth_data[i]
        current_ranges = range_data[i]

        # === NEUE DYNAMISCHE LOGIK (KERN DER ÄNDERUNG) ===
        # Wir bestimmen die minimale Länge, um sicherzustellen, dass wir für jede BBox 
        # auch einen passenden Depth- und Range-Wert haben.
        num_bboxes = len(current_bboxes)
        min_length = min(num_bboxes, len(current_depths), len(current_ranges))
        
        if num_bboxes != min_length:
             print(f"WARNUNG Frame {i}: Nur {min_length} von {num_bboxes} BBoxes werden mit Depth/Range angezeigt.")

        # Die Schleife läuft nur bis zur minimalen Länge
        if min_length > 0:
            for k in range(min_length):
                
                bbox = current_bboxes[k]
                depth_val = current_depths[k]
                range_val = current_ranges[k]
                
                # BBox-Berechnung (dynamisch)
                x_min_orig, y_min_orig, w_orig, h_orig = bbox
                x_min_scaled = x_min_orig * scale_x
                y_min_scaled = y_min_orig * scale_y
                w_scaled = w_orig * scale_x
                h_scaled = h_orig * scale_y
                
                # Rechteck hinzufügen
                rect = Rectangle(
                    (x_min_scaled, y_min_scaled), 
                    w_scaled, h_scaled,           
                    linewidth=4,                    
                    edgecolor='lime',               
                    facecolor='none'
                )
                ax.add_patch(rect)
                rectangles.append(rect)
                
                # Text-Anzeige
                text_label = f"D: {depth_val:.1f} m\nR: {range_val:.1f} m"
                
                # Positioniere den Text leicht oberhalb der Box
                txt = ax.text(
                    x_min_scaled, 
                    y_min_scaled - 1,
                    text_label, 
                    color='cyan', 
                    fontsize=10, 
                    bbox=dict(facecolor='black', alpha=0.5, edgecolor='none', pad=2)
                )
                texts.append(txt)
        
        # 4. Titel aktualisieren (dynamisch)
        updated_title_text = (
            f"Frame {i+1} von {ira_matrix.shape[0]} / T: {time_stamps[i]:.2f} | BBoxes: {num_bboxes} (mit Daten: {min_length})"
        )
        title.set_text(updated_title_text)
        
        # Rückgabe aller zu aktualisierenden Elemente
        return [therm_image, title] + rectangles + texts

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
    ira_matrix, time_stamps, bbox_data, depth_data, range_data = load_data(file_path)
    
    if ira_matrix is not None and bbox_data is not None and depth_data is not None and range_data is not None:
        visualize_frames(ira_matrix, time_stamps, bbox_data, depth_data, range_data)
    else:
        print("Daten konnten nicht vollständig geladen werden. Beende das Skript.")