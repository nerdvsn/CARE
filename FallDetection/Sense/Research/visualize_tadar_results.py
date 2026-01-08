import pickle
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib as mpl
import os

# WICHTIG: Setzt das Matplotlib-Backend für die Fensteranzeige (TkAgg)
try:
    mpl.use('TkAgg') 
except:
    pass

# === EINSTELLUNGEN ===
# Name der Datei, die von test.py gespeichert wurde (OHNE .pkl)
# Hier wird angenommen, dass die erste Datei aus der test.py Liste verwendet wird.
FILE_TO_LOAD = 'Bathroom1_0_sensor_1' 
OUTPUT_PATH = 'Outputs/'
FULL_PATH = os.path.join(OUTPUT_PATH, FILE_TO_LOAD + '.pkl')

# === VISUALISIERUNGSFUNKTION ===

def visualize_tadar_output_animation(output_data):
    """
    Erstellt eine Animation, die das thermische Bild (IRA) mit 
    den erkannten Bounding Boxes (ROI) und optional die Range Map (Entfernung) 
    darstellt.
    """
    
    if 'ira_matrix' not in output_data or 'frame_index' not in output_data:
        print("FEHLER: 'ira_matrix' oder 'frame_index' fehlt in der Output-Datei.")
        return

    # *****************************************************************
    # KORREKTUR: Umwandlung der Listen in das korrekte NumPy-Format
    # np.array() ist robuster als np.concatenate bei unsauberen Listenformaten.
    # *****************************************************************
    
    # ira_frames: Liste von 24x32 Arrays -> einzelnes 3D-Array (Frames, 24, 32)
    # Behebt den Fehler "zero-dimensional arrays cannot be concatenated" für ira_matrix.
    ira_frames = np.array(output_data['ira_matrix']) 
    
    # frame_indices: Liste von Skalaren -> 1D-Array
    # Behebt den Fehler "zero-dimensional arrays cannot be concatenated" für frame_index.
    frame_indices = np.array(output_data['frame_index']).flatten()
    
    # Optional: Bounding Boxes und Range Maps
    # Annahme: valid_BBoxes und Range_Map wurden der test()-Funktion hinzugefügt.
    boxes_per_frame = output_data.get('valid_BBoxes', [[]] * ira_frames.shape[0])
    range_maps = output_data.get('Range_Map', None) # Die 24x32 Range Map pro Frame

    total_frames = ira_frames.shape[0]

    # Feste Skala für thermische Darstellung (Celsius)
    THERMAL_VMIN = 15.0  
    THERMAL_VMAX = 45.0  
    
    # Feste Skala für Entfernung (Meter)
    RANGE_VMIN = 0.5
    RANGE_VMAX = 5.0 

    # Setup des Plots (entweder nur Thermal oder Thermal + Range Map)
    if range_maps is not None:
        # Konvertierung der Range Maps
        range_maps = np.array(range_maps)
        fig, (ax_thermal, ax_range) = plt.subplots(1, 2, figsize=(16, 8))
    else:
        fig, ax_thermal = plt.subplots(figsize=(10, 8))
        ax_range = None
    
    # --- Thermal Plot Initialisierung ---
    therm_image = ax_thermal.imshow(
        np.fliplr(ira_frames[0]),
        cmap='inferno',
        vmin=THERMAL_VMIN,
        vmax=THERMAL_VMAX,
        interpolation='nearest' 
    )
    fig.colorbar(therm_image, ax=ax_thermal).set_label('Temp [°C]', fontsize=14)
    ax_thermal.set_title(f"Thermal ROI Detection (Frame 0)")
    
    # --- Range Map Plot Initialisierung ---
    if ax_range:
        range_image = ax_range.imshow(
            np.fliplr(range_maps[0]),
            cmap='jet', # Typische Farbskala für Tiefe/Entfernung
            vmin=RANGE_VMIN,
            vmax=RANGE_VMAX,
            interpolation='nearest' 
        )
        fig.colorbar(range_image, ax=ax_range).set_label('Range [m]', fontsize=14)
        ax_range.set_title("Generated Range Map")
    
    title = fig.suptitle(f"File: {FILE_TO_LOAD} | Frame 0/{total_frames}")

    # Liste für die Bounding Boxes
    rectangles = []

    def update_frame(i):
        """Aktualisiert Frame, ROIs und Titel."""
        
        # 1. Thermalbild aktualisieren
        therm_image.set_data(np.fliplr(ira_frames[i]))
        
        # 2. Range Map aktualisieren
        if ax_range and i < len(range_maps):
            # Nur die Range Map aktualisieren, wenn sie vorhanden ist
            # Stellen Sie sicher, dass Range_Map-Arrays die gleiche Dimension (24, 32) haben.
            range_image.set_data(np.fliplr(range_maps[i]))

        # 3. Bounding Boxes aktualisieren
        for rect in rectangles:
            rect.remove()
        rectangles.clear()

        # Füge neue Bounding Boxes hinzu
        if i < len(boxes_per_frame) and boxes_per_frame[i] is not None:
            # Annahme: boxes_per_frame[i] ist eine Liste von (x, y, w, h) Tupeln für Frame i
            for box in boxes_per_frame[i]:
                # cv2 verwendet [x, y, w, h], Matplotlib.Rectangle verwendet (x, y, width, height)
                x, y, w, h = box 
                
                # Wenn x, y, w, h Listen sind, müssen sie in float umgewandelt werden
                x, y, w, h = float(x), float(y), float(w), float(h)
                
                rect = plt.Rectangle((x, y), w, h, 
                                     linewidth=2, edgecolor='lime', facecolor='none')
                ax_thermal.add_patch(rect)
                rectangles.append(rect)
        
        # 4. Titel aktualisieren
        title.set_text(f"File: {FILE_TO_LOAD} | Frame {i+1}/{total_frames}")
        
        # Rückgabe aller geänderten Matplotlib-Objekte
        # Die Liste muss alle Objekte enthalten, die sich ändern können.
        artists = [therm_image, title]
        if ax_range:
            artists.append(range_image)
        artists.extend(rectangles)
        return artists

    ani = animation.FuncAnimation(
        fig, 
        update_frame, 
        frames=total_frames, 
        interval=100, 
        blit=False, 
        repeat=True
    )

    plt.show()

# === HAUPTAUSFÜHRUNG ===
if __name__ == "__main__":
    if not os.path.exists(FULL_PATH):
        print(f"FEHLER: Ergebnisdatei nicht gefunden: {FULL_PATH}")
        print("Stellen Sie sicher, dass 'test.py' ausgeführt wurde und die Datei existiert.")
    else:
        print(f"Lade Ergebnisse aus: {FULL_PATH}...")
        try:
            with open(FULL_PATH, 'rb') as f:
                output_data = pickle.load(f)
            
            # Da test.py die Ergebnisse iteriert und speichert, prüfen wir das Format.
            if isinstance(output_data, dict):
                # Wenn es ein einzelnes Dictionary ist (z.B. nur eine Datei getestet)
                data_to_visualize = output_data
            elif isinstance(output_data, list) and len(output_data) > 0 and isinstance(output_data[0], dict):
                # Wenn es eine Liste von Ergebnissen ist, nehmen wir das erste Element
                print("HINWEIS: Es wurden Ergebnisse mehrerer Dateien gefunden. Visualisiere nur die erste Datei.")
                data_to_visualize = output_data[0]
            else:
                print("FEHLER: Das geladene Format wird nicht unterstützt (kein Dictionary/keine Liste von Dictionaries).")
                data_to_visualize = None

            if data_to_visualize:
                # WICHTIG: Die Keys 'valid_BBoxes' und 'Range_Map' müssen 
                # der test()-Funktion hinzugefügt worden sein, damit die ROIs und die Map erscheinen.
                visualize_tadar_output_animation(data_to_visualize)

        except Exception as e:
            print(f"FEHLER beim Laden oder Verarbeiten der Datei: {e}")
            print("Mögliche Ursache: Inkompatibilität der NumPy-Versionen (erforderlich: NumPy < 2.0)")