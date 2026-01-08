import pickle
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib as mpl

# === WICHTIG: TKAGG-BACKEND FÜR VNC/TERMINAL AUSFÜHRUNG ===
# Dies stellt sicher, dass Matplotlib ein Fenster öffnen kann.
try:
    # Muss vor jedem Matplotlib-Import erfolgen, wenn das Backend geändert wird
    mpl.use('TkAgg') 
except ImportError:
    print("Warnung: TkAgg-Backend nicht verfügbar. Bitte 'sudo apt install python3-tk' ausführen.")

# === PFAD ZUR DATENDATEI ANPASSEN ===
# Die Datei liegt im selben Ordner wie das Skript (relative Pfadangabe).
file_path = "FiveUser_Static_3_sensor_4.pkl"  


def load_data(file_path):
    """Lädt die IRA-Matrix und den Frame-Index aus der Pickle-Datei."""
    print(f"Lade Daten aus: {file_path}...")
    try:
        with open(file_path, 'rb') as f:
            data = pickle.load(f)
        
        # Überprüfen der notwendigen Schlüssel aus dem TADAR-Datensatz
        if 'ira_matrix' in data and 'frame_index' in data:
            # ira_matrix sollte die Form (Anzahl Frames, 24, 32) haben
            ira_matrix = np.array(data['ira_matrix'])
            frame_indices = np.array(data['frame_index']).flatten()
            print(f"Erfolgreich geladen. Gesamt-Frames: {ira_matrix.shape[0]}")
            return ira_matrix, frame_indices
        else:
            print("FEHLER: Schlüssel 'ira_matrix' oder 'frame_index' fehlt in der Datei.")
            return None, None
            
    except FileNotFoundError:
        # Hier ist der verbesserte Fehler, um dem Benutzer zu helfen, das Pfadproblem zu beheben
        print(f"FEHLER: Datei nicht gefunden unter {file_path}. Überprüfen Sie den Pfad oder das Arbeitsverzeichnis.")
        return None, None
    except Exception as e:
        print(f"FEHLER beim Laden oder Verarbeiten der Pickle-Datei: {e}")
        return None, None


def visualize_frames(ira_matrix, frame_indices):
    """Erstellt eine Matplotlib-Animation der thermischen Aufnahmen."""
    if ira_matrix is None or ira_matrix.size == 0:
        print("Keine Frames zum Visualisieren vorhanden.")
        return

    # Initiales Setup des Plots (24x32 Pixel)
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # === LÖSUNG FÜR DAS SCHWARZBILD-PROBLEM: Feste Farbskala ===
    # Setzen Sie Vmin/Vmax auf einen realistischen Temperaturbereich (z.B. 15°C bis 45°C), 
    # um Ausreißer in den Rohdaten zu ignorieren und die relevanten Daten sichtbar zu machen.
    global_min = 15.0  
    global_max = 45.0  
    
    cmap = 'inferno' 
    
    # Initiales Bild, horizontal gespiegelt (häufige Korrektur für MLX90640)
    therm_image = ax.imshow(
        np.fliplr(ira_matrix[0]),
        cmap=cmap,
        vmin=global_min,
        vmax=global_max,
        interpolation='nearest' 
    )

    # Farbleiste hinzufügen
    cbar = fig.colorbar(therm_image)
    cbar.set_label('Temperature [°C]', fontsize=14)
    
    # Titel für den Frame-Index
    title = ax.set_title(f"Frame 0 / Index: {frame_indices[0]}")

    def update_frame(i):
        """Funktion, die in jedem Animationsschritt aufgerufen wird."""
        # Aktualisiere die Daten für den aktuellen Frame i (mit horizontaler Spiegelung)
        therm_image.set_data(np.fliplr(ira_matrix[i]))
        # Aktualisiere den Titel, um den Fortschritt anzuzeigen
        title.set_text(f"Frame {i+1} von {ira_matrix.shape[0]} (Original Index: {frame_indices[i]})")
        return [therm_image, title]

    # Erstelle die Animation
    # interval=100 ms ergibt 10 Frames pro Sekunde (10 Hz)
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
    ira_matrix, frame_indices = load_data(file_path)
    if ira_matrix is not None:
        visualize_frames(ira_matrix, frame_indices)