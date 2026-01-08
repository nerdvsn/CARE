import matplotlib.pyplot as plt
import numpy as np
from ultralytics import YOLO # Importiere YOLO-Klasse für Modell-Initialisierung
from ultralytics.engine.results import Results # Beibehalten für Typ-Hinweis

# ====================================================================
# 1. KONFIGURATION & INFERENZ
# ====================================================================
# Pfad zu Ihrem besten trainierten Modell
MODEL_PATH = 'YOLOv8_Fall_Detection/finetune_v1/weights/best.pt'
# Pfad zu Ihrem Testbild
TEST_IMAGE_PATH = 'Le2i-2/test/images/000101_jpg.rf.09273c65f547b44dfffe2a345be205ac.jpg' 

# Modell laden
try:
    model = YOLO(MODEL_PATH)
    print(f"Bestes Modell von {MODEL_PATH} erfolgreich geladen.")
except Exception as e:
    print(f"Fehler beim Laden des Modells: {e}")
    exit()

# Inferenz durchführen
# Speichert die Ergebnisse in der results_list
results_list = model.predict(source=TEST_IMAGE_PATH, save=False, conf=0.25, iou=0.7, verbose=False)

# Wir nehmen an, Ihr 'results'-Objekt ist das erste Element der Liste:
first_result = results_list[0] 

# ====================================================================
# 2. DATEN EXTRAHIEREN
# ====================================================================
# Bilddaten (H, W, C)
image_array = first_result.orig_img 

# Bounding Boxes (xyxy-Format: xmin, ymin, xmax, ymax)
# Konvertierung zu NumPy-Array ist für Matplotlib notwendig
boxes = first_result.boxes.xyxy.cpu().numpy() 

# Konfidenzwerte
confidences = first_result.boxes.conf.cpu().numpy()

# Klassen-IDs
class_ids = first_result.boxes.cls.cpu().numpy().astype(int)

# Namen der Klassen (wird automatisch aus dem geladenen Modell extrahiert)
class_names = first_result.names 

# ====================================================================
# 3. MATPLOTLIB VISUALISIERUNG
# ====================================================================
plt.figure(figsize=(10, 6))

# Das Bild anzeigen. YOLOv8s orig_img ist in der Regel RGB.
plt.imshow(image_array) 

ax = plt.gca() # Aktuelle Achse holen

# Iterieren und Bounding Boxes zeichnen
for box, conf, class_id in zip(boxes, confidences, class_ids):
    xmin, ymin, xmax, ymax = box
    width = xmax - xmin
    height = ymax - ymin
    
    label = f"{class_names[class_id]} ({conf:.2f})"
    
    # Rechteck zeichnen
    rect = plt.Rectangle((xmin, ymin), width, height, 
                         fill=False, 
                         edgecolor='red', 
                         linewidth=2)
    ax.add_patch(rect)
    
    # Label hinzufügen
    ax.text(xmin, ymin - 10, label, 
            color='red', 
            fontsize=12, 
            bbox=dict(facecolor='white', alpha=0.6, edgecolor='none', pad=2))

plt.title(f"YOLOv8 Detektion auf: {TEST_IMAGE_PATH.split('/')[-1]}", fontsize=14)
plt.axis('off') # Achsenbeschriftung ausschalten
plt.show()