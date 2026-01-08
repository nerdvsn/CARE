from ultralytics import YOLO
import os


# Pfad .yaml
DATASET_YAML = 'Le2i-2/data.yaml' 

# The model
BASE_MODEL = 'yolov8s.pt' 

# Trainingseinstellungen
EPOCHS_COUNT = 100         
IMAGE_SIZE = 640           
PROJECT_NAME = 'YOLOv8_Fall_Detection' 
RUN_NAME = 'finetune_v1'   


# MODELL LADEN
print(f"Lade das vortrainierte Modell: {BASE_MODEL}")
try:
    # Das Basismodell laden. Wenn es nicht existiert, wird es heruntergeladen.
    model = YOLO(BASE_MODEL)
except Exception as e:
    print(f"Fehler beim Laden des Modells {BASE_MODEL}: {e}")
    exit()


print("Modell erfolgreich geladen. Starte Fine-Tuning...")

# 3.TRAINING
results = model.train(
   data=DATASET_YAML,              
   epochs=EPOCHS_COUNT,            
   imgsz=IMAGE_SIZE,               
   batch=4,                       
   project=PROJECT_NAME,           
   name=RUN_NAME,                  
   workers=2,                      
   patience=20,                    
   optimizer='AdamW',              
   seed=42,                        
   val=True                        
)


print("Fine-Tuning abgeschlossen.")

# 4. BESTES MODELL VALIDIEREN
final_metrics = model.val()

print("\n--- ZUSAMMENFASSUNG ---")
print(f"Endgültige mAP50-95 (Validierung): {final_metrics.box.map:.4f}")
print(f"Endgültige mAP50 (Validierung): {final_metrics.box.map50:.4f}")

# 5. INFERENZ-TEST (Optional)
try:
    test_path = 'data/split_data/images/test/DJI_scene_0188.png' # Pfad anpassen
    if os.path.exists(test_path):
        print(f"\nTeste Modell auf {test_path}...")
        results_inference = model(test_path)
        # Speichert die annotierten Bilder in runs/detect/predict
        model(test_path, save=True) 
        print("Inferenz-Test abgeschlossen. Annotiertes Bild gespeichert.")
    else:
        print("Hinweis: Überspringe Inferenz-Test, da kein Testbild-Pfad gefunden wurde.")
except Exception as e:
    print(f"Fehler bei der Inferenz: {e}")