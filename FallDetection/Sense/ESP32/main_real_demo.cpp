#include <Arduino.h>
#include <Wire.h>
#include <Adafruit_MLX90640.h>

Adafruit_MLX90640 mlx;
float frame[32*24]; // Puffer für 768 Pixel

void setup() {
  // HIER IST DIE KORREKTUR:
  // Wir nutzen exakt die Geschwindigkeit, die du wolltest.
  Serial.begin(1000000); 
  
  // I2C Turbo (1 MHz) für schnelles Sensor-Lesen
  Wire.begin();
  Wire.setClock(1000000); 

  if (!mlx.begin()) {
    Serial.println("Sensor Error");
    while(1);
  }

  // Einstellungen
  mlx.setMode(MLX90640_CHESS);
  mlx.setResolution(MLX90640_ADC_18BIT);
  
  // Bei JSON über 921600 Baud sind 8 Hz oder 16 Hz realistisch.
  // Wir versuchen 16 Hz. Falls es stockt, geh auf 8 Hz runter.
  mlx.setRefreshRate(MLX90640_16_HZ); 

  Serial.println("ESP32 ready @ 32Hz Refresh (Target: 16 FPS).");
}

void loop() {
  // Versuchen, ein Bild zu lesen (0 = Erfolg)
  if (mlx.getFrame(frame) == 0) {
    
    // Umgebungstemperatur berechnen (wichtig für dein Skript)
    float ambientTemp = mlx.getTa(false); 

    // --- JSON STREAMING (Speicherschonend) ---
    // Wir bauen keinen riesigen String, sondern senden Stück für Stück.
    // Das Format ist: {"temperature":[23.1, 24.5, ...], "at": 25.0}
    
    Serial.print("{\"temperature\":[");
    
    for (int i = 0; i < 768; i++) {
      // Wert mit 2 Nachkommastellen senden
      Serial.print(frame[i], 2);
      
      // Komma setzen, außer beim letzten Wert
      if (i < 767) {
        Serial.print(",");
      }
    }
    
    Serial.print("],\"at\":");
    Serial.print(ambientTemp, 2);
    Serial.println("}"); // Ende des JSON-Objekts + Zeilenumbruch
    
    // Puffer leeren, damit es sofort rausgeht
    Serial.flush();
  }
}