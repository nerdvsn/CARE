#include <Arduino.h>
#include <Wire.h>
#include <Adafruit_MLX90640.h>

Adafruit_MLX90640 mlx;
float frame[32*24]; // Puffer für 768 Pixel

// ===== KORREKTE AMBIENT-TEMPERATURBERECHNUNG (Paper-konform) =====
float calculateAmbientTemp(const float* pixels, int size) {
  const int n_lowest = size * 0.3; // 30% der Pixel (ca. 230 bei 768)
  float lowest_vals[256]; // Fester Puffer für kälteste Werte
  
  // Initialisiere Puffer mit hohen Werten
  for (int i = 0; i < n_lowest; i++) {
    lowest_vals[i] = 100.0f; // "Unendlich" für float
  }
  
  // Finde die kältesten n_lowest Werte (O(n) statt O(n log n))
  for (int i = 0; i < size; i++) {
    if (pixels[i] < lowest_vals[n_lowest - 1]) {
      int pos = n_lowest - 1;
      // Schiebe Werte nach rechts, um Platz zu machen
      while (pos > 0 && pixels[i] < lowest_vals[pos - 1]) {
        lowest_vals[pos] = lowest_vals[pos - 1];
        pos--;
      }
      lowest_vals[pos] = pixels[i];
    }
  }
  
  // Median der kältesten Werte berechnen
  if (n_lowest % 2 == 0) {
    return (lowest_vals[n_lowest/2 - 1] + lowest_vals[n_lowest/2]) / 2.0f;
  } else {
    return lowest_vals[n_lowest/2];
  }
}
// ===== ENDE KORREKTE BERECHNUNG =====

void setup() {
  Serial.begin(1000000); // 921.6 kbit/s für Echtzeit
  
  // I2C Turbo (1 MHz)
  Wire.begin(21, 22); // SDA=21, SCL=22 für ESP32 DevKitC V4
  Wire.setClock(1000000);

  if (!mlx.begin()) {
    Serial.println("Sensor Error");
    while(1);
  }

  // Optimale Einstellungen für Geschwindigkeit
  mlx.setMode(MLX90640_CHESS); // Schachbrettmuster für schnelleres Lesen
  mlx.setResolution(MLX90640_ADC_18BIT);
  mlx.setRefreshRate(MLX90640_16_HZ); // 8 Hz statt 16 Hz für Stabilität
  Serial.println("✅ ESP32 + MLX90640 initialisiert (Paper-konforme Ambient-Temp)");
}

void loop() {
  if (mlx.getFrame(frame) == 0) {
    // ===== NEU: Paper-konforme Ambient-Temperatur =====
    float ambientTemp = calculateAmbientTemp(frame, 768);
    // ===== ENDE NEU =====

    // JSON-Streaming (speichereffizient)
    Serial.print("{\"temperature\":[");
    
    for (int i = 0; i < 768; i++) {
      Serial.print(frame[i], 1); // 1 Nachkommastelle reicht
      
      if (i < 767) {
        Serial.print(",");
      }
    }
    
    Serial.print("],\"at\":");
    Serial.print(ambientTemp, 1); // 1 Nachkommastelle
    Serial.println("}");
    Serial.flush(); // Sofort senden
    
    delay(10); // Minimaler Delay für 8 Hz
  }
}