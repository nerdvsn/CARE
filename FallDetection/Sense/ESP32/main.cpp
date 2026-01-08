#include <Arduino.h>
#include <Wire.h>
#include <Adafruit_MLX90640.h>

Adafruit_MLX90640 mlx;

float frame[24 * 32];

// Header zur Synchronisation
const uint8_t HEADER[4] = {0xAA, 0xBB, 0xCC, 0xDD};

void setup() {
  // 1. Serielle Verbindung (Turbo)
  Serial.begin(1000000); 
)
  // 2. I2C Verbindung
  Wire.begin(21, 22);
  Wire.setClock(1000000) ; // 1 MHz ist Pflicht für >16Hz

  delay(50);
  if (!mlx.begin()) {
    while (1) { delay(1000); }
  }

  // 3. WICHTIG: Sensor-Konfiguration für Sturzerkennung
  mlx.setMode(MLX90640_CHESS);       // Schachbrettmuster (weniger Rauschen bei Bewegung)
  mlx.setResolution(MLX90640_ADC_18BIT); // Gute Balance
  
  // HIER IST DER SCHLÜSSEL:
  // Wir stellen 32 Hz ein, um realistische 16 Vollbilder (FPS) zu bekommen
  mlx.setRefreshRate(MLX90640_32_HZ); 
  
  Serial.println("ESP32 ready @ 32Hz Refresh (Target: 16 FPS).");
}

void loop() {
  // Blockierender Read vom Sensor; getFrame liefert in der Regel flott
  if (mlx.getFrame(frame) == 0) {
    // Convert to int16 (temp * 100)
    int16_t packed[24 * 32];
    for (int i = 0; i < 24*32; ++i) {
      // clamp, scale
      float v = frame[i];
      if (isnan(v)) v = 0.0;
      int32_t val = (int32_t)round(v * 100.0f); // two decimals
      if (val > 32767) val = 32767;
      if (val < -32768) val = -32768;
      packed[i] = (int16_t)val;
    }

    // Write header + data (little-endian)
    Serial.write(HEADER, 4);
    Serial.write((uint8_t*)packed, sizeof(packed));
    Serial.flush();
  }
}
