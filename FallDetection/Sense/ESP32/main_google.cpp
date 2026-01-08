#include <Arduino.h>
#include <Wire.h>
#include <Adafruit_MLX90640.h>

Adafruit_MLX90640 mlx;
float frame[768];

// Header zur Synchronisation
const uint8_t HEADER[4] = {0xAA, 0xBB, 0xCC, 0xDD};

void setup() {
  Serial.begin(1000000);  // 1 Mbit Turbo (Linux kompatibel)
  
  Wire.begin(21, 22);
  Wire.setClock(1000000); // 1 MHz I2C

  if (!mlx.begin()) {
    while (1);
  }

  mlx.setMode(MLX90640_CHESS); 
  mlx.setResolution(MLX90640_ADC_18BIT);
  
  // WICHTIG: 16 Hz für Sturzerkennung
  mlx.setRefreshRate(MLX90640_16_HZ); 
}

void loop() {
  if (mlx.getFrame(frame) == 0) {
    // 1. Header senden
    Serial.write(HEADER, 4);
    
    // 2. Daten senden (768 Floats roh = 3072 Bytes)
    Serial.write((uint8_t*)frame, sizeof(frame));
    
    Serial.flush();
  }
}