#include <Arduino.h>
#include <Wire.h>
#include <Adafruit_MLX90640.h>

// Instanz des Sensors
Adafruit_MLX90640 mlx;

// Buffer für 32x24 Pixel = 768 Floats
float frame[32 * 24]; 

// Header Definition (Muss exakt zum Python Skript passen!)
const uint8_t HEADER[] = {0xAA, 0xBB, 0xCC, 0xDD};

void setup() {
  // 1. Serielle Kommunikation starten
  // WICHTIG: Muss 1000000 sein, genau wie in Python BAUD_RATE
  Serial.begin(1000000);
  while (!Serial); // Warten bis Serial bereit ist

  // 2. I2C Konfiguration für ESP32
  // SDA = 21, SCL = 22 (Standard beim ESP32 DevKit)
  Wire.begin(21, 22);
  Wire.setClock(1000000); // 1 MHz I2C Takt für schnelle Auslesung

  // 3. Sensor initialisieren
  if (!mlx.begin()) {
    // Fehlerbehandlung: Blinkendes Signal oder Endlosschleife
    while (1) {
        delay(100);
    }
  }

  // 4. Sensor Einstellungen für Performance
  mlx.setMode(MLX90640_CHESS);       // Chess Pattern (genauer, weniger Rauschen)
  mlx.setResolution(MLX90640_ADC_18BIT); // Gute Balance zw. Rauschen und Speed
  mlx.setRefreshRate(MLX90640_16_HZ);    // 16 FPS ist optimal für Sturzerkennung
}

void loop() {
  // Versuche einen Frame zu lesen (Rückgabewert 0 bedeutet Erfolg)
  if (mlx.getFrame(frame) == 0) {
    
    // --- SCHRITT 1: Header senden (4 Bytes) ---
    // Python Zeile: if ser.read(3) == b'\xbb\xcc\xdd':
    Serial.write(HEADER, 4);

    // --- SCHRITT 2: Daten senden (3072 Bytes) ---
    // Python Zeile: raw_data = ser.read(FRAME_BYTES)
    // Wir casten den float-Pointer zu einem Byte-Pointer und senden den gesamten Block.
    // 768 floats * 4 bytes/float = 3072 bytes.
    Serial.write((uint8_t*)frame, sizeof(frame));

    // Sicherstellen, dass der Buffer geleert wird
    Serial.flush();
  }
}


// gucke ob die Umgebungstemperatur korrekt ist.