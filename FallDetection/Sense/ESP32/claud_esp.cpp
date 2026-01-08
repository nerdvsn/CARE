#include <Arduino.h>
#include <Wire.h>
#include <Adafruit_MLX90640.h>

Adafruit_MLX90640 mlx;
float frame[32*24]; // 768 Pixel

// ===== OPTIMIERTE AMBIENT-TEMPERATUR (Paper-konform + schnell) =====
float calculateAmbientTemp(const float* pixels, int size) {
    const int n_lowest = (int)(size * 0.3f); // 30% von 768 = 230
    
    // KORREKTUR: Dynamischer Puffer oder größerer fester Puffer
    static float lowest_vals[256]; // Genug für 30% von 768 (230)
    
    // Initialisiere mit ersten n_lowest Werten (statt 100.0)
    for (int i = 0; i < n_lowest && i < size; i++) {
        lowest_vals[i] = pixels[i];
    }
    
    // Sortiere initial (einfaches Bubble Sort für kleine Arrays)
    for (int i = 0; i < n_lowest - 1; i++) {
        for (int j = 0; j < n_lowest - i - 1; j++) {
            if (lowest_vals[j] > lowest_vals[j + 1]) {
                float temp = lowest_vals[j];
                lowest_vals[j] = lowest_vals[j + 1];
                lowest_vals[j + 1] = temp;
            }
        }
    }
    
    // Prüfe restliche Pixel
    for (int i = n_lowest; i < size; i++) {
        if (pixels[i] < lowest_vals[n_lowest - 1]) {
            // Finde Einfügeposition (binäre Suche wäre schneller)
            int pos = n_lowest - 1;
            while (pos > 0 && pixels[i] < lowest_vals[pos - 1]) {
                lowest_vals[pos] = lowest_vals[pos - 1];
                pos--;
            }
            lowest_vals[pos] = pixels[i];
        }
    }
    
    // Median berechnen
    if (n_lowest % 2 == 0) {
        return (lowest_vals[n_lowest/2 - 1] + lowest_vals[n_lowest/2]) / 2.0f;
    } else {
        return lowest_vals[n_lowest/2];
    }
}
// ===== ENDE OPTIMIERTE BERECHNUNG =====

void setup() {
    Serial.begin(1000000); // 1 Mbit/s für Echtzeit
    while (!Serial) { delay(10); } // Warte auf Serial
    
    // I2C mit stabiler Geschwindigkeit (800 kHz ist sicherer als 1 MHz)
    Wire.begin(21, 22); // SDA=21, SCL=22 (ESP32 DevKitC V4)
    Wire.setClock(800000); // 800 kHz statt 1 MHz (stabiler!)
    
    if (!mlx.begin(MLX90640_I2CADDR_DEFAULT, &Wire)) {
        Serial.println("{\"error\":\"Sensor Init Failed\"}");
        while(1) { delay(1000); }
    }
    
    // Optimale Einstellungen (Paper: 16 Hz, aber 8 Hz stabiler für ESP32)
    mlx.setMode(MLX90640_CHESS); // Schachbrett-Modus (Paper Section 3.1)
    mlx.setResolution(MLX90640_ADC_18BIT); // 18-Bit Auflösung
    mlx.setRefreshRate(MLX90640_8_HZ); // 8 Hz (Paper: 16 Hz, aber ESP32 Serial limitiert)
    
    Serial.println("{\"status\":\"ESP32 MLX90640 Ready\"}");
    delay(100); // Sensor stabilisieren
}

void loop() {
    unsigned long start = millis();
    
    // Frame mit Fehlerbehandlung lesen
    int status = mlx.getFrame(frame);
    
    if (status != 0) {
        Serial.println("{\"error\":\"Frame Read Failed\"}");
        delay(50);
        return;
    }
    
    // Ambient-Temperatur berechnen (Paper-konform)
    float ambientTemp = calculateAmbientTemp(frame, 768);
    
    // JSON senden (optimiert)
    Serial.print("{\"temperature\":[");
    for (int i = 0; i < 768; i++) {
        Serial.print(frame[i], 2); // 2 Nachkommastellen für Genauigkeit
        if (i < 767) Serial.print(",");
    }
    Serial.print("],\"at\":");
    Serial.print(ambientTemp, 2);
    Serial.println("}");
    
    // Timing-Kontrolle für 8 Hz (125 ms pro Frame)
    unsigned long elapsed = millis() - start;
    if (elapsed < 125) {
        delay(125 - elapsed);
    }
}