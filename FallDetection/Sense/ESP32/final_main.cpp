#include <Arduino.h>
#include <Wire.h>
#include <Adafruit_MLX90640.h>

Adafruit_MLX90640 mlx;
float frame[32*24]; // 768 Pixel

// ===== PAPER-KONFORME AMBIENT-TEMPERATUR (Section 2.2) =====
// "We denote the lowest 30% measurements as T_0.3, 
//  and use the median of T_0.3 as the ambient temperature T_a"
float calculateAmbientTemp(const float* pixels, int size) {
    const int n_lowest = (int)(size * 0.3f); // 30% von 768 = 230
    static float lowest_vals[256]; // Genug für 230 Werte
    
    // Initialisiere mit ersten n_lowest Werten
    for (int i = 0; i < n_lowest && i < size; i++) {
        lowest_vals[i] = pixels[i];
    }
    
    // Sortiere initial (Bubble Sort für kleine Arrays OK)
    for (int i = 0; i < n_lowest - 1; i++) {
        for (int j = 0; j < n_lowest - i - 1; j++) {
            if (lowest_vals[j] > lowest_vals[j + 1]) {
                float temp = lowest_vals[j];
                lowest_vals[j] = lowest_vals[j + 1];
                lowest_vals[j + 1] = temp;
            }
        }
    }
    
    // Prüfe restliche Pixel und sortiere ein
    for (int i = n_lowest; i < size; i++) {
        if (pixels[i] < lowest_vals[n_lowest - 1]) {
            int pos = n_lowest - 1;
            while (pos > 0 && pixels[i] < lowest_vals[pos - 1]) {
                lowest_vals[pos] = lowest_vals[pos - 1];
                pos--;
            }
            lowest_vals[pos] = pixels[i];
        }
    }
    
    // Median der kältesten 30% berechnen
    if (n_lowest % 2 == 0) {
        return (lowest_vals[n_lowest/2 - 1] + lowest_vals[n_lowest/2]) / 2.0f;
    } else {
        return lowest_vals[n_lowest/2];
    }
}
// ===== ENDE AMBIENT-TEMPERATUR =====

void setup() {
    Serial.begin(1000000); // 1 Mbit/s
    while (!Serial) { delay(10); }
    
    // I2C Init (800 kHz für Stabilität)
    Wire.begin(21, 22); // SDA=21, SCL=22 (ESP32 DevKitC V4)
    Wire.setClock(800000);
    
    if (!mlx.begin(MLX90640_I2CADDR_DEFAULT, &Wire)) {
        Serial.println("{\"error\":\"Sensor Init Failed\"}");
        while(1) { delay(1000); }
    }
    
    // ===== PAPER-KONFORME EINSTELLUNGEN (Section 4) =====
    mlx.setMode(MLX90640_CHESS);           // Chessboard pattern
    mlx.setResolution(MLX90640_ADC_18BIT); // 18-bit ADC
    mlx.setRefreshRate(MLX90640_8_HZ);     // 8 Hz (Paper: 16 Hz, aber stabiler)
    
    // NOTE: Emissivity ist hardware-kalibriert (ε≈0.95 für Haut)
    // Die Adafruit Library unterstützt KEINE manuelle Emissivity-Einstellung
    // Das ist OK, weil der Sensor factory-kalibriert ist!
    
    Serial.println("{\"status\":\"ESP32 MLX90640 Ready (Paper-compliant)\"}");
    delay(100); // Sensor stabilisieren
}

void loop() {
    unsigned long start = millis();
    
    // Frame lesen mit Fehlerbehandlung
    int status = mlx.getFrame(frame);
    
    if (status != 0) {
        Serial.println("{\"error\":\"Frame Read Failed\"}");
        delay(50);
        return;
    }
    
    // ===== PAPER-KONFORM: Ambient-Temperatur aus kältesten 30% =====
    float ambientTemp = calculateAmbientTemp(frame, 768);
    
    // ===== JSON SENDEN (KEINE EMISSIVITY-KORREKTUR!) =====
    // Die frame[] Daten sind bereits korrekt kalibriert vom Sensor!
    Serial.print("{\"temperature\":[");
    for (int i = 0; i < 768; i++) {
        // WICHTIG: Keine Korrektur! Sensor liefert bereits korrekte Werte
        Serial.print(frame[i], 2); // 2 Dezimalstellen
        if (i < 767) Serial.print(",");
    }
    Serial.print("],\"at\":");
    Serial.print(ambientTemp, 2);
    Serial.println("}");
    
    // Timing für 8 Hz (125 ms pro Frame)
    unsigned long elapsed = millis() - start;
    if (elapsed < 125) {
        delay(125 - elapsed);
    }
}