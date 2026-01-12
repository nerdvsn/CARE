#include <WiFi.h>
#include <WebSocketsServer.h>
#include "DFRobot_HumanDetection.h"

// WLAN Zugangsdaten
const char* ssid = "Vodafone-463C";
const char* password = "eghd3KEER6hy9XTsddddddddd";

DFRobot_HumanDetection hu(&Serial1);
WebSocketsServer webSocket(81);

void setup() {
  Serial.begin(115200);
  delay(1000);

  // UART zum Sensor (RX=16, TX=17)
  Serial1.begin(115200, SERIAL_8N1, 16, 17);

  // WLAN Verbindung
  WiFi.mode(WIFI_STA);
  WiFi.begin(ssid, password);
  while (WiFi.status() != WL_CONNECTED) {
    delay(500);
    Serial.print(".");
  }
  Serial.println("\n✅ WLAN verbunden!");
  Serial.print("IP: "); Serial.println(WiFi.localIP());

  webSocket.begin();

  // Radar Setup
  while (hu.begin() != 0) {
    Serial.println("Radar nicht gefunden...");
    delay(1000);
  }

  // Schlafmodus für Vitaldaten
  hu.configWorkMode(hu.eSleepMode);
  hu.sensorRet();
  delay(1000);
  Serial.println("System bereit!");
}

void loop() {
  webSocket.loop();

  // Wir fragen die Werte einzeln ab für maximale Aktualität
  int presence = hu.smHumanData(hu.eHumanPresence);
  int heart = hu.getHeartRate();
  int resp = hu.getBreatheValue();
  int mov = hu.smHumanData(hu.eHumanMovingRange);

  // JSON String bauen
  String json = "{";
  json += "\"presence\":" + String(presence) + ",";
  json += "\"heart\":" + String(heart) + ",";
  json += "\"resp\":" + String(resp) + ",";
  json += "\"mov\":" + String(mov);
  json += "}";

  Serial.println(json); // Debug im Serial Monitor
  webSocket.broadcastTXT(json); // Senden an Flutter

  delay(500); // 2 Messungen pro Sekunde
}