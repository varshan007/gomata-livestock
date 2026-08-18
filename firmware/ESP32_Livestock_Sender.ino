#include <WiFi.h>
#include <PubSubClient.h>
#include <TinyGPS++.h>
#include <OneWire.h>
#include <DallasTemperature.h>

// ============================================
// CONFIGURATION
// ============================================

// WiFi Credentials
const char* ssid = "OnePlus";
const char* password = "Nitin@000";

// MQTT Broker Settings - MATCHING YOUR BACKEND
const char* mqtt_server = "54.36.178.49"; // Corrected IP from your backend
const int mqtt_port = 1883;
const char* mqtt_topic = "livestock/sensor/ESP32_002"; // Topic for Aman

// Device Configuration
const char* deviceId = "ESP32_002"; // ID for "Aman"

// Pin Definitions
#define GPS_RX_PIN 16       // GPS TX connects here
#define GPS_TX_PIN 17       // GPS RX connects here
#define TEMP_SENSOR_PIN 4   // DS18B20 Data pin

// Update Interval
const long updateInterval = 30000; // 30 seconds

// ============================================
// GLOBAL OBJECTS
// ============================================

WiFiClient espClient;
PubSubClient mqttClient(espClient);
TinyGPSPlus gps;
HardwareSerial gpsSerial(2);

OneWire oneWire(TEMP_SENSOR_PIN);
DallasTemperature tempSensor(&oneWire);

unsigned long lastUpdate = 0;
float latitude = 0.0;
float longitude = 0.0;
float temperature = 0.0;
int batteryLevel = 0;
int dataCount = 0;

// ============================================
// SETUP
// ============================================

void setup() {
  Serial.begin(115200);
  delay(1000);
  
  Serial.println("\n\n=================================");
  Serial.println("🐄 ESP32 Livestock Monitor - Unit: Aman");
  Serial.println("=================================\n");

  // GPS
  gpsSerial.begin(9600, SERIAL_8N1, GPS_RX_PIN, GPS_TX_PIN);
  Serial.println("✓ GPS Serial Started");

  // Temp Sensor
  tempSensor.begin();
  if (tempSensor.getDeviceCount() > 0) {
    Serial.println("✓ Temp Sensor Detected");
  } else {
    Serial.println("✗ WARNING: Temp Sensor NOT Found");
  }

  // WiFi
  connectWiFi();

  // MQTT
  mqttClient.setServer(mqtt_server, mqtt_port);
  connectMQTT();
}

// ============================================
// LOOP
// ============================================

void loop() {
  if (WiFi.status() != WL_CONNECTED) connectWiFi();
  if (!mqttClient.connected()) connectMQTT();
  mqttClient.loop();

  // GPS Read
  while (gpsSerial.available() > 0) {
    gps.encode(gpsSerial.read());
  }

  // Timer
  if (millis() - lastUpdate >= updateInterval) {
    lastUpdate = millis();
    readSensorsAndSend();
  }
  
  delay(10);
}

// ============================================
// HELPERS
// ============================================

void connectWiFi() {
  Serial.print("Connecting to WiFi: ");
  Serial.println(ssid);
  WiFi.mode(WIFI_STA);
  WiFi.begin(ssid, password);
  
  while (WiFi.status() != WL_CONNECTED) {
    delay(500);
    Serial.print(".");
  }
  Serial.println("\n✓ WiFi Connected");
  Serial.print("IP: "); Serial.println(WiFi.localIP());
}

void connectMQTT() {
  while (!mqttClient.connected()) {
    Serial.print("Connecting to MQTT... ");
    String clientId = "ESP32_Client_";
    clientId += String(random(0xffff), HEX);
    
    if (mqttClient.connect(clientId.c_str())) {
      Serial.println("✓ Connected");
    } else {
      Serial.print("Failed, rc=");
      Serial.print(mqttClient.state());
      Serial.println(" retrying in 5s");
      delay(5000);
    }
  }
}

void readSensorsAndSend() {
  Serial.println("\n--- Reading Sensors ---");

  // GPS
  if (gps.location.isValid()) {
    latitude = gps.location.lat();
    longitude = gps.location.lng();
    Serial.println("✓ GPS Fix Valid");
  } else {
    Serial.println("⚠️ No GPS Fix (using 0.0 or last known)");
    // Keep last known or use dummy fallback for testing if 0.0
    if (latitude == 0.0) { 
        // Optional: remove this fallback for true production
        // latitude = 19.0760; longitude = 72.8777; 
    }
  }

  // Temp
  tempSensor.requestTemperatures();
  temperature = tempSensor.getTempCByIndex(0);
  if (temperature == -127.00) {
    Serial.println("⚠️ Temp Error");
    temperature = 0.0;
  } else {
    Serial.print("✓ Temp: "); Serial.println(temperature);
  }

  // Battery
  batteryLevel = random(80, 100); // Simulated for now

  // JSON
  String payload = "{";
  payload += "\"deviceId\":\"" + String(deviceId) + "\",";
  payload += "\"temperature\":" + String(temperature, 1) + ",";
  payload += "\"latitude\":" + String(latitude, 6) + ",";
  payload += "\"longitude\":" + String(longitude, 6) + ",";
  payload += "\"batteryLevel\":" + String(batteryLevel);
  payload += "}";

  // Send
  Serial.print("Sending: "); Serial.println(payload);
  if (mqttClient.publish("livestock/sensor/ESP32_002", payload.c_str())) {
    Serial.println("✓ Data Sent Successfully");
  } else {
    Serial.println("✗ MQTT Publish Failed");
  }
}
