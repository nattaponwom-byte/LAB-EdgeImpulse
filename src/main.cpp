
#include <Arduino.h>
#include <WiFi.h>
#include <PubSubClient.h>
#include <ArduinoJson.h>
#include <Wire.h>
#include <MPU6050_light.h>

#include <cstring>
#include <cmath>
#include <ESP32S3_MPU6050_Predictive_Maintenance_inferencing.h>

// ================= Wi-Fi / ThingsBoard =================
char WIFI_SSID[] = "Wokwi-GUEST";
char WIFI_PASSWORD[] = "";

const char* token = "";
const char* thingsboard_server = "thingsboard.cloud";
const int port = 1883;

const char STUDENT_ID[] = "";
const char FIRMWARE_VERSION[] = "1.0";

// ================= Hardware =================
#define pinLED 4
#define LED_ON LOW
#define LED_OFF HIGH

#define pinSDA 8
#define pinSCL 9
#define NUM_MPUS 1

MPU6050 mpu[NUM_MPUS] = { MPU6050(Wire) };
const int AD0pin[NUM_MPUS] = {5};

WiFiClient espClient;
PubSubClient client(espClient);

bool ledState = false;
bool machineAnomaly = false;

// ================= Model =================
const float ANOMALY_THRESHOLD = 0.60f;

static float features[EI_CLASSIFIER_DSP_INPUT_FRAME_SIZE];

// ================= Reconnect timers =================
unsigned long lastWiFiRetry = 0;
const unsigned long wifiRetryInterval = 500;

unsigned long lastMQTTRetry = 0;
const unsigned long mqttRetryInterval = 5000;

// ================= Function declarations =================
void onMessage(char* topic, byte* payload, unsigned int length);
void connectToWiFi();
void connectToMQTTBroker();
void setupMPU();
void SelectMPU(int selection);
void publishAttributes();
void publishAnomalyTelemetry(
    const char* machineState,
    const char* fault,
    float anomalyScore,
    float unbalanceProb
);
bool runPrediction();

// ========================================================
// SETUP
// ========================================================
void setup() {
  Serial.begin(115200);

  pinMode(pinLED, OUTPUT);
  digitalWrite(pinLED, LED_OFF);

  setupMPU();

  WiFi.mode(WIFI_STA);
  WiFi.begin(WIFI_SSID, WIFI_PASSWORD);

  Serial.print("Connecting to WiFi");
  unsigned long start = millis();

  while (WiFi.status() != WL_CONNECTED &&
         millis() - start < 10000) {
    delay(250);
    Serial.print(".");
  }

  Serial.println();

  if (WiFi.status() == WL_CONNECTED) {
    Serial.println("WiFi connected");
  } else {
    Serial.println("WiFi pending; retrying in loop");
  }

  client.setServer(thingsboard_server, port);
  client.setBufferSize(512);
  client.setCallback(onMessage);
}

// ========================================================
// LOOP
// ========================================================
void loop() {
  if (WiFi.status() != WL_CONNECTED) {
    connectToWiFi();
    return;
  }

  if (!client.connected()) {
    connectToMQTTBroker();
    return;
  }

  client.loop();

  // Collect sensor data and run inference.
  // No raw MPU telemetry is sent.
  runPrediction();
}

// ========================================================
// MPU6050
// ========================================================
void setupMPU() {
  Wire.begin(pinSDA, pinSCL);
  Wire.setTimeOut(1000);

  for (int i = 0; i < NUM_MPUS; i++) {
    pinMode(AD0pin[i], OUTPUT);
    digitalWrite(AD0pin[i], HIGH);
  }

  delay(500);

  for (int i = 0; i < NUM_MPUS; i++) {
    SelectMPU(i);

    byte status = mpu[i].begin();

    Serial.print("MPU6050 ");
    Serial.print(i);
    Serial.print(" init status = ");
    Serial.println(status);

    if (status != 0) {
      Serial.println("ERROR: MPU6050 initialization failed");
    }
  }

  Serial.println("MPU setup complete");
}

void SelectMPU(int selection) {
  for (int i = 0; i < NUM_MPUS; i++) {
    digitalWrite(AD0pin[i], i == selection ? LOW : HIGH);
  }
}

// ========================================================
// EDGE IMPULSE INFERENCE
// ========================================================
bool runPrediction() {
  const size_t frameSize = EI_CLASSIFIER_DSP_INPUT_FRAME_SIZE;
  const size_t axes = EI_CLASSIFIER_RAW_SAMPLES_PER_FRAME;

  if (axes != 3 || frameSize % 3 != 0) {
    Serial.println("ERROR: Model input is not 3-axis accelerometer");
    return false;
  }

  const size_t sampleCount = frameSize / axes;

  // Collect the model's required window at approximately 100 Hz.
  for (size_t i = 0; i < sampleCount; i++) {
    unsigned long sampleStart = millis();

    SelectMPU(0);
    mpu[0].update();

    features[i * 3]     = mpu[0].getAccX();
    features[i * 3 + 1] = mpu[0].getAccY();
    features[i * 3 + 2] = mpu[0].getAccZ();

    while (millis() - sampleStart < EI_CLASSIFIER_INTERVAL_MS) {
      delay(1);
    }
  }

  signal_t signal;

  int signalStatus = numpy::signal_from_buffer(
      features,
      frameSize,
      &signal
  );

  if (signalStatus != 0) {
    Serial.println("ERROR: Failed to create inference signal");
    return false;
  }

  ei_impulse_result_t result = {0};

  EI_IMPULSE_ERROR status = run_classifier(
      &signal,
      &result,
      false
  );

  if (status != EI_IMPULSE_OK) {
    Serial.print("ERROR: Inference failed: ");
    Serial.println((int)status);
    return false;
  }

  float normalProb = 0.0f;
  float unbalanceProb = 0.0f;

  for (size_t i = 0; i < EI_CLASSIFIER_LABEL_COUNT; i++) {
    const char* label = result.classification[i].label;
    float probability = result.classification[i].value;

    if (strcmp(label, "Normal_Operation") == 0) {
      normalProb = probability;
    } else if (
      strcmp(label, "Unbalance_In_Rotating_Parts") == 0
    ) {
      unbalanceProb = probability;
    }
  }

  // Derived score: unbalance probability minus the threshold.
  // Negative = below threshold; zero or positive = anomaly.
  float anomalyScore = unbalanceProb - ANOMALY_THRESHOLD;

  machineAnomaly = (unbalanceProb >= ANOMALY_THRESHOLD);

  // Active-LOW LED: LOW means ON.
  ledState = machineAnomaly;
  digitalWrite(pinLED, ledState ? LED_ON : LED_OFF);

  // Print status after every successful inference.
  Serial.print("Machine state: ");
  Serial.print(machineAnomaly ? "anomaly" : "normal");

  if (machineAnomaly) {
    Serial.print(" | fault=unbalance");
  }

  Serial.print(" | normal=");
  Serial.print(normalProb, 3);

  Serial.print(" unbalance=");
  Serial.print(unbalanceProb, 3);

  Serial.print(" anomaly_score=");
  Serial.println(anomalyScore, 3);

  // Only publish telemetry when anomalous.
  if (machineAnomaly) {
    publishAnomalyTelemetry(
        "anomaly",
        "unbalance",
        anomalyScore,
        unbalanceProb
    );
  }

  return true;
}

// ========================================================
// THINGSBOARD: ANOMALY TELEMETRY ONLY
// ========================================================
void publishAnomalyTelemetry(
    const char* machineState,
    const char* fault,
    float anomalyScore,
    float unbalanceProb
) {
  if (!client.connected()) {
    Serial.println("Telemetry skipped: MQTT disconnected");
    return;
  }

  JsonDocument doc;

  doc["machine_state"] = machineState;
  doc["fault"] = fault;
  doc["anomaly_score"] = anomalyScore;
  doc["unbalance_prob"] = unbalanceProb;

  char payload[256];
  size_t len = serializeJson(doc, payload, sizeof(payload));

  if (len == 0) {
    Serial.println("ERROR: Telemetry JSON serialization failed");
    return;
  }

  bool sent = client.publish(
      "v1/devices/me/telemetry",
      payload
  );

  Serial.print("Anomaly telemetry: ");
  Serial.println(payload);

  if (!sent) {
    Serial.println("ERROR: MQTT publish failed");
  }
}

// ========================================================
// RPC CALLBACK: setLed
// ========================================================
void onMessage(
    char* topic,
    byte* payload,
    unsigned int length
) {
  Serial.print("RPC received: ");
  Serial.println(topic);

  char responseTopic[128];
  strncpy(responseTopic, topic, sizeof(responseTopic) - 1);
  responseTopic[sizeof(responseTopic) - 1] = '\0';

  char* reqPtr = strstr(responseTopic, "request");

  if (reqPtr) {
    size_t tailLen = strlen(reqPtr + 7);
    memmove(reqPtr + 8, reqPtr + 7, tailLen + 1);
    memcpy(reqPtr, "response", 8);
  }

  JsonDocument doc;
  DeserializationError err = deserializeJson(
      doc,
      payload,
      length
  );

  if (err) {
    Serial.print("RPC JSON error: ");
    Serial.println(err.c_str());
    return;
  }

  const char* methodName = doc["method"];
  bool handled = false;

  if (methodName && strcmp(methodName, "setLed") == 0) {
    handled = true;

    // Accept RPC, but keep LED governed by anomaly detection.
    Serial.println(
      "RPC setLed received; LED is controlled by the model"
    );
  }

  JsonDocument response;
  response["success"] = handled;
  response["led_state"] = ledState;

  char responsePayload[96];
  serializeJson(response, responsePayload, sizeof(responsePayload));

  if (responseTopic[0] != '\0') {
    client.publish(responseTopic, responsePayload);
  }
}

// ========================================================
// WIFI RECONNECT
// ========================================================
void connectToWiFi() {
  if (millis() - lastWiFiRetry < wifiRetryInterval) {
    return;
  }

  lastWiFiRetry = millis();

  Serial.println("Retrying WiFi...");
  WiFi.reconnect();
}

// ========================================================
// MQTT RECONNECT
// ========================================================
void connectToMQTTBroker() {
  if (millis() - lastMQTTRetry < mqttRetryInterval) {
    return;
  }

  lastMQTTRetry = millis();

  Serial.println("Connecting to ThingsBoard...");

  if (client.connect("ESP32_Client", token, NULL)) {
    Serial.println("ThingsBoard MQTT connected");

    client.subscribe("v1/devices/me/rpc/request/+");
    publishAttributes();
  } else {
    Serial.print("MQTT connection failed, rc=");
    Serial.println(client.state());
  }
}

// ========================================================
// THINGSBOARD ATTRIBUTES
// ========================================================
void publishAttributes() {
  JsonDocument doc;

  doc["student_id"] = STUDENT_ID;
  doc["firmware_version"] = FIRMWARE_VERSION;
  doc["led_state"] = ledState;

  char payload[128];
  size_t len = serializeJson(doc, payload, sizeof(payload));

  if (len > 0) {
    client.publish("v1/devices/me/attributes", payload);
  }
}
