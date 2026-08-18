const mqtt = require('mqtt');

// Connect to Local MQTT Broker
const MQTT_URL = 'mqtt://127.0.0.1:1883';
const client = mqtt.connect(MQTT_URL);

client.on('connect', () => {
    console.log(`🚨 Test Script Connected to MQTT Broker (${MQTT_URL}).\n`);

    const mockHardwareId = 'COW001';
    const topic = `gomata/${mockHardwareId}/telemetry`;

    console.log(`📤 Publishing SEVERE HEALTH SPIKE to: ${topic}`);
    console.log(`\n--- EXPECTED PIPELINE RUN ---`);
    console.log(`1. HealthAgent detects 41.5°C (Threshold > 39.5°C).`);
    console.log(`2. Gemini AI Diagnoses Severe Heat Stress / Fever.`);
    console.log(`3. AlertAgent saves to DB and locks Redis deduplication for 10 minutes.`);
    console.log(`4. NotificationDeliveryAgent Dispatches SMS via Twilio.`);
    console.log(`NOTE: If you run this script twice within 10 minutes, step 4 will NOT happen!\n`);

    // Create a mock payload with critical thresholds to force the AI trigger
    const payload = JSON.stringify({
        temperature: 41.5,   // Critical Fever Threshold
        heartRate: 110,      // Critical Tachycardia
        lat: 28.5355,
        lng: 77.3910,
        battery: 88
    });

    console.log(`[SPIKE] -> ${payload}`);
    client.publish(topic, payload);

    setTimeout(() => {
        console.log(`\n✅ Critical Payload sent. Watch the backend logs for the AI response and Twilio dispatch!`);
        process.exit(0);
    }, 2000); // Give MQTT a second to flush
});

client.on('error', (err) => {
    console.error("❌ MQTT Error:", err);
    process.exit(1);
});
