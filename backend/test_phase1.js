const mqtt = require('mqtt');

// Connect to Local MQTT Broker
const MQTT_URL = 'mqtt://127.0.0.1:1883';
const client = mqtt.connect(MQTT_URL);

client.on('connect', () => {
    console.log(`🟢 Test Script Connected to MQTT Broker (${MQTT_URL}).\n`);

    const mockHardwareId = 'COW001';
    const topic = `gomata/${mockHardwareId}/telemetry`;

    console.log(`📤 Publishing Mock IoT Data as a continuous stream to: ${topic}`);
    console.log(`\n--- VERIFICATION CHECKLIST ---`);
    console.log(`1. Check your Backend Terminal. You should see Logs if you added console.logs to the EventBus.`);
    console.log(`2. Check your React App Browser Console. You should see "Live Telemetry Received:" continuously.`);
    console.log(`Open the dashboard. The COW001 row will jump to the live parameters instantly! Press Ctrl+C to stop.\n`);

    setInterval(() => {
        // Create a mock payload with slight variations for visual effect
        const payload = JSON.stringify({
            temperature: (39.5 + (Math.random() * 0.4 - 0.2)).toFixed(1), // Fluctuating around 39.5°C
            heartRate: Math.floor(85 + (Math.random() * 6 - 3)),          // Fluctuating around 85 bpm
            lat: 28.5355,
            lng: 77.3910,
            battery: 88
        });

        console.log(`[STREAM] -> ${payload}`);
        client.publish(topic, payload);
    }, 2000);
});

client.on('error', (err) => {
    console.error("❌ MQTT Error:", err);
    process.exit(1);
});
