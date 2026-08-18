const mqtt = require('mqtt');

const client = mqtt.connect('mqtt://test.mosquitto.org');

client.on('connect', () => {
    console.log('✅ Connected to MQTT broker');

    // Simulate ESP32 sending data every 5 seconds
    setInterval(() => {
        const data = {
            deviceId: 'ESP32_001',
            temperature: 37.5 + Math.random() * 2, // Random temp between 37.5-39.5
            latitude: 19.0760 + (Math.random() - 0.5) * 0.001,
            longitude: 72.8777 + (Math.random() - 0.5) * 0.001,
            batteryLevel: 80 + Math.floor(Math.random() * 20)
        };

        client.publish('livestock/sensor/ESP32_001', JSON.stringify(data));
        console.log('📤 Published:', data);
    }, 5000);
});