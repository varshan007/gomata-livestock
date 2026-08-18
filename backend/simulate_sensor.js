const mqtt = require('mqtt');
const dns = require('dns');
const url = require('url');

// Configuration matches backend/services/mqttService.js
const MQTT_BROKER_URL_STRING = 'mqtt://broker.hivemq.com';
const parsedUrl = url.parse(MQTT_BROKER_URL_STRING);
const MQTT_HOSTNAME = parsedUrl.hostname;
const TOPIC = 'livestock/sensor/ESP32_001'; // Matches livestock/sensor/# wildcard

// Device ID for "Raju" (from seedData.js)
const DEVICE_ID = 'ESP32_001';

// Simulate movement and temperature changes
const baseLat = 19.0760;
const baseLng = 72.8777;

console.log(`🔍 Resolving MQTT broker: ${MQTT_HOSTNAME}`);

// Resolve hostname to IPv4 address manually to avoid IPv6 issues
dns.lookup(MQTT_HOSTNAME, { family: 4 }, (err, address, family) => {
    if (err) {
        console.error('❌ DNS Lookup error for MQTT broker:', err);
        return;
    }

    console.log(`✅ MQTT Broker resolved to: ${address}`);

    const client = mqtt.connect({
        host: address, // Use resolved IP
        protocol: 'mqtt',
        port: 1883,
        clientId: `livestock_sensor_${Math.random().toString(16).substr(2, 8)}`,
        clean: true,
        reconnectPeriod: 1000
    });

    client.on('connect', () => {
        console.log('✅ Connected to MQTT Broker');

        // Simulate data transmission every 10 seconds
        setInterval(() => {
            // Randomize data slightly
            const randomTemp = (38 + Math.random() * 2).toFixed(1); // 38.0 - 40.0
            // const randomLat = baseLat + (Math.random() - 0.5) * 0.001;
            // const randomLng = baseLng + (Math.random() - 0.5) * 0.001;
            const randomLat = baseLat; // keep stable for now
            const randomLng = baseLng;

            const battery = Math.floor(Math.random() * 20) + 80; // 80-100%

            const payload = {
                deviceId: DEVICE_ID,
                temperature: parseFloat(randomTemp),
                latitude: randomLat,
                longitude: randomLng,
                batteryLevel: battery
            };

            client.publish(TOPIC, JSON.stringify(payload));
            console.log(`📤 Sent data for ${DEVICE_ID}: Temp=${payload.temperature}°C, Lat=${payload.latitude.toFixed(6)}`);

        }, 5000); // Send every 5 seconds
    });

    client.on('error', (err) => {
        console.error('❌ MQTT Error:', err);
    });

    client.on('close', () => {
        console.log('🔌 MQTT Connection closed');
    });
});

