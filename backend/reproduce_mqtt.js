const mqtt = require('mqtt');

const MQTT_BROKER = 'mqtt://broker.hivemq.com';
const MQTT_PORT = 1883;

console.log('Testing MQTT connection...');

const options = {
    port: MQTT_PORT,
    clientId: `test_client_${Math.random().toString(16).substr(2, 8)}`,
    clean: true,
    reconnectPeriod: 1000,
    family: 4 // Force IPv4
};

console.log('Connecting with options:', options);

const client = mqtt.connect(MQTT_BROKER, options);

client.on('connect', () => {
    console.log('✅ Connected successfully!');
    client.end();
});

client.on('error', (err) => {
    console.error('❌ Connection error:', err);
    client.end();
});

client.on('close', () => {
    console.log('🔌 Connection closed');
});
