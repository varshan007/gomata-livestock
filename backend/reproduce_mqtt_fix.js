const mqtt = require('mqtt');

console.log('Testing MQTT connection with FIX...');

const options = {
    host: 'broker.hivemq.com',
    protocol: 'mqtt',
    port: 1883,
    clientId: `test_client_fix_${Math.random().toString(16).substr(2, 8)}`,
    clean: true,
    reconnectPeriod: 1000,
    family: 4 // Force IPv4
};

console.log('Connecting with options only:', options);

// Pass options only, no URL string
const client = mqtt.connect(options);

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
