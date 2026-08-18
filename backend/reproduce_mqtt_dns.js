const mqtt = require('mqtt');
const dns = require('dns');

console.log('Testing MQTT connection with Manual DNS Resolution...');

dns.lookup('broker.hivemq.com', { family: 4 }, (err, address, family) => {
    if (err) {
        console.error('❌ DNS Lookup error:', err);
        return;
    }

    console.log(`✅ Resolved to IPv4: ${address}`);

    const options = {
        host: address, // Use resolved IP
        protocol: 'mqtt',
        port: 1883,
        clientId: `test_client_dns_${Math.random().toString(16).substr(2, 8)}`,
        clean: true,
        reconnectPeriod: 1000
    };

    console.log('Connecting with options:', options);

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
});
