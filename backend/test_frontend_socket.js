const { io } = require('socket.io-client');

const socket = io('http://127.0.0.1:8001', {
    transports: ['websocket', 'polling']
});

socket.on('connect', () => {
    console.log(`🔌 Connected to Backend Socket.IO Server (ID: ${socket.id})`);
    console.log('Listening for sync:telemetry_live and alert:new...');
});

socket.on('sync:telemetry_live', (data) => console.log('📡 [TELEMETRY]', data.hwId, data.temp));
socket.on('alert:new', (data) => {
    console.log('\n🚨 [ALERT RECEIVED ON SOCKET] 🚨');
    console.log(JSON.stringify(data, null, 2));
});

socket.on('connect_error', (err) => console.error('Connection Error:', err.message));

setTimeout(() => {
    console.log('Test complete.');
    process.exit(0);
}, 15000);
