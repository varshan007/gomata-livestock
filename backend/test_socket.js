const { io } = require("socket.io-client");

const socket = io("http://localhost:8000", {
    reconnectionAttempts: 5,
    reconnectionDelay: 1000,
});

socket.on('connect', () => {
    console.log('✅ Mock Client Connected to Socket Tracker');
});

socket.on('sync:telemetry_live', (data) => {
    console.log('📡 RECEIVED LIVE TELEMETRY:', data);
});

socket.on('disconnect', () => {
    console.log('🔌 Disconnected.');
});

setTimeout(() => {
    console.log("Terminating tracking.");
    process.exit();
}, 10000);
