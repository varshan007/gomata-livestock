const Redis = require('ioredis');
const subscriber = new Redis('redis://127.0.0.1:6379', { enableReadyCheck: false });

subscriber.subscribe('telemetry:normalized', 'alert:create', 'alert:saved', (err) => {
    if (err) console.error(err);
    else console.log('✅ Listening for backend AI Alert events...');
});

subscriber.on('message', (channel, message) => {
    console.log(`\n--- [EVENT: ${channel}] ---`);
    console.log(JSON.parse(message));
});

setTimeout(() => {
    console.log('Terminating sniffer.');
    process.exit();
}, 15000);
