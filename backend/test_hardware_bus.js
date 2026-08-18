const Redis = require('ioredis');
const subscriber = new Redis('redis://127.0.0.1:6379', { enableReadyCheck: false });

subscriber.subscribe('telemetry:received', (err) => {
    if (err) console.error(err);
    else console.log('✅ Listening for: telemetry:received');
});

subscriber.on('message', (channel, message) => {
    console.log(`\n--- [EVENT: ${channel}] ---`);
    console.log(JSON.parse(message));
});

setTimeout(() => process.exit(), 15000);
