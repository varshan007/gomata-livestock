require('dotenv').config();
const mongoose = require('mongoose');
const Redis = require('ioredis');
const Geofence = require('./models/Geofence');
const Livestock = require('./models/Livestock');
const crypto = require('crypto');

const bus = new Redis('redis://127.0.0.1:6379', { enableReadyCheck: false });

async function run() {
    await mongoose.connect(process.env.MONGODB_URI || 'mongodb://localhost:27017/livestock-monitoring');
    console.log('Connected to DB');

    // Create a mock cow
    const cow = await Livestock.findOneAndUpdate(
        { tagNumber: 'LOC_TEST_COW' },
        { tagNumber: 'LOC_TEST_COW', type: 'Cow', status: 'Healthy' },
        { upsert: true, new: true }
    );

    // Create a Safe polygon out in the ocean [long, lat] (0,0 to 1,1)
    await Geofence.deleteMany({ name: 'Ocean Safe Zone' });
    const fence = await Geofence.create({
        name: 'Ocean Safe Zone',
        type: 'Safe',
        shape: 'Polygon',
        livestockId: cow._id,
        coordinates: [
            { longitude: 0, latitude: 0 },
            { longitude: 1, latitude: 0 },
            { longitude: 1, latitude: 1 },
            { longitude: 0, latitude: 1 },
            { longitude: 0, latitude: 0 }
        ]
    });

    // Listen for the alert:create event
    const subscriber = new Redis('redis://127.0.0.1:6379', { enableReadyCheck: false });
    subscriber.subscribe('alert:create', (err) => {
        if (err) console.error(err);
        else console.log('Listening for alert:create...');
    });

    subscriber.on('message', (channel, message) => {
        console.log(`[RECEIVED ${channel}]`, JSON.parse(message));
        setTimeout(() => process.exit(0), 1000);
    });

    // Publish telemetry outside the safe zone (lat: 2, lng: 2)
    const payload = {
        traceId: crypto.randomUUID(),
        timestamp: new Date(),
        data: {
            hwId: 'LOC_TEST_COW',
            lat: 2,
            lng: 2
        }
    };

    console.log('Publishing out-of-bounds telemetry...');
    bus.publish('telemetry:received', JSON.stringify(payload));

    // Fallback exit if no alert received
    setTimeout(() => {
        console.log('Test timed out. No alert received.');
        process.exit(1);
    }, 5000);
}

run();
