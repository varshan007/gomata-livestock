/**
 * One-time script to clear old alerts with "Unknown" metadata.
 * Run: node scripts/clearUnknownAlerts.js
 */
require('dotenv').config();
const mongoose = require('mongoose');
const Alert = require('../models/Alert');

(async () => {
    await mongoose.connect(process.env.MONGO_URI);
    console.log('Connected to MongoDB');

    // Delete alerts with Unknown animalName or empty animalName
    const result = await Alert.deleteMany({
        $or: [
            { animalName: 'Unknown' },
            { animalName: '' },
            { animalName: { $exists: false } },
            { farmName: 'Unknown farm' }
        ]
    });

    console.log(`Deleted ${result.deletedCount} old "Unknown" alerts`);

    // Also clear any that have the old generic messages
    const result2 = await Alert.deleteMany({
        alertSource: 'ml_health_agent',
        animalName: { $in: ['Unknown', '', null] }
    });

    console.log(`Deleted ${result2.deletedCount} additional stale ML alerts`);

    await mongoose.disconnect();
    console.log('Done. Restart backend to generate fresh alerts with proper metadata.');
    process.exit(0);
})();
