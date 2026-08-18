const path = require('path');
const envFile = process.env.NODE_ENV === 'production' ? '.env.production' : '.env.development';
require('dotenv').config({ path: path.join(__dirname, '../', envFile) });

const mongoose = require('mongoose');
const { queues } = require('../config/bullmq');

// Define minimal schema for Livestock to fetch tenantId and animalId
const LivestockSchema = new mongoose.Schema({
    userId: { type: mongoose.Schema.Types.ObjectId, required: true }
}, { collection: 'livestocks' });
const Livestock = mongoose.model('Livestock', LivestockSchema);

async function runBackfill() {
    try {
        console.log(`Connecting to MongoDB... (${process.env.MONGO_URI})`);
        await mongoose.connect(process.env.MONGO_URI, { useNewUrlParser: true, useUnifiedTopology: true });
        console.log("Connected.");

        console.log("Fetching livestock records...");
        const animals = await Livestock.find({}).lean();
        console.log(`Found ${animals.length} animals. Queueing backfill tasks...`);

        let count = 0;
        for (const animal of animals) {
            const tenantId = animal.userId ? animal.userId.toString() : "000000000000000000000000";
            const animalId = animal._id.toString();

            // Simulate a baseline telemetry payload to bootstrap the feature vector
            const initialTelemetry = {
                tenantId,
                animalId,
                temperature: 38.5,
                heartRate: 65,
                activity: 'still'
            };

            await queues.featureTelemetryExtract.add('backfill-telemetry', initialTelemetry);
            count++;
        }

        console.log(`Bootstrap complete. Enqueued ${count} telemetry extraction jobs.`);
        process.exit(0);
    } catch (e) {
        console.error("Backfill failed:", e);
        process.exit(1);
    }
}

runBackfill();
