const mongoose = require('mongoose');
const TrainingEvent = require('./models/TrainingEvent');
require('dotenv').config({ path: '.env.production' });

async function runMigration() {
    await mongoose.connect(process.env.MONGO_URI);
    console.log("Connected to MongoDB for migration");

    // 1. Find all simulation_v2 events where correlatedSignals might be a boolean
    const eventsToUpdate = await TrainingEvent.find({
        source: 'simulation_v2',
        'metadata.correlatedSignals': { $in: [true, false] }
    });

    console.log(`Found ${eventsToUpdate.length} legacy boolean correlation records.`);

    let updatedCount = 0;
    for (const event of eventsToUpdate) {
        let newSignals = [];
        if (event.metadata.correlatedSignals === true) {
            if (event.eventType === 'fever') {
                newSignals = ['heartRate'];
            } else if (event.eventType === 'tachycardia') {
                newSignals = ['temperature'];
            } else if (event.eventType === 'stillness') {
                newSignals = ['heartRate', 'temperature'];
            } else {
                newSignals = ['unknown']; // fallback
            }
        }

        // Update the document
        await TrainingEvent.updateOne(
            { _id: event._id },
            { 
                $set: { 'metadata.correlatedSignals': newSignals }
            }
        );
        updatedCount++;
    }
    
    console.log(`Successfully normalized ${updatedCount} correlatedSignals to arrays.`);

    // 2. Run Aggregation to validate percentages
    const pipeline = [
        {
            $match: { source: 'simulation_v2', label: 1 }
        },
        {
            $group: {
                _id: "$metadata.intensity",
                totalAnomalies: { $sum: 1 },
                correlatedCount: {
                    $sum: {
                        $cond: [
                            { $gt: [{ $size: { $ifNull: ["$metadata.correlatedSignals", []] } }, 0] },
                            1,
                            0
                        ]
                    }
                }
            }
        },
        {
            $project: {
                intensity: "$_id",
                totalAnomalies: 1,
                correlatedCount: 1,
                correlationPercentage: {
                    $multiply: [
                        { $divide: ["$correlatedCount", "$totalAnomalies"] },
                        100
                    ]
                }
            }
        },
        { $sort: { correlationPercentage: 1 } }
    ];

    const stats = await TrainingEvent.aggregate(pipeline);
    console.log("\n--- CORRELATION STATISTICS ---");
    console.table(stats);

    await mongoose.disconnect();
}

runMigration().catch(console.error);
