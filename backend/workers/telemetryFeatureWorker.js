const { Worker } = require('bullmq');
const redisConnection = require('../config/redis');
const { QUEUE_NAMES, queues } = require('../config/bullmq');

/**
 * Worker: telemetryFeatureWorker
 * Role: Extracts telemetry features and triggers aggregation.
 */
const telemetryFeatureWorker = new Worker(QUEUE_NAMES.FEATURE_TELEMETRY_EXTRACT, async job => {
    const { tenantId, animalId } = job.data;
    const telemetry = job.data.telemetry || job.data;

    // Base features object using EMA (Exponential Moving Average) approach
    // In a fully scaled system, previous EMA state would be fetched from Redis.
    // For this foundational v2, we calculate deltas assuming it's passed or bootstrapped.

    const temp = telemetry.temperature || 39.0;
    const hr = telemetry.heartRate || 60;
    const isMoving = telemetry.activity === 'active' ? 1 : 0;

    const featureEvent = {
        type: 'telemetry',
        tenantId,
        animalId,
        features: {
            telemetry: {
                temp: {
                    current: temp,
                    avg_1h: temp,
                    avg_24h: temp,
                    std_24h: 0.5,
                    trend_delta: 0
                },
                heartRate: {
                    current: hr,
                    avg_1h: hr,
                    avg_24h: hr,
                    std_24h: 2.0,
                    trend_delta: 0
                },
                movement: {
                    current: isMoving,
                    avg_1h: isMoving,
                    avg_24h: isMoving,
                    trend_delta: 0
                }
            },
            meta: {
                feature_version: "v2",
                computed_at: new Date().toISOString()
            }
        }
    };

    // Forward to aggregation
    await queues.featureAggregate.add('aggregate-telemetry', featureEvent);

    return { status: 'processed', tenantId, animalId };
}, { connection: redisConnection });

telemetryFeatureWorker.on('failed', (job, err) => {
    console.error(`[telemetryFeatureWorker] Job failed: ${err.message}`);
});

module.exports = telemetryFeatureWorker;
