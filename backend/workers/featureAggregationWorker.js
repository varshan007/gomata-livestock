const { Worker } = require('bullmq');
const redisConnection = require('../config/redis');
const { QUEUE_NAMES } = require('../config/bullmq');
const featureStoreClient = require('../services/featureStoreClient');

/**
 * Worker: featureAggregationWorker
 * Role: Merges extracted features, computes derived metrics, and writes to Redis online store.
 */
const featureAggregationWorker = new Worker(QUEUE_NAMES.FEATURE_AGGREGATE, async job => {
    const { type, tenantId, animalId, features: newFeatures } = job.data;
    const startTime = Date.now();

    // 1. Fetch current vector from Feature Store
    let currentVector = await featureStoreClient.getFeatures(tenantId, animalId);

    // 2. Initialize if null
    if (!currentVector) {
        currentVector = {
            telemetry: {},
            episode: {
                active: false, type: 'normal', phase: 'monitoring',
                intensity: 0, ticksElapsed: 0, ticksRemaining: null, progress: 0
            },
            correlation: {
                temp_heartRate: 0, temp_movement: 0, heartRate_movement: 0, confidence: 0
            },
            baseline: { temp: 39.0, heartRate: 60, movement: 0.5 },
            deviation: { temp_from_baseline: 0, hr_from_baseline: 0, movement_from_baseline: 0 },
            derived: {
                anomaly_probability: 0.0, recovery_probability: 1.0, forecast_1h_risk: 0.0,
                signal_strength: 0.9, noise_probability: 0.1
            },
            meta: { feature_version: "v2" }
        };
    }

    // 3. Merge new features based on event type
    if (type === 'telemetry') {
        currentVector.telemetry = { ...currentVector.telemetry, ...newFeatures.telemetry };
    } else if (type === 'episode') {
        currentVector.episode = { ...currentVector.episode, ...newFeatures.episode };
    }

    // 4. Compute deviations from baseline
    const temp = currentVector.telemetry.temp?.current || currentVector.baseline.temp;
    const hr = currentVector.telemetry.heartRate?.current || currentVector.baseline.heartRate;
    const move = currentVector.telemetry.movement?.current || currentVector.baseline.movement;

    currentVector.deviation.temp_from_baseline = temp - currentVector.baseline.temp;
    currentVector.deviation.hr_from_baseline = hr - currentVector.baseline.heartRate;
    currentVector.deviation.movement_from_baseline = move - currentVector.baseline.movement;

    // 5. Compute derived intelligence (Simplified placeholder logic)
    const anomalyFactor = Math.abs(currentVector.deviation.temp_from_baseline) * 0.2 +
        Math.abs(currentVector.deviation.hr_from_baseline) * 0.05;

    currentVector.derived.anomaly_probability = Math.min(1.0, anomalyFactor);
    currentVector.derived.forecast_1h_risk = currentVector.derived.anomaly_probability * 1.1; // Extrapolating risk

    // Update metadata
    currentVector.meta.computed_at = new Date().toISOString();
    currentVector.meta.latency_ms = Date.now() - startTime;

    // 6. Save to Redis
    await featureStoreClient.setFeatures(tenantId, animalId, currentVector);

    return { status: 'aggregated', tenantId, animalId, latency: currentVector.meta.latency_ms };

}, { connection: redisConnection });

featureAggregationWorker.on('failed', (job, err) => {
    console.error(`[featureAggregationWorker] Job failed: ${err.message}`);
});

module.exports = featureAggregationWorker;
