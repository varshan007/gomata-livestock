const { Worker } = require('bullmq');
const Redis = require('ioredis');
const { QUEUE_NAMES, queues } = require('../config/bullmq');

/**
 * Worker: episodeFeatureWorker
 * Role: Extracts training episode features and triggers aggregation.
 */
const episodeFeatureWorker = new Worker(QUEUE_NAMES.FEATURE_EPISODE_EXTRACT, async job => {
    const { tenantId, animalId, episodeData } = job.data;

    const featureEvent = {
        type: 'episode',
        tenantId,
        animalId,
        features: {
            episode: {
                active: episodeData.active || false,
                type: episodeData.type || 'normal',
                phase: episodeData.phase || 'monitoring',
                intensity: episodeData.intensity || 0,
                ticksElapsed: episodeData.ticksElapsed || 0,
                ticksRemaining: episodeData.ticksRemaining || null,
                progress: episodeData.progress || 0
            },
            meta: {
                feature_version: "v2",
                computed_at: new Date().toISOString()
            }
        }
    };

    // Forward to aggregation
    await queues.featureAggregate.add('aggregate-episode', featureEvent);

    return { status: 'processed', tenantId, animalId };
}, { connection: new Redis(process.env.REDIS_URL, { maxRetriesPerRequest: null, enableReadyCheck: false }) });

episodeFeatureWorker.on('failed', (job, err) => {
    console.error(`[episodeFeatureWorker] Job failed: ${err.message}`);
});

module.exports = episodeFeatureWorker;
