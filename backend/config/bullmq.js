const { Queue, Worker } = require('bullmq');
const redisConnection = require('../config/redis');

// Centralize Queue definitions
const QUEUE_NAMES = {
    ML_PREDICTIONS: 'ml-predictions-queue',
    ALERTS_PROCESSING: 'alerts-processing-queue',
    FEATURE_TELEMETRY_EXTRACT: 'feature.telemetry.extract',
    FEATURE_EPISODE_EXTRACT: 'feature.episode.extract',
    FEATURE_AGGREGATE: 'feature.aggregate'
};

// Queue instances
const queues = {
    mlPredictions: new Queue(QUEUE_NAMES.ML_PREDICTIONS, {
        connection: redisConnection,
        defaultJobOptions: {
            removeOnComplete: { age: 3600, count: 5000 }, // Keep last 5000 completed jobs for 1 hour
            removeOnFail: { age: 24 * 3600, count: 1000 } // Keep last 1000 failed jobs for 24 hours
        }
    }),
    alertsProcessing: new Queue(QUEUE_NAMES.ALERTS_PROCESSING, {
        connection: redisConnection,
        defaultJobOptions: {
            removeOnComplete: { age: 3600, count: 5000 },
            removeOnFail: { age: 24 * 3600, count: 1000 }
        }
    }),
    featureTelemetryExtract: new Queue(QUEUE_NAMES.FEATURE_TELEMETRY_EXTRACT, { connection: redisConnection }),
    featureEpisodeExtract: new Queue(QUEUE_NAMES.FEATURE_EPISODE_EXTRACT, { connection: redisConnection }),
    featureAggregate: new Queue(QUEUE_NAMES.FEATURE_AGGREGATE, { connection: redisConnection })
};

module.exports = {
    QUEUE_NAMES,
    queues,
    // Export Worker class for job consumers to use
    Worker
};
