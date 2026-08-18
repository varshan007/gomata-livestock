const { Queue, Worker } = require('bullmq');
const Redis = require('ioredis');

// Ensure BullMQ gets its own explicitly configured Redis connections
const bullmqRedisConfig = process.env.REDIS_URL;
const bullmqRedisOptions = { maxRetriesPerRequest: null, enableReadyCheck: false };

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
        connection: new Redis(bullmqRedisConfig, bullmqRedisOptions),
        defaultJobOptions: {
            removeOnComplete: { age: 3600, count: 5000 }, // Keep last 5000 completed jobs for 1 hour
            removeOnFail: { age: 24 * 3600, count: 1000 } // Keep last 1000 failed jobs for 24 hours
        }
    }),
    alertsProcessing: new Queue(QUEUE_NAMES.ALERTS_PROCESSING, {
        connection: new Redis(bullmqRedisConfig, bullmqRedisOptions),
        defaultJobOptions: {
            removeOnComplete: { age: 3600, count: 5000 },
            removeOnFail: { age: 24 * 3600, count: 1000 }
        }
    }),
    featureTelemetryExtract: new Queue(QUEUE_NAMES.FEATURE_TELEMETRY_EXTRACT, { connection: new Redis(bullmqRedisConfig, bullmqRedisOptions) }),
    featureEpisodeExtract: new Queue(QUEUE_NAMES.FEATURE_EPISODE_EXTRACT, { connection: new Redis(bullmqRedisConfig, bullmqRedisOptions) }),
    featureAggregate: new Queue(QUEUE_NAMES.FEATURE_AGGREGATE, { connection: new Redis(bullmqRedisConfig, bullmqRedisOptions) })
};

module.exports = {
    QUEUE_NAMES,
    queues,
    // Export Worker class for job consumers to use
    Worker
};
