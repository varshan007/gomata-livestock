const path = require('path');
const envFile = process.env.NODE_ENV === 'production'
    ? '.env.production'
    : process.env.NODE_ENV === 'staging'
        ? '.env.staging'
        : process.env.NODE_ENV === 'accelerated'
            ? '.env.accelerated'
            : '.env.development';

require('dotenv').config({ path: path.join(__dirname, '../', envFile) });

const { Worker, QUEUE_NAMES } = require('../config/bullmq');
const Redis = require('ioredis');
const mlServiceClient = require('./mlServiceClient');
const logger = require('../utils/logger');

const mlWorker = new Worker(QUEUE_NAMES.ML_PREDICTIONS, async job => {
    logger.info({ action: 'ml_job_start', jobId: job.id, animalId: job.data.animalId, service: 'bullmq' }, 'Processing ML prediction job');
    const startTime = Date.now();
    try {
        const result = await mlServiceClient.predict(job.data);
        const duration = Date.now() - startTime;
        logger.info({ action: 'ml_job_complete', jobId: job.id, animalId: job.data.animalId, duration, result: 'success', service: 'bullmq' }, 'ML prediction job completed successfully');
        return result;
    } catch (error) {
        const duration = Date.now() - startTime;
        logger.error({ action: 'ml_job_error', jobId: job.id, animalId: job.data.animalId, duration, result: 'error', error: error.message, service: 'bullmq' }, 'ML prediction job failed');
        throw error;
    }
}, { connection: new Redis(process.env.REDIS_URL, { maxRetriesPerRequest: null, enableReadyCheck: false }) });

mlWorker.on('completed', job => {
    logger.info({ action: 'worker_job_completed', jobId: job.id, service: 'bullmq' }, 'Job completed in worker');
});

mlWorker.on('failed', (job, err) => {
    logger.error({ action: 'worker_job_failed', jobId: job ? job.id : 'unknown', error: err.message, service: 'bullmq' }, 'Job failed in worker');
});

module.exports = mlWorker;
