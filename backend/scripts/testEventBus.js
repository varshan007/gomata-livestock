const path = require('path');
require('dotenv').config({ path: path.join(__dirname, '../.env.development') });

const { queues, Worker } = require('../config/bullmq');
const redisConnection = require('../config/redis');

async function testEventBus() {
    try {
        console.log("Adding test job to ML_PREDICTIONS queue...");
        const job = await queues.mlPredictions.add('test-job', { foo: 'bar' });
        console.log(`Job added with ID: ${job.id}`);

        // Set up a temporary worker to consume it
        const worker = new Worker('ml-predictions-queue', async (j) => {
            console.log(`Processing job ${j.id}...`);
            return 'success';
        }, { connection: redisConnection });

        worker.on('completed', (j) => {
            console.log(`Job ${j.id} has completed!`);
            console.log("published + consumed successfully");
            worker.close();
            process.exit(0);
        });

        worker.on('failed', (j, err) => {
            console.error(`Job ${j.id} has failed with ${err.message}`);
            worker.close();
            process.exit(1);
        });

        // Timeout fallback
        setTimeout(() => {
            console.error("Timeout waiting for job to complete.");
            worker.close();
            process.exit(1);
        }, 5000);

    } catch (e) {
        console.error("Failed to test Event Bus:", e);
        process.exit(1);
    }
}

testEventBus();
