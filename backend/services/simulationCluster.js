const cluster = require('cluster');
const os = require('os');
const path = require('path');
const logger = require('../utils/logger');

// Load appropriate environment configurations
const envFile = process.env.NODE_ENV === 'production'
    ? '.env.production'
    : process.env.NODE_ENV === 'staging'
        ? '.env.staging'
        : process.env.NODE_ENV === 'accelerated'
            ? '.env.accelerated'
            : '.env.development';

require('dotenv').config({ path: path.join(__dirname, '../', envFile) });

if (cluster.isPrimary) {
    const cpuCount = os.cpus().length;

    logger.info({
        action: "SIMULATION_CLUSTER_START",
        workers: cpuCount
    });

    for (let i = 0; i < cpuCount; i++) {
        cluster.fork({
            WORKER_INDEX: i,
            TOTAL_WORKERS: cpuCount
        });
    }

    cluster.on('exit', (worker, code, signal) => {
        logger.warn({ action: 'SIMULATION_WORKER_EXIT', workerId: worker.id }, `Worker ${worker.id} died.`);
        // Optional: restart worker here if desired
    });
} else {
    // Worker Process
    const mongoose = require('mongoose');

    mongoose.connect(process.env.MONGO_URI, {
        useNewUrlParser: true,
        useUnifiedTopology: true
    }).then(() => {
        logger.info({ action: 'db_connect', service: 'simulation_worker' }, 'Worker connected to MongoDB');
        // Require and start the simulation service
        const simulationService = require('./HardwareSimulationService');
        simulationService.start();
    }).catch(err => {
        logger.error({ action: 'db_connect_error', error: err.message }, 'Failed to connect Worker to MongoDB');
        process.exit(1);
    });
}
