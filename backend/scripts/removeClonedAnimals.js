const mongoose = require('mongoose');
const path = require('path');
const logger = require('../utils/logger');
const LivestockMaster = require('../models/LivestockMaster');

// Load appropriate environment configurations
const envFile = process.env.NODE_ENV === 'production' ? '.env.production'
    : process.env.NODE_ENV === 'staging' ? '.env.staging'
        : process.env.NODE_ENV === 'accelerated' ? '.env.accelerated'
            : '.env.development';
require('dotenv').config({ path: path.join(__dirname, '../', envFile) });

async function run() {
    try {
        await mongoose.connect(process.env.MONGO_URI, { useNewUrlParser: true, useUnifiedTopology: true });

        // 1. Remove cloned animals safely
        const result = await LivestockMaster.collection.deleteMany({ isTurbo: true });

        // 2. Log result
        logger.info({
            action: "CLONED_ANIMALS_REMOVED",
            deletedCount: result.deletedCount,
            timestamp: new Date()
        });

        process.exit(0);
    } catch (e) {
        logger.error({ action: "CLONED_ANIMALS_REMOVED", result: 'error', error: e.message });
        process.exit(1);
    }
}

run();
