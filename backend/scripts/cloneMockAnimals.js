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

        // 1. Load existing mock animals
        // The user's system query uses isActive: true. Using fallback logic for LivestockMaster's device_status if schema varies.
        let sourceAnimals = await LivestockMaster.find({ isActive: true }).limit(10);

        if (sourceAnimals.length === 0) {
            sourceAnimals = await LivestockMaster.find({ device_status: { $ne: 'Unassigned' } }).limit(10);
        }

        if (sourceAnimals.length === 0) {
            logger.warn({ action: "CLONE_MOCK_ANIMALS" }, "No template animals found to clone.");
            process.exit(0);
        }

        const args = process.argv.slice(2);
        let cloneCount = 500;
        args.forEach(arg => {
            if (arg.startsWith('--count=')) {
                cloneCount = parseInt(arg.split('=')[1]);
            }
        });

        // 2. Clone each animal
        const clones = [];
        for (let i = 0; i < cloneCount; i++) {
            const template = sourceAnimals[i % sourceAnimals.length];

            const clone = {
                ...template.toObject(),
                _id: new mongoose.Types.ObjectId(),
                tagId: `CLONE-${Date.now()}-${i}`,
                deviceId: new mongoose.Types.ObjectId(), // From requested snippet

                // Mapped explicitly for required unique MongoDB constraints natively in the exact LivestockMaster schema
                livestock_id: `CLONE-LS-${Date.now()}-${i}`,
                device_id: `CLONE-DEV-${Date.now()}-${i}`,
                mapping_id: `CLONE-MAP-${Date.now()}-${i}`,

                isTurbo: true,
                clonedFrom: template._id,
                createdAt: new Date(),
                updatedAt: new Date() // Preserve structural timestamp fields
            };

            delete clone.__v;
            clones.push(clone);
        }

        // 3. Insert using batch insert bypassing strict schema validation
        await LivestockMaster.collection.insertMany(clones, { ordered: false });

        // 4. Log results
        logger.info({
            action: "MOCK_ANIMALS_CLONED",
            sourceCount: sourceAnimals.length,
            cloneCount: clones.length,
            tenantId: sourceAnimals.length > 0 ? sourceAnimals[0].userId : null,
            timestamp: new Date()
        });

        process.exit(0);
    } catch (e) {
        logger.error({ action: "MOCK_ANIMALS_CLONED", result: 'error', error: e.message });
        process.exit(1);
    }
}

run();
