const mongoose = require('mongoose');
const dotenv = require('dotenv');

// Load environment variables
dotenv.config({ path: '../.env' });

const MONGO_URI = process.env.MONGO_URI || 'mongodb://127.0.0.1:27017/livestock_monitoring';

async function resetDatabase() {
    try {
        console.log(`Connecting to MongoDB at ${MONGO_URI}...`);
        await mongoose.connect(MONGO_URI);
        console.log('✅ Connected successfully.');

        console.log('🧹 Wiping Collections...');

        const collectionsToDrop = ['users', 'livestocks', 'alerts', 'geofences', 'sensordatas', 'farms', 'zones'];
        const collections = await mongoose.connection.db.collections();

        for (let collection of collections) {
            if (collectionsToDrop.includes(collection.collectionName)) {
                console.log(`Dropping collection: ${collection.collectionName}`);
                await mongoose.connection.db.dropCollection(collection.collectionName);
            }
        }

        console.log('✅ Database reset complete! Clean slate.');
        process.exit(0);
    } catch (error) {
        console.error('❌ Error resetting database:', error);
        process.exit(1);
    }
}

resetDatabase();
