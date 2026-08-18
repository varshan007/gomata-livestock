const mongoose = require('mongoose');
const Livestock = require('./models/Livestock');
const SensorData = require('./models/SensorData');
const Alert = require('./models/Alert');
const User = require('./models/User');
const Farm = require('./models/Farm');
const Zone = require('./models/Zone');
require('dotenv').config();

mongoose.connect(process.env.MONGO_URI || 'mongodb://localhost:27017/livestock_monitoring', {
    useNewUrlParser: true,
    useUnifiedTopology: true
})
    .then(() => console.log('✅ Connected to MongoDB'))
    .catch(err => console.error('❌ MongoDB Connection Error:', err));

const BASE_LAT = 19.0760;
const BASE_LNG = 72.8777;

async function seedData() {
    try {
        console.log('🧹 Clearing existing data...');
        await Livestock.deleteMany({});
        await SensorData.deleteMany({});
        await Alert.deleteMany({});
        await Farm.deleteMany({});
        await Zone.deleteMany({});

        const user = await User.findOne({ email: 'admin@gomata.com' });
        if (!user) throw new Error("Admin user not found. Run seedUser.js first!");

        const farm = await Farm.create({
            userId: user._id,
            name: 'Demo Farm',
            locationType: 'Circular Mapping',
            geofence: { type: 'Point', coordinates: [72.8777, 19.0760], radius: 1000 }
        });

        const zone = await Zone.create({
            farmId: farm._id,
            name: 'Grazing Area',
            locationType: 'Circular Mapping',
            geofence: { type: 'Point', coordinates: [72.8777, 19.0760], radius: 500 }
        });

        console.log('🐄 Creating livestock...');
        const livestock1 = await Livestock.create({
            tagNumber: 'COW001',
            name: 'Raju',
            breed: 'Holstein',
            age: 3,
            weight: 450,
            gender: 'Female',
            deviceId: 'ESP32_001',
            status: 'Active',
            farmId: farm._id,
            zoneId: zone._id
        });

        const livestock2 = await Livestock.create({
            tagNumber: 'COW002',
            name: 'Aman',
            breed: 'Jersey',
            age: 2,
            weight: 380,
            gender: 'Female',
            deviceId: 'ESP32_002',
            status: 'Active',
            farmId: farm._id,
            zoneId: zone._id
        });

        const livestock3 = await Livestock.create({
            tagNumber: 'BULL001',
            name: 'Himanshu',
            breed: 'Angus',
            age: 4,
            weight: 650,
            gender: 'Male',
            deviceId: 'ESP32_003',
            status: 'Active',
            farmId: farm._id,
            zoneId: zone._id
        });

        const livestock4 = await Livestock.create({
            tagNumber: 'GOAT001',
            name: 'Billy',
            breed: 'Boer',
            age: 2,
            weight: 85,
            gender: 'Male',
            deviceId: 'ESP32_004',
            status: 'Active',
            farmId: farm._id,
            zoneId: zone._id
        });

        const livestock5 = await Livestock.create({
            tagNumber: 'SHEEP001',
            name: 'Dolly',
            breed: 'Merino',
            age: 3,
            weight: 70,
            gender: 'Female',
            deviceId: 'ESP32_005',
            status: 'Active',
            farmId: farm._id,
            zoneId: zone._id
        });

        console.log('📊 Creating sensor history (24 hours)...');

        // Helper to create history for a device
        const createHistory = async (livestockId, deviceId, baseTemp, variance) => {
            const history = [];
            const now = new Date();
            for (let i = 0; i < 24; i++) {
                const timestamp = new Date(now - i * 60 * 60 * 1000); // Hourly

                // Simulate daily temperature cycle
                const timeOfDay = timestamp.getHours();
                const cycle = Math.sin((timeOfDay - 6) * Math.PI / 12); // Peak at 12pm
                const temp = baseTemp + (cycle * 0.5) + (Math.random() * variance);

                history.push({
                    livestockId: livestockId,
                    deviceId: deviceId,
                    temperature: parseFloat(temp.toFixed(1)),
                    latitude: BASE_LAT + (Math.random() - 0.5) * 0.002, // Small movement
                    longitude: BASE_LNG + (Math.random() - 0.5) * 0.002,
                    batteryLevel: Math.max(10, 100 - (i * 0.5)), // Draining battery
                    timestamp: timestamp
                });
            }
            return SensorData.insertMany(history);
        };

        await createHistory(livestock1._id, 'ESP32_001', 38.5, 0.3); // Normal
        await createHistory(livestock2._id, 'ESP32_002', 39.8, 0.4); // High temp
        await createHistory(livestock3._id, 'ESP32_003', 38.0, 0.2); // Normal
        await createHistory(livestock4._id, 'ESP32_004', 39.0, 0.5); // Normal Goat Temp
        await createHistory(livestock5._id, 'ESP32_005', 39.2, 0.3); // Normal Sheep Temp

        console.log('🚨 Creating test alerts...');
        await Alert.create({
            userId: user._id,
            livestockId: livestock2._id,
            type: 'Temperature', // Ensure matches enum if validation exists
            alertType: 'Temperature', // For frontend display consistency
            severity: 'High',
            message: 'Elevated body temperature detected (39.8°C)',
            timestamp: new Date(),
            resolved: false
        });

        await Alert.create({
            userId: user._id,
            livestockId: livestock1._id,
            type: 'Geofence',
            alertType: 'Geofence',
            severity: 'Low',
            message: 'Livestock near boundary fence',
            timestamp: new Date(Date.now() - 3600000), // 1 hour ago
            resolved: true
        });

        console.log('\n✅ Database seeded successfully!');
        console.log('   - 3 Livestock');
        console.log('   - 72 Sensor Records');
        console.log('   - 2 Alerts');

        process.exit(0);
    } catch (error) {
        console.error('❌ Error seeding data:', error);
        process.exit(1);
    }
}

seedData();