const mongoose = require('mongoose');
const Livestock = require('../models/Livestock');
const SensorData = require('../models/SensorData');
const Alert = require('../models/Alert');
const User = require('../models/User');
const Farm = require('../models/Farm');
const Zone = require('../models/Zone');
require('dotenv').config({ path: '../.env' });

mongoose.connect(process.env.MONGO_URI || 'mongodb://127.0.0.1:27017/livestock_monitoring', {
    useNewUrlParser: true,
    useUnifiedTopology: true
})
.then(() => console.log('✅ Connected to MongoDB'))
.catch(err => console.error('❌ MongoDB Connection Error:', err));

async function seedProduction() {
    try {
        console.log('🧹 Clearing existing production data...');
        await Livestock.deleteMany({});
        await SensorData.deleteMany({});
        await Alert.deleteMany({});
        await Farm.deleteMany({});
        await Zone.deleteMany({});
        // Leave the User admin alone if it exists, but we'll add Vets.

        const adminEmail = process.env.SEED_ADMIN_EMAIL;
        const adminPassword = process.env.SEED_ADMIN_PASSWORD;

        if (!adminEmail || !adminPassword) {
            throw new Error("Missing required environment variables: SEED_ADMIN_EMAIL and SEED_ADMIN_PASSWORD must be provided to seed the production database.");
        }

        const adminUser = await User.findOne({ email: adminEmail }) || 
                          await User.create({ name: 'Admin', email: adminEmail, password: adminPassword, role: 'Admin' });

        console.log('🌾 Creating Farm & Zones...');
        const farm = await Farm.create({
            userId: adminUser._id,
            name: 'Green Valley Pastures',
            locationType: 'Polygon Mapping',
            geofence: { type: 'Polygon', coordinates: [[[72.87, 19.07], [72.88, 19.07], [72.88, 19.08], [72.87, 19.08], [72.87, 19.07]]] }
        });

        const zone = await Zone.create({
            farmId: farm._id,
            name: 'North Grazing Field',
            locationType: 'Polygon Mapping',
            geofence: { type: 'Polygon', coordinates: [[[72.87, 19.07], [72.875, 19.07], [72.875, 19.075], [72.87, 19.075], [72.87, 19.07]]] }
        });

        console.log('🐄 Creating 15 Cattle...');
        const breeds = ['Holstein', 'Jersey', 'Angus', 'Hereford', 'Brahman'];
        const livestockList = [];
        for (let i = 1; i <= 15; i++) {
            const isBessie = (i === 1);
            const ls = await Livestock.create({
                tagNumber: isBessie ? 'BESSIE-001' : `TAG-${1000 + i}`,
                name: isBessie ? 'Bessie' : `Cow ${i}`,
                breed: breeds[i % breeds.length],
                age: Math.floor(Math.random() * 5) + 1,
                weight: Math.floor(Math.random() * 200) + 400,
                gender: 'Female',
                deviceId: isBessie ? 'ESP32_BESSIE' : `ESP32_${2000 + i}`,
                status: 'Active',
                farmId: farm._id,
                zoneId: zone._id
            });
            livestockList.push(ls);
        }

        console.log('📈 Generating Historical Data (30 days)...');
        // Generate 30 days of data, 1 reading per hour
        const now = new Date();
        const thirtyDaysAgo = new Date(now.getTime() - (30 * 24 * 60 * 60 * 1000));
        
        let sensorDocs = [];
        for (const ls of livestockList) {
            for (let hours = 0; hours < 30 * 24; hours++) {
                const timestamp = new Date(thirtyDaysAgo.getTime() + (hours * 60 * 60 * 1000));
                const hourOfDay = timestamp.getHours();
                
                // Normal sine wave pattern
                let temp = 38.0 + Math.sin(hourOfDay * Math.PI / 12) * 0.5 + (Math.random() * 0.2 - 0.1);
                let hr = 60 + Math.sin(hourOfDay * Math.PI / 12) * 10 + (Math.random() * 5 - 2.5);

                // BESSIE'S FEVER: Last 6 hours temp spikes up to 40.5
                if (ls.name === 'Bessie' && hours > (30 * 24 - 6)) {
                    temp = 39.5 + (hours - (30 * 24 - 6)) * 0.2; // Rises steadily
                    hr += 20; // Elevated heart rate
                }

                sensorDocs.push({
                    livestockId: ls._id,
                    deviceId: ls.deviceId,
                    timestamp: timestamp,
                    temperature: parseFloat(temp.toFixed(2)),
                    heartRate: parseFloat(hr.toFixed(2)),
                    activityLevel: 0.5,
                    latitude: 19.07 + Math.random()*0.01,
                    longitude: 72.87 + Math.random()*0.01,
                    batteryLevel: 90 - (hours % 24)
                });

                // Batch insert to avoid memory overload
                if (sensorDocs.length > 5000) {
                    await SensorData.insertMany(sensorDocs);
                    sensorDocs = [];
                }
            }
        }
        if (sensorDocs.length > 0) {
            await SensorData.insertMany(sensorDocs);
        }

        console.log('🚨 Generating AI Alert for Bessie...');
        const bessie = livestockList.find(l => l.name === 'Bessie');
        await Alert.create({
            livestockId: bessie._id,
            alertType: 'Health',
            severity: 'High',
            message: 'AI Early Warning: Consistent temperature rise detected over the last 6 hours (up to 40.5°C). High risk of fever/infection.',
            status: 'New',
            timestamp: now
        });

        console.log('✅ Production Seed Complete!');
        process.exit(0);

    } catch (error) {
        console.error('❌ Error Seeding Data:', error);
        process.exit(1);
    }
}

seedProduction();
