const mongoose = require('mongoose');
const Device = require('./models/Device');
require('dotenv').config();

const seedDevices = async () => {
    try {
        await mongoose.connect(process.env.MONGO_URI || 'mongodb://localhost:27017/livestock_monitoring');
        console.log('✅ Connected to DB');

        const deviceIds = [
            'GM-SN-1001', 'GM-SN-1002', 'GM-SN-1003', 'GM-SN-1004', 'GM-SN-1005',
            'GM-SN-1006', 'GM-SN-1007', 'GM-SN-1008', 'GM-SN-1009', 'GM-SN-1010'
        ];

        // Clear existing to avoid duplicates in seed running
        await Device.deleteMany({ deviceId: { $in: deviceIds } });

        const devices = deviceIds.map(id => ({
            deviceId: id,
            deviceType: 'Neck Collar',
            status: 'Unassigned',
            batteryLevel: 100,
            signalStrength: -50
        }));

        await Device.insertMany(devices);
        console.log(`✅ Successfully seeded ${deviceIds.length} hardware devices for testing.`);

    } catch (err) {
        console.error('❌ Error seeding devices:', err);
    } finally {
        mongoose.disconnect();
    }
};

seedDevices();
