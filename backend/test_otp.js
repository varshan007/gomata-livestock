require('dotenv').config({ path: '.env.development' });
const mongoose = require('mongoose');
const OTP = require('./models/OTP');

(async () => {
    try {
        await mongoose.connect(process.env.MONGO_URI || 'mongodb://127.0.0.1:27017/livestock_monitoring');
        console.log("Connected to MongoDB");
        await OTP.create({ identifier: 'test@test.com', code: '123456', type: 'email' });
        console.log("OTP created successfully");
        process.exit(0);
    } catch (err) {
        console.error("Error:", err);
        process.exit(1);
    }
})();
