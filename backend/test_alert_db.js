const Alert = require('./models/Alert');
const mongoose = require('mongoose');
require('dotenv').config();

mongoose.connect(process.env.MONGODB_URI || "mongodb://127.0.0.1:27017/livestock-monitoring")
  .then(async () => {
        try {
            console.log("Attempting direct Alert creation...");
            const newAlert = await Alert.create({
                livestockId: "679803bfcbefdb005b63edba",
                alertType: 'Temperature', 
                severity: 'Critical',
                message: "Test constraint check",
                actionRequired: 'Test action',
                sensorMetrics: {}
            });
            console.log("Success:", newAlert);
        } catch(e) {
            console.log("Mongoose Validation Error:", e);
        }
        process.exit();
}).catch(e => console.error(e));
