const mongoose = require('mongoose');
const User = require('./models/User');
require('dotenv').config();

mongoose.connect(process.env.MONGODB_URI || "mongodb://127.0.0.1:27017/livestock-monitoring")
    .then(async () => {
        const result = await User.updateMany({}, { $set: { role: 'Admin' } });
        console.log(`Updated ${result.modifiedCount} users to Admin role.`);
        process.exit(0);
    }).catch(e => { console.error(e); process.exit(1); });
