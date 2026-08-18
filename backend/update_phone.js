const mongoose = require('mongoose');
const User = require('./models/User');
require('dotenv').config();

mongoose.connect(process.env.MONGODB_URI || "mongodb://127.0.0.1:27017/livestock-monitoring")
    .then(async () => {
        // 1. Give everyone Admin rights
        // 2. Force the correct Twilio Verified phone number format: +919600957236
        const result = await User.updateMany({}, {
            $set: {
                role: 'Admin',
                phone: '+919600957236'
            }
        });
        console.log(`Updated ${result.modifiedCount} users to Admin role and set phone to +919600957236.`);

        // Verify
        const users = await User.find({ role: 'Admin' });
        users.forEach(u => console.log(`Name: ${u.name}, Phone: ${u.phone}`));

        process.exit(0);
    }).catch(e => { console.error(e); process.exit(1); });
