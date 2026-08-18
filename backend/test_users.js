const mongoose = require('mongoose');
const User = require('./models/User');
require('dotenv').config();

mongoose.connect(process.env.MONGODB_URI || "mongodb://127.0.0.1:27017/livestock-monitoring")
    .then(async () => {
        const users = await User.find({});
        console.log(`Found ${users.length} users.`);
        users.forEach(u => console.log(`Name: ${u.name}, Role: ${u.role}, Phone: ${u.phone}`));
        process.exit(0);
    }).catch(e => { console.error(e); process.exit(1); });
