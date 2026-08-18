const mongoose = require('mongoose');
const User = require('./models/User');
require('dotenv').config();

mongoose.connect(process.env.MONGODB_URI || 'mongodb://localhost:27017/livestock-monitoring') // Try hyphen first as per server.js default
    .then(async () => {
        console.log('Connected to DB...');
        const email = 'kanna123@gmail.com';
        const password = '123456';

        const user = await User.findOne({ email });
        if (!user) {
            console.log('❌ User not found:', email);
        } else {
            console.log('✅ User found:', user.name);
            console.log('   Stored Password Hash starts with:', user.password.substring(0, 10) + '...');

            const isMatch = await user.matchPassword(password);
            if (isMatch) {
                console.log('🎉 PASSWORD MATCH SUCCESS!');
            } else {
                console.log('❌ PASSWORD MATCH FAILED.');
            }
        }
        process.exit(0);
    })
    .catch(err => {
        console.error(err);
        process.exit(1);
    });
