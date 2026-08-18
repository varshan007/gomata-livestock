const mongoose = require('mongoose');
require('dotenv').config();
const User = require('./models/User');

mongoose.connect(process.env.MONGODB_URI || 'mongodb://localhost:27017/livestock-monitoring')
    .then(async () => {
        console.log('Connected to DB...');

        const email = 'kanna123@gmail.com';
        const password = '123456';

        // Check if exists
        const existing = await User.findOne({ email });
        if (existing) {
            console.log('User already exists. Updating password...');
            existing.password = password; // Request prompt re-hashing if logic allows, but better to delete and re-create to trigger pre-save hook cleanly
            await User.deleteOne({ email });
        }

        // Create fresh
        await User.create({
            name: 'Kanna',
            email: email,
            password: password, // Logic in model will hash this
            farm: {
                name: 'Kanna\'s Farm',
                location: {
                    address: '123 Farm Lane',
                    city: 'Pune',
                    state: 'MH',
                    country: 'India'
                },
                livestockType: 'Mixed',
                livestockCount: 50
            }
        });

        console.log(`✅ Restored User: ${email} / ${password}`);
        process.exit(0);
    })
    .catch(err => {
        console.error(err);
        process.exit(1);
    });
