const mongoose = require('mongoose');
const bcrypt = require('bcryptjs');
require('dotenv').config();

const userSchema = new mongoose.Schema({
    name: { type: String, required: true },
    email: { type: String, required: true, unique: true },
    password: { type: String, required: true },
    farm: {
        name: String,
        location: String,
        livestockType: String,
        livestockCount: Number
    }
});

const User = mongoose.model('User', userSchema);

mongoose.connect(process.env.MONGODB_URI || 'mongodb://localhost:27017/livestock_monitoring')
    .then(async () => {
        console.log('Connected to DB...');

        // delete existing test user if any
        await User.deleteOne({ email: 'admin@gomata.com' });

        // Hash the password since the inline schema doesn't have the pre-save hook
        const salt = await bcrypt.genSalt(10);
        const hashedPassword = await bcrypt.hash('password123', salt);
        await User.create({
            name: 'Demo Farmer',
            email: 'admin@gomata.com',
            password: hashedPassword,
            farm: {
                name: 'Green Valley Farm',
                location: 'Pune, India',
                livestockType: 'Mixed',
                livestockCount: 30
            }
        });

        console.log('✅ Created User: admin@gomata.com / password123');
        process.exit(0);
    })
    .catch(err => {
        console.error(err);
        process.exit(1);
    });
