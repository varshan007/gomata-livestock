require('dotenv').config({ path: 'backend/.env.development' });
const mongoose = require('mongoose');
const User = require('./models/User');
const Livestock = require('./models/Livestock');

(async () => {
    try {
        await mongoose.connect(process.env.MONGO_URI);
        const email = 'varshananand31@gmail.com';
        const user = await User.findOne({ email });
        
        if (!user) {
            console.log(`User ${email} NOT FOUND in database.`);
        } else {
            console.log(`User ${email} FOUND (ID: ${user._id})`);
            const count = await Livestock.countDocuments({ userId: user._id });
            console.log(`Animals for this user: ${count}`);
        }
        process.exit(0);
    } catch (err) {
        console.error("Error:", err);
        process.exit(1);
    }
})();
