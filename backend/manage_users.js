require('dotenv').config({ path: 'backend/.env.development' });
const mongoose = require('mongoose');
const User = require('./models/User');
const Farm = require('./models/Farm');
const Zone = require('./models/Zone');
const Livestock = require('./models/Livestock');

(async () => {
    try {
        await mongoose.connect(process.env.MONGO_URI);
        const brokenEmail = 'varshananand31@gmail.com';
        
        // 1. Delete broken user
        const brokenUser = await User.findOne({ email: brokenEmail });
        if (brokenUser) {
            console.log(`\nDeleting broken user: ${brokenEmail} (ID: ${brokenUser._id})`);
            await User.deleteOne({ _id: brokenUser._id });
            await Farm.deleteMany({ userId: brokenUser._id });
            await Zone.deleteMany({ farmId: { $in: await Farm.find({ userId: brokenUser._id }).distinct('_id') } });
            await Livestock.deleteMany({ userId: brokenUser._id });
            console.log(`Successfully deleted ${brokenEmail} and any orphaned farm data.`);
        } else {
            console.log(`\nUser ${brokenEmail} already deleted or not found.`);
        }

        // 2. List all registered users
        console.log(`\n=== REGISTERED USERS ===`);
        const users = await User.find({});
        console.log(`Total Users: ${users.length}`);
        
        for (const user of users) {
            const animals = await Livestock.countDocuments({ userId: user._id });
            const farms = await Farm.countDocuments({ userId: user._id });
            console.log(`\n- Name: ${user.name}`);
            console.log(`  Email: ${user.email}`);
            console.log(`  Phone: ${user.phone || 'N/A'}`);
            console.log(`  DOB: ${user.dob ? new Date(user.dob).toLocaleDateString() : 'N/A'}`);
            console.log(`  Role: ${user.role}`);
            console.log(`  Total Farms: ${farms}`);
            console.log(`  Total Animals: ${animals}`);
        }
        
        process.exit(0);
    } catch (err) {
        console.error("Error:", err);
        process.exit(1);
    }
})();
