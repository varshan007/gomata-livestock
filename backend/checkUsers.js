const mongoose = require('mongoose');
require('dotenv').config();

const userSchema = new mongoose.Schema({
    name: String,
    email: String,
    password: String // We won't print this obviously, just checking existence
});

const User = mongoose.model('User', userSchema);

mongoose.connect(process.env.MONGODB_URI || 'mongodb://localhost:27017/livestock_monitoring')
    .then(async () => {
        console.log('Connected to DB...');
        const users = await User.find({});
        console.log(`Found ${users.length} users:`);
        users.forEach(u => console.log(` - ${u.name} (${u.email})`));
        process.exit(0);
    })
    .catch(err => {
        console.error(err);
        process.exit(1);
    });
