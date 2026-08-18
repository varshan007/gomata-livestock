const mongoose = require('mongoose');
const User = require('./models/User');
const uri = "mongodb+srv://gomata_admin:IITB01%40GOOgle@gomata-cluster.onpcltq.mongodb.net/livestock_monitoring?appName=gomata-cluster";
mongoose.connect(uri).then(async () => {
    const users = await User.find().select('email role');
    console.log("Users:", users);
    process.exit(0);
});
