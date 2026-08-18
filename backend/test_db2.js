const mongoose = require('mongoose');
const User = require('./models/User');
const uri = "mongodb+srv://gomata_admin:IITB01%40GOOgle@gomata-cluster.onpcltq.mongodb.net/livestock_monitoring?appName=gomata-cluster";
mongoose.connect(uri).then(async () => {
    const admin = await User.findOne({email: 'admin@gomata.com'});
    console.log("Admin ID:", admin._id, typeof admin._id);
    process.exit(0);
});
