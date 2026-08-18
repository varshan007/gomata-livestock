const mongoose = require('mongoose');
const LivestockMaster = require('./backend/models/LivestockMaster');
require('dotenv').config({ path: './backend/.env' });
const uri = "mongodb+srv://gomata_admin:IITB01%40GOOgle@gomata-cluster.onpcltq.mongodb.net/livestock_monitoring?appName=gomata-cluster";
mongoose.connect(uri).then(async () => {
    const count = await LivestockMaster.countDocuments();
    console.log("LivestockMaster count:", count);
    if(count > 0) {
       const docs = await LivestockMaster.find().limit(1);
       console.log(docs);
    }
    process.exit(0);
});
