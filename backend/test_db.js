const mongoose = require('mongoose');
const LivestockMaster = require('./models/LivestockMaster');
const uri = "mongodb+srv://gomata_admin:IITB01%40GOOgle@gomata-cluster.onpcltq.mongodb.net/livestock_monitoring?appName=gomata-cluster";
mongoose.connect(uri).then(async () => {
    const count = await LivestockMaster.countDocuments();
    console.log("LivestockMaster count:", count);
    if(count > 0) {
       const docs = await LivestockMaster.find().limit(1);
       console.log("Doc userId:", docs[0].userId, typeof docs[0].userId);
    }
    process.exit(0);
});
