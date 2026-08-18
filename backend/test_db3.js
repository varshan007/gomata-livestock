const mongoose = require('mongoose');
const LivestockMaster = require('./models/LivestockMaster');
const uri = "mongodb+srv://gomata_admin:IITB01%40GOOgle@gomata-cluster.onpcltq.mongodb.net/livestock_monitoring?appName=gomata-cluster";
mongoose.connect(uri).then(async () => {
    let baseQuery = { userId: new mongoose.Types.ObjectId('6a840f80bf8bde1bb70b956b') };
    const livestock = await LivestockMaster.find(baseQuery)
      .select('livestock_id name breed temperature heart_rate battery signal_strength zone_name last_updated health_status farm_id zone_id last_location')
      .sort({ last_updated: -1 });
    console.log("Docs found:", livestock.length);
    process.exit(0);
});
