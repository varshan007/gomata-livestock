const mongoose = require('mongoose');
require('dotenv').config({ path: '/Users/googledoodle/Downloads/livestock_monitoring/backend/.env' });
const Farm = require('./models/Farm');
const LivestockMaster = require('./models/LivestockMaster');

mongoose.connect(process.env.MONGO_URI, { useNewUrlParser: true, useUnifiedTopology: true })
    .then(async () => {
        const livestock = await LivestockMaster.findOne({});
        console.log(`Livestock: ${livestock.name}, FarmId: ${livestock.farm_id}, FarmName: "${livestock.farm_name}", User: ${livestock.userId}`);

        const farmByName = await Farm.findOne({ userId: livestock.userId, name: livestock.farm_name });
        console.log(`Farm by Name: ${farmByName ? farmByName.name : 'Not Found'} with Zone Count: ${farmByName ? await Zone.countDocuments({ farmId: farmByName._id }) : 0}`);

        process.exit(0);
    })
    .catch(console.error);
