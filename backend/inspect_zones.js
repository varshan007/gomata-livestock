const mongoose = require('mongoose');
require('dotenv').config({ path: '/Users/googledoodle/Downloads/livestock_monitoring/backend/.env' });
const Farm = require('./models/Farm');
const Zone = require('./models/Zone');

mongoose.connect(process.env.MONGO_URI, { useNewUrlParser: true, useUnifiedTopology: true })
    .then(async () => {
        const farms = await Farm.find({});
        console.log(`Farms count: ${farms.length}`);

        const zones = await Zone.find({});
        console.log(`Zones count: ${zones.length}`);
        zones.forEach(z => {
            console.log(`Zone: ${z.name}, FarmId: ${z.farmId}, Geofence: ${z.geofence?.type}`);
        });

        process.exit(0);
    })
    .catch(console.error);
