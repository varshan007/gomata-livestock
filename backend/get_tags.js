const mongoose = require('mongoose');
const Livestock = require('./models/Livestock');
require('dotenv').config();

mongoose.connect(process.env.MONGODB_URI || "mongodb://127.0.0.1:27017/livestock-monitoring")
  .then(async () => {
    const animals = await Livestock.find({}).limit(5);
    if (animals.length === 0) console.log("NO ANIMALS FOUND");
    animals.forEach(a => console.log(`Tag: ${a.tagNumber}, ID: ${a._id}`));
    process.exit(0);
  }).catch(e => { console.error(e); process.exit(1); });
