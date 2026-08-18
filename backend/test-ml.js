require('dotenv').config({ path: '.env.development' });
console.log("URL:", process.env.ML_SERVICE_URL);
const mlServiceClient = require('./services/mlServiceClient');
mlServiceClient.getHealth().then(console.log).catch(console.error);
