const mongoose = require('mongoose');
const HealthAgent = require('./src/agents/health/HealthAgent');
require('dotenv').config();

const mockBus = {
    emit: (event, data) => console.log(`[MOCK BUS EMIT] ${event} ->`, JSON.stringify(data))
};

mongoose.connect(process.env.MONGODB_URI || "mongodb://127.0.0.1:27017/livestock-monitoring")
    .then(async () => {
        console.log('Connected to DB. Testing AI Diagnosis...');
        const agent = new HealthAgent(mockBus, process.env.GOOGLE_API_KEY);

        const hwId = 'COW001';
        const history = [{
            hwId: hwId, temp: 41.5, heartRate: 110, battery: 88
        }];

        await agent.diagnoseWithAI(hwId, history, 'trace-123');
        console.log('Test completed.');
        process.exit(0);
    }).catch(e => {
        console.error(e);
        process.exit(1);
    });
