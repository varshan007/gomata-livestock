const mongoose = require('mongoose');
const AlertAgent = require('./src/agents/alerts/AlertAgent');
require('dotenv').config();

const mockBus = {
    publisher: {
        set: async () => 'OK' // Mock redis set to bypass deduplication
    },
    on: (event, handler) => {
        if (event === 'alert:create') {
            global.alertHandler = handler;
        }
    },
    emit: (event, data) => console.log(`[MOCK BUS EMIT] ${event} ->`, JSON.stringify(data))
};

mongoose.connect(process.env.MONGODB_URI || "mongodb://127.0.0.1:27017/livestock-monitoring")
    .then(async () => {
        console.log('Connected to DB. Testing AlertAgent processing...');
        const agent = new AlertAgent(mockBus);
        agent.start();

        const payload = {
            data: {
                hwId: 'COW001',
                type: 'Health Emergency',
                severity: 'CRITICAL',
                message: 'Test message',
                action: 'Test action',
                metrics: {}
            },
            traceId: 'trace-123'
        };

        await global.alertHandler(payload);
        console.log('Test completed. Waiting for async DB save...');
    }).catch(e => {
        console.error(e);
        process.exit(1);
    });
