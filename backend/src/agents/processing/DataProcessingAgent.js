const logger = require('../../../utils/logger');
const { isMainThread, workerData, parentPort } = require('worker_threads');

class DataProcessingAgent {
    constructor(bus) {
        this.bus = bus;
    }

    start() {
        logger.info('[DataProcessingAgent] Started. Listening for raw telemetry...');

        this.bus.on('telemetry:received', (payload) => {
            const rawData = payload.data;
            const normalizedData = this.normalize(rawData);

            if (normalizedData) {
                // Pass traceId along the chain
                this.bus.emit('telemetry:normalized', normalizedData, payload.traceId);
            }
        });
    }

    normalize(data) {
        // 1. Drop wildly invalid outliers that might be sensor errors (e.g. 0°C or 100°C cow)
        if (data.temp < 30 || data.temp > 45) {
            logger.info(`[DataProcessingAgent] Dropped invalid temp reading from ${data.hwId}: ${data.temp}`);
            return null;
        }

        // 2. Impute missing values (e.g., if heartRate failed this tick, use last known or default)
        const normalized = {
            ...data,
            temp: parseFloat(parseFloat(data.temp).toFixed(1)), // Ensure float
            heartRate: data.heartRate || 80,
            battery: data.battery || 100,
            processedAt: new Date().toISOString()
        };

        return normalized;
    }
}

module.exports = DataProcessingAgent;
