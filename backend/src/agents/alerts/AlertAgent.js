const logger = require('../../../utils/logger');
const Alert = require('../../../models/Alert');
const Livestock = require('../../../models/Livestock');

class AlertAgent {
    constructor(bus) {
        this.bus = bus;
        // Redis connection to use for atomic deduplication locks
        // Re-use the publisher connection from the bus for simple operations
        this.redis = bus.publisher;

        // 10 minutes deduplication window to prevent spamming the farmer
        this.DEDUP_SECONDS = 600;
    }

    start() {
        logger.info('[AlertAgent] Started. Listening for new AI health alerts...');

        this.bus.on('alert:create', async (payload) => {
            logger.info(`\n[AlertAgent] 📥 Received alert:create payload:`, JSON.stringify(payload).substring(0, 150) + '...');
            const alertData = payload.data;
            const traceId = payload.traceId;
            const { hwId, type, severity, message, action, metrics } = alertData;

            // Generate a unique cache key for this specific animal and alert type
            // e.g., "alert_lock:COW001:CRITICAL"
            const dedupKey = `alert_lock:${hwId}:${severity}`;

            try {
                // Try to set the key in Redis if it DOES NOT EXIST (NX).
                // If it succeeds, it returns 1 (meaning it's a new alert).
                // If the key already exists, it returns 0 (meaning we just alerted this recently).
                const isNewAlert = await this.redis.set(dedupKey, 'locked', 'EX', this.DEDUP_SECONDS, 'NX');

                if (!isNewAlert) {
                    logger.info(`[AlertAgent] 🛡️ Deduplicated alert for ${hwId} (${severity}). A similar alert was already sent in the last 10 minutes.`);
                    return; // Stop processing, do not write to DB, do not send SMS.
                }

                logger.info(`[AlertAgent] ⚠️ Writing new ${severity} Alert to DB for ${hwId}: ${message}`);

                const severityMap = {
                    'WARNING': 'High',
                    'CRITICAL': 'Critical',
                    'HEALTHY': 'Low'
                };
                const mappedSeverity = severityMap[severity] || 'Medium';

                const queryConditions = [{ deviceId: hwId }, { tagNumber: hwId }];
                if (hwId.length === 24) {
                    queryConditions.push({ _id: hwId });
                }
                const animal = await Livestock.findOne({ $or: queryConditions });
                const livestockObjectId = animal ? animal._id : null;

                if (!livestockObjectId) {
                    logger.info(`[AlertAgent] ❌ Could not find Livestock ObjectId for ${hwId}. Skipping DB save.`);
                    return;
                }

                // Write the Alert to MongoDB
                const newAlert = await Alert.create({
                    livestockId: livestockObjectId,
                    alertType: 'Temperature', // Map AI health events to strictly 'Temperature' enum
                    severity: mappedSeverity,
                    message: message,
                    resolved: false
                });

                // Emit event that a brand new, valid, deduplicated alert is ready in the DB
                this.bus.emit('alert:saved', {
                    alertId: newAlert._id,
                    livestockId: hwId,
                    severity: severity,
                    message: message
                }, traceId);

                // Also trigger the SyncAgent to push this to the frontend React UI navbar!
                this.bus.emit('db:write', { type: 'alert', data: newAlert }, traceId);

            } catch (error) {
                logger.error(`[AlertAgent] Failed to process alert for ${hwId}:`, error);
            }
        });
    }
}

module.exports = AlertAgent;
