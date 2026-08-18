const logger = require('../utils/logger');
const LivestockMaster = require('../models/LivestockMaster');

class DataManagementService {
    /**
     * Ingests a raw DeviceTelemetry record and immediately aggregates its
     * findings upward into the materialized LivestockMaster view.
     * @param {Object} telemetryRecord - from DeviceTelemetry
     */
    static async processTelemetryUpdate(telemetryRecord) {
        try {
            if (!telemetryRecord || !telemetryRecord.deviceId) return;

            // Simple medical diagnosis block (demo purposes)
            let health = 'Normal';
            if (telemetryRecord.temperature > 39.0 || telemetryRecord.temperature < 37.8) health = 'Attention Required';
            if (telemetryRecord.deviceStatus === 'Offline') health = 'Unknown';
            if (telemetryRecord.deviceStatus === 'Low Battery') health = 'Maintenance Needed';

            // Find the active LivestockMaster mapped to this device and update its materialized fields
            await LivestockMaster.updateOne(
                { device_id: telemetryRecord.deviceId },
                {
                    $set: {
                        temperature: telemetryRecord.temperature,
                        heart_rate: telemetryRecord.heartRate,
                        battery: telemetryRecord.battery,
                        signal_strength: telemetryRecord.signalStrength,
                        device_status: telemetryRecord.deviceStatus,
                        last_location: telemetryRecord.location,
                        health_status: health,
                        last_updated: telemetryRecord.timestamp
                    }
                }
            );

        } catch (error) {
            logger.error('[DataManagementService] Failed to aggregate telemetry into LivestockMaster: ', error);
        }
    }
}

module.exports = DataManagementService;
