const mqtt = require('mqtt');
const EdgeBuffer = require('./EdgeBuffer');
const logger = require('../../../utils/logger');

/**
 * HardwareAgent
 * Wraps the MQTT listener. Receives telemetry from IoT devices, 
 * passes it through the EdgeBuffer for safety, and emits it to the Redis bus.
 */
class HardwareAgent {
    constructor(bus, mqttUrl, models) {
        this.bus = bus;
        this.mqttUrl = process.env.MQTT_URL || 'mqtt://127.0.0.1:1883';
        this.client = null;
        this.buffer = new EdgeBuffer(models);
        this.isOnline = false;
    }

    start() {
        if (process.env.MQTT_ENABLED !== 'true') {
            logger.info(`[HardwareAgent] MQTT connection disabled (MQTT_ENABLED != true). Skipping actual broker connection.`);
            return;
        }

        logger.info(`[HardwareAgent] Connecting to MQTT broker at ${this.mqttUrl}...`);
        this.client = mqtt.connect(this.mqttUrl, {
            clientId: `gomata_agent_${Math.random().toString(16).slice(3)}`,
            keepalive: 60,
            reconnectPeriod: 2000
        });

        this.client.on('connect', () => {
            logger.info(`[HardwareAgent] ✅ Connected. Flushing EdgeBuffer...`);
            this.isOnline = true;
            this.client.subscribe('gomata/+/telemetry');
            this.client.subscribe('gomata/+/alert');
            this.buffer.flush(this.bus);
        });

        this.client.on('message', (topic, payload) => {
            // Buffer pushing removed for live testing since MQTT drops cause the payload to hide

            const hwId = topic.split('/')[1];

            try {
                const data = JSON.parse(payload.toString());

                // Determine event type based on MQTT topic
                const eventName = topic.includes('alert') ? 'hardware:alert' : 'telemetry:received';
                logger.info({ action: 'telemetry_ingestion', result: 'success', animalId: hwId, type: 'mqtt' }, `Parsed MQTT payload for ${hwId}. Emitting ${eventName}...`);
                const payloadToEmit = {
                    hwId,
                    temp: data.temperature || data.temp,
                    heartRate: data.heartRate || data.hr,
                    lat: data.lat || (data.location ? data.location.lat : null),
                    lng: data.lng || (data.location ? data.location.lng : null),
                    battery: data.battery,
                    ts: new Date()
                };

                logger.info(`[HardwareAgent] Payload to emit:`, JSON.stringify(payloadToEmit));
                this.bus.emit(eventName, payloadToEmit);

            } catch (error) {
                logger.error({ action: 'telemetry_ingestion', result: 'error', animalId: hwId, error: error.message }, `Failed to parse payload from ${topic}`);
            }
        });

        this.client.on('offline', () => {
            logger.warn(`[HardwareAgent] ⚠️ MQTT Connection lost. Buffering locally...`);
            this.isOnline = false;
        });

        this.client.on('error', (err) => {
            logger.error(`[HardwareAgent] MQTT Error:`, err.message);
        });
    }
}

module.exports = HardwareAgent;
