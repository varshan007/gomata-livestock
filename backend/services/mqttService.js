const logger = require('../utils/logger');
const mqtt = require('mqtt');
const SensorData = require('../models/SensorData');
const Livestock = require('../models/Livestock');
const Alert = require('../models/Alert');

const dns = require('dns');
const url = require('url');

// MQTT Broker Configuration
const MQTT_BROKER_URL_STRING = process.env.MQTT_BROKER_URL || 'mqtt://broker.hivemq.com'; // Fallback to HiveMQ
// Parse the URL to get the hostname
const parsedUrl = url.parse(MQTT_BROKER_URL_STRING);
const MQTT_HOSTNAME = parsedUrl.hostname;
const MQTT_PORT = process.env.MQTT_PORT || parsedUrl.port || 1883;

const MQTT_TOPIC = 'livestock/sensor/#'; // Subscribe to all livestock sensors

let mqttClient = null;

// Initialize MQTT Connection
function initializeMQTT() {
    logger.info(`🔍 Resolving MQTT broker: ${MQTT_HOSTNAME}`);

    // Resolve hostname to IPv4 address manually to avoid IPv6 issues
    dns.lookup(MQTT_HOSTNAME, { family: 4 }, (err, address, family) => {
        if (err) {
            logger.error('❌ DNS Lookup error for MQTT broker:', err);
            // Retry after 5 seconds
            setTimeout(initializeMQTT, 5000);
            return;
        }

        logger.info(`✅ MQTT Broker resolved to: ${address}`);

        const options = {
            host: address, // Use resolved IP
            protocol: 'mqtt', // Force protocol
            port: MQTT_PORT,
            clientId: `livestock_server_${Math.random().toString(16).substr(2, 8)}`,
            clean: true,
            reconnectPeriod: 1000
        };

        mqttClient = mqtt.connect(options);

        setupMqttEventHandlers();
    });
}

function setupMqttEventHandlers() {
    mqttClient.on('connect', () => {
        logger.info('✅ MQTT Connected to broker');

        // Subscribe to sensor data topic
        mqttClient.subscribe(MQTT_TOPIC, (err) => {
            if (!err) {
                logger.info(`📡 Subscribed to topic: ${MQTT_TOPIC}`);
            } else {
                logger.error('❌ MQTT Subscription error:', err);
            }
        });
    });

    mqttClient.on('message', async (topic, message) => {
        try {
            // Parse incoming message
            const data = JSON.parse(message.toString());
            logger.info(`📩 Received data from ${topic}:`, data);

            // Process sensor data
            await processSensorData(data, topic);
        } catch (error) {
            logger.error('❌ Error processing MQTT message:', error);
        }
    });

    mqttClient.on('error', (error) => {
        logger.error('❌ MQTT Error:', error);
    });

    mqttClient.on('close', () => {
        logger.info('🔌 MQTT Connection closed');
    });

    mqttClient.on('reconnect', () => {
        logger.info('🔄 MQTT Reconnecting...');
    });
}

// Process incoming sensor data
async function processSensorData(data, topic) {
    try {
        const { deviceId, temperature, latitude, longitude, batteryLevel } = data;

        // Find livestock by device ID
        const livestock = await Livestock.findOne({ deviceId });

        if (!livestock) {
            logger.warn(`⚠️ No livestock found for device: ${deviceId}`);
            return;
        }

        // Save sensor data to database
        const sensorData = new SensorData({
            livestockId: livestock._id,
            deviceId,
            temperature,
            latitude,
            longitude,
            batteryLevel,
            timestamp: new Date()
        });

        await sensorData.save();
        logger.info(`💾 Sensor data saved for ${livestock.name}`);

        // Check for temperature alerts
        await checkTemperatureAlert(sensorData, livestock);

        // Check for battery alerts
        await checkBatteryAlert(sensorData, livestock);

        // Check for geofence alerts (optional)
        // await checkGeofenceAlert(sensorData, livestock);

    } catch (error) {
        logger.error('❌ Error processing sensor data:', error);
    }
}

// Check temperature and create alerts
async function checkTemperatureAlert(sensorData, livestock) {
    const temp = sensorData.temperature;
    let severity = null;
    let message = null;

    // Cattle normal temp: 38-39°C
    if (temp > 40) {
        severity = 'Critical';
        message = `🚨 CRITICAL: ${livestock.name} temperature ${temp}°C - Immediate veterinary attention needed!`;
    } else if (temp >= 39.5) {
        severity = 'High';
        message = `⚠️ WARNING: ${livestock.name} temperature ${temp}°C - Monitor for fever`;
    } else if (temp < 37.5) {
        severity = 'Medium';
        message = `❄️ ALERT: ${livestock.name} low temperature ${temp}°C - Check for hypothermia`;
    }

    if (severity) {
        // Check if similar unresolved alert exists (avoid spam)
        const existingAlert = await Alert.findOne({
            livestockId: sensorData.livestockId,
            alertType: 'Temperature',
            resolved: false,
            timestamp: { $gte: new Date(Date.now() - 30 * 60 * 1000) } // Last 30 mins
        });

        if (!existingAlert) {
            const alert = new Alert({
                livestockId: sensorData.livestockId,
                alertType: 'Temperature',
                severity: severity,
                message: message
            });
            await alert.save();
            logger.info(`🔔 Alert created: ${message}`);
        }
    }
}

// Check battery level
async function checkBatteryAlert(sensorData, livestock) {
    if (sensorData.batteryLevel && sensorData.batteryLevel < 20) {
        const existingAlert = await Alert.findOne({
            livestockId: sensorData.livestockId,
            alertType: 'Battery',
            resolved: false
        });

        if (!existingAlert) {
            const alert = new Alert({
                livestockId: sensorData.livestockId,
                alertType: 'Battery',
                severity: 'Medium',
                message: `🔋 Low battery: ${livestock.name} device at ${sensorData.batteryLevel}% - Needs charging`
            });
            await alert.save();
            logger.info(`🔔 Battery alert created for ${livestock.name}`);
        }
    }
}

// Publish message to MQTT (optional - for commands to ESP32)
function publishMessage(topic, message) {
    if (mqttClient && mqttClient.connected) {
        mqttClient.publish(topic, JSON.stringify(message), { qos: 1 });
        logger.info(`📤 Published to ${topic}:`, message);
    } else {
        logger.error('❌ MQTT client not connected');
    }
}

// Close MQTT connection
function closeMQTT() {
    if (mqttClient) {
        mqttClient.end();
        logger.info('🔌 MQTT Connection closed');
    }
}

module.exports = {
    initializeMQTT,
    publishMessage,
    closeMQTT
};