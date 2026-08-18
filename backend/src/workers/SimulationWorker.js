const Livestock = require('../../models/Livestock');
const SensorData = require('../../models/SensorData');
const logger = require('../../utils/logger');

class SimulationWorker {
    constructor(bus) {
        this.bus = bus;
        this.interval = parseInt(process.env.SIMULATION_TICK_MS || '60000', 10);
        this.timer = null;
    }

    start() {
        logger.info(`[SimulationWorker] Starting background simulator (tick: ${this.interval}ms). Bypassing MQTT TCP.`);
        this.timer = setInterval(() => this.tick(), this.interval);
    }

    stop() {
        if (this.timer) {
            clearInterval(this.timer);
            this.timer = null;
        }
    }

    async tick() {
        try {
            // OPTION B: Only actively simulate real-time telemetry for 3 "hero" animals to save Redis commands.
            // This prevents exceeding the 10,000 commands/day limit on Upstash free tier.
            const cattle = await Livestock.find({ 
                status: 'Active', 
                name: { $in: ['Bessie', 'Cow 1', 'Cow 2'] } 
            });
            
            if (!cattle.length) return;

            const now = new Date();
            const hourOfDay = now.getHours();

            for (const ls of cattle) {
                // Generate plausible data
                let temp = 38.0 + Math.sin(hourOfDay * Math.PI / 12) * 0.5 + (Math.random() * 0.2 - 0.1);
                let hr = 60 + Math.sin(hourOfDay * Math.PI / 12) * 10 + (Math.random() * 5 - 2.5);

                // If it's Bessie, keep her temp elevated to show the AI alert scenario is ongoing
                if (ls.name === 'Bessie') {
                    temp = 40.2 + (Math.random() * 0.4 - 0.2);
                    hr = 85 + (Math.random() * 5 - 2.5);
                }

                const payload = {
                    deviceId: ls.deviceId,
                    livestockId: ls._id.toString(),
                    timestamp: now.toISOString(),
                    temperature: parseFloat(temp.toFixed(2)),
                    heartRate: parseFloat(hr.toFixed(2)),
                    activityLevel: 0.5,
                    location: { latitude: 19.07 + Math.random()*0.005, longitude: 72.87 + Math.random()*0.005 },
                    batteryLevel: 85
                };

                // Push directly to internal event bus instead of MQTT
                this.bus.emit('telemetry:raw', {
                    topic: `gomata/${ls.deviceId}/telemetry`,
                    payload: Buffer.from(JSON.stringify(payload))
                });
            }
            logger.info(`[SimulationWorker] Pushed telemetry for ${cattle.length} animals directly to Event Bus.`);
        } catch (error) {
            logger.error('[SimulationWorker] Error during tick:', error);
        }
    }
}

module.exports = SimulationWorker;
