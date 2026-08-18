const logger = require('../utils/logger');
const DeviceTelemetry = require('../models/DeviceTelemetry');
const LivestockMaster = require('../models/LivestockMaster');
const TrainingEvent = require('../models/TrainingEvent');
const DataManagementService = require('./DataManagementService');
const { queues } = require('../config/bullmq');
const stats = require('../utils/stats');

// Base simulation constraints
const TARGET_LOCATION = [77.161358, 28.634064]; // [lng, lat]
const METERS_PER_DEGREE = 111139;

// Configurable ML Simulation Vectors
const SIMULATION_MODE = process.env.SIMULATION_MODE || "production";
const SIMULATION_TICK_MS = process.env.SIMULATION_TICK_MS ? parseInt(process.env.SIMULATION_TICK_MS) : 10000;
const ANOMALY_INJECTION_RATE = process.env.ANOMALY_INJECTION_RATE ? parseFloat(process.env.ANOMALY_INJECTION_RATE) : 0.15;

// Uniform feature window — same for anomaly + normal to prevent label leakage
const FEATURE_WINDOW = 12;

const ANOMALY_PROBABILITIES = { fever: 0.15, tachycardia: 0.35, stillness: 0.50 };
const INTENSITY_PROBABILITIES = { mild: 0.6, moderate: 0.3, severe: 0.1 };

// Helper to pick randomly from a weighted probability object
function weightedRandom(probabilities) {
    let sum = 0;
    let r = Math.random();
    for (const [key, weight] of Object.entries(probabilities)) {
        sum += weight;
        if (r <= sum) return key;
    }
    return Object.keys(probabilities)[0];
}

function getRandomInt(min, max) {
    return Math.floor(Math.random() * (max - min + 1)) + min;
}

class HardwareSimulationService {
    constructor() {
        this.intervalId = null;
        this.devicesToSimulate = [
            'GM-SN-1001', 'GM-SN-1002', 'GM-SN-1003', 'GM-SN-1004', 'GM-SN-1005',
            'GM-SN-1006', 'GM-SN-1007', 'GM-SN-1008', 'GM-SN-1009', 'GM-SN-1010'
        ];

        // Internal state tracker to enable smooth drifting over time
        this.activeStates = {};

        // Track ongoing anomaly episodes per device
        this.activeEpisodes = {};

        // High-Performance Batch Buffers
        this.telemetryBuffer = [];
        this.trainingEventBuffer = [];
        this.flushIntervalId = null;

        // Cluster Info
        this.workerIndex = parseInt(process.env.WORKER_INDEX || "0");
        this.totalWorkers = parseInt(process.env.TOTAL_WORKERS || "1");
    }

    start() {
        if (this.intervalId) return;

        logger.info({
            action: 'SIMULATION_WORKER_STARTED',
            workerIndex: this.workerIndex,
            totalWorkers: this.totalWorkers,
            service: 'simulation_engine',
            simulationMode: SIMULATION_MODE,
            tickIntervalMs: SIMULATION_TICK_MS,
            anomalyRate: ANOMALY_INJECTION_RATE
        }, `📡 [Hardware Simulation Service Worker ${this.workerIndex}/${this.totalWorkers}] Engine Started.`);

        // Initialize Base States
        this.devicesToSimulate.forEach(id => {
            // Generate a random distance between 200 and 800 meters
            const randomDistanceMeters = Math.floor(Math.random() * (800 - 200 + 1)) + 200;
            // Generate a random angle between 0 and 360 degrees (in radians)
            const randomAngleRads = Math.random() * 2 * Math.PI;

            // Calculate x (lng) and y (lat) offsets in meters
            const xOffsetMeters = randomDistanceMeters * Math.cos(randomAngleRads);
            const yOffsetMeters = randomDistanceMeters * Math.sin(randomAngleRads);

            this.activeStates[id] = {
                temperature: 38.0,
                heartRate: 60,
                battery: Math.floor(Math.random() * (100 - 40 + 1)) + 40,
                signalStrength: -60,
                lngOffset: xOffsetMeters / METERS_PER_DEGREE,
                latOffset: yOffsetMeters / METERS_PER_DEGREE
            };
        });

        // Loop
        this.intervalId = setInterval(() => this.simulateTelemetryTick(), SIMULATION_TICK_MS);

        // Batch flush interval
        this.flushIntervalId = setInterval(async () => {
            const telemetryCount = this.telemetryBuffer.length;
            const eventCount = this.trainingEventBuffer.length;

            if (telemetryCount === 0 && eventCount === 0) return;

            const startMs = Date.now();

            if (telemetryCount > 0) {
                const batch = this.telemetryBuffer.splice(0, telemetryCount);
                try {
                    await DeviceTelemetry.insertMany(batch, { ordered: false, lean: true });
                } catch (e) {
                    logger.error({ action: "BATCH_WRITE_ERROR", error: e.message }, "Telemetry batch write failed");
                }
            }

            if (eventCount > 0) {
                const batch = this.trainingEventBuffer.splice(0, eventCount);
                try {
                    await TrainingEvent.insertMany(batch, { ordered: false, lean: true });
                } catch (e) {
                    logger.error({ action: "BATCH_WRITE_ERROR", error: e.message }, "TrainingEvent batch write failed");
                }
            }

            const durationMs = Date.now() - startMs;

            logger.info({
                action: "SIMULATION_BATCH_FLUSH",
                telemetryCount,
                eventCount,
                durationMs
            });

            logger.info({
                action: "BATCH_WRITE_COMPLETED",
                telemetryInserted: telemetryCount,
                eventsInserted: eventCount,
                durationMs
            });

        }, 500);
    }

    stop() {
        if (this.intervalId) clearInterval(this.intervalId);
        if (this.flushIntervalId) clearInterval(this.flushIntervalId);
    }

    // Math utils for natural drift
    drift(current, min, max, maxDelta) {
        let delta = (Math.random() * maxDelta * 2) - maxDelta;
        let next = current + delta;
        if (next < min) return min + Math.abs(delta);
        if (next > max) return max - Math.abs(delta);
        return parseFloat(next.toFixed(2));
    }

    async computeRollingFeatures(animalId, currentTelemetry) {
        // Fetch historical points (FEATURE_WINDOW - 1 because we prepend current)
        let historical = await DeviceTelemetry.find({ animalId })
            .sort({ timestamp: -1 })
            .limit(FEATURE_WINDOW - 1)
            .lean();

        // Chronological array ordering
        let telemetryList = [currentTelemetry, ...historical].reverse();

        if (telemetryList.length < 3) return null;

        let temps = [];
        let hrs = [];
        let activities = [];

        for (const t of telemetryList) {
            temps.push(t.temperature || 39.0);
            hrs.push(t.heartRate || 60);
            activities.push(typeof t.activity === 'number' ? t.activity : 75);
        }

        // Base stats
        const tempAvg = stats.avg(temps);
        const tempStd = stats.std(temps);
        const hrAvg = stats.avg(hrs);
        const hrStd = stats.std(hrs);
        const actAvg = stats.avg(activities);

        return {
            // Original 15 features
            temp_current: Number(temps[temps.length - 1]),
            temp_6h_avg: Number(tempAvg),
            temp_6h_std: Number(tempStd),
            temp_6h_slope: Number(stats.slope(temps)),
            temp_max_6h: Number(Math.max(...temps)),
            temp_min_6h: Number(Math.min(...temps)),
            temp_range_6h: Number(Math.max(...temps) - Math.min(...temps)),

            hr_current: Number(hrs[hrs.length - 1]),
            hr_6h_avg: Number(hrAvg),
            hr_6h_std: Number(hrStd),
            hr_6h_slope: Number(stats.slope(hrs)),

            activity_current: Number(activities[activities.length - 1]),
            activity_6h_avg: Number(stats.avg(activities)),
            activity_6h_std: Number(stats.std(activities)),
            activity_6h_slope: Number(stats.slope(activities)),

            // Ratio features — current vs baseline
            temp_ratio: Number(temps[temps.length - 1] / (tempAvg || 1)),
            hr_ratio: Number(hrs[hrs.length - 1] / (hrAvg || 1)),
            activity_ratio: Number(activities[activities.length - 1] / (actAvg || 0.01)),

            // Z-score features
            temp_zscore: Number(tempStd > 0 ? (temps[temps.length - 1] - tempAvg) / tempStd : 0),
            hr_zscore: Number(hrStd > 0 ? (hrs[hrs.length - 1] - hrAvg) / hrStd : 0),

            // Recent vs baseline (last 6 vs full window)
            temp_recent_vs_baseline: Number(stats.avg(temps.slice(-6)) - tempAvg),
            hr_recent_vs_baseline: Number(stats.avg(hrs.slice(-6)) - hrAvg),
            activity_recent_vs_baseline: Number(stats.avg(activities.slice(-6)) - actAvg)
        };
    }

    async startAnomalyEpisode(map) {
        const id = map.device_id;

        // 2️⃣ Weighted Probability Engine with mid-episode conditions
        let modifiedProbs = { ...ANOMALY_PROBABILITIES };
        if (this.activeStates[id].temperature > 39.0) {
            modifiedProbs.fever = 0.40;
            modifiedProbs.tachycardia = 0.40;
            modifiedProbs.stillness = 0.20;
        }

        const eventType = weightedRandom(modifiedProbs);
        const intensity = weightedRandom(INTENSITY_PROBABILITIES);

        // 3️⃣ Variable Duration (2-10 ticks)
        let minTicks = intensity === 'severe' ? 6 : (intensity === 'moderate' ? 4 : 2);
        const totalTicks = getRandomInt(minTicks, 10);

        // 4️⃣ Gradual Onset and Recovery
        const onsetTicks = Math.max(1, Math.floor(totalTicks * 0.25));
        const decayTicks = Math.max(1, Math.floor(totalTicks * 0.35));
        const peakTicks = totalTicks - onsetTicks - decayTicks;

        let targetAmplitude = {};
        let correlatedSignals = [];
        let correlationStrength = 0;

        // Base correlation probability based on intensity to hit target metric distributions
        // Targets: mild ~10%, moderate ~30%, severe ~60%
        const baseProb = intensity === 'mild' ? 0.25 : (intensity === 'moderate' ? 0.40 : 0.75);

        // 7️⃣ Borderline Zones (Overlapping) & Correlation Logic
        if (eventType === 'fever') {
            if (intensity === 'mild') targetAmplitude.temperature = this.drift(39.0, 39.0, 39.8, 0.4);
            else if (intensity === 'moderate') targetAmplitude.temperature = this.drift(39.5, 39.5, 40.8, 0.5);
            else {
                targetAmplitude.temperature = this.drift(40.5, 40.5, 42.0, 0.6);
            }
            // Severe fever ALWAYS causes HR elevation, otherwise use base probability
            let feverProb = intensity === 'severe' ? 1.0 : baseProb;
            if (Math.random() < feverProb) {
                correlatedSignals.push('heartRate');
                correlationStrength = intensity === 'mild' ? 0.3 : (intensity === 'moderate' ? 0.6 : 0.9);
                const hrBaseIncrease = correlationStrength * 40;
                targetAmplitude.heartRate = this.drift(80 + hrBaseIncrease, 80 + hrBaseIncrease - 5, 80 + hrBaseIncrease + 15, 5);
            }
        } else if (eventType === 'tachycardia') {
            if (intensity === 'mild') targetAmplitude.heartRate = getRandomInt(80, 105);
            else if (intensity === 'moderate') targetAmplitude.heartRate = getRandomInt(100, 125);
            else {
                targetAmplitude.heartRate = getRandomInt(120, 150);
            }
            // Only moderate or severe may trigger temperature slight elevation
            if ((intensity === 'moderate' || intensity === 'severe') && Math.random() < baseProb) {
                correlatedSignals.push('temperature');
                correlationStrength = intensity === 'moderate' ? 0.4 : 0.7;
                targetAmplitude.temperature = this.drift(39.0, 39.0, 39.8, 0.2);
            }
        } else if (eventType === 'stillness') {
            targetAmplitude.activity = 'still';
            // Probability lower than fever correlation
            let stillProb = baseProb * 0.5;
            if (Math.random() < stillProb) {
                correlatedSignals.push('heartRate', 'temperature');
                correlationStrength = intensity === 'mild' ? 0.2 : (intensity === 'moderate' ? 0.5 : 0.8);
                targetAmplitude.heartRate = getRandomInt(50, 65);
                targetAmplitude.temperature = this.drift(37.5, 37.0, 38.0, 0.2);
            }
        }

        // 1️⃣ Episode State Machine
        this.activeEpisodes[id] = {
            type: eventType,
            intensity,
            phase: 'onset', // onset, peak, decay
            totalTicks,
            currentTick: 0,
            onsetTicks,
            peakTicks,
            decayTicks,
            baselineSnapshot: { ...this.activeStates[id] },
            targetAmplitude,
            correlatedSignals,
            correlationStrength,
            peakSaved: false,   // TrainingEvent will be saved at peak, not here
            mapRef: map         // keep reference for peak capture
        };

        logger.info({
            action: "ANOMALY_INJECTED",
            service: "simulation_engine",
            simulationMode: SIMULATION_MODE,
            tenantId: map.userId,
            animalId: map._id,
            eventType: eventType
        }, `Anomaly [${eventType}] injected for ${id} (Intensity: ${intensity}, Ticks: ${totalTicks})`);
    }

    async maybeInjectAnomaly(map) {
        const id = map.device_id;

        // If an episode is already running, progress state machine
        if (this.activeEpisodes[id]) {
            const ep = this.activeEpisodes[id];
            ep.currentTick++;

            const prevPhase = ep.phase;
            if (ep.currentTick <= ep.onsetTicks) {
                ep.phase = 'onset';
            } else if (ep.currentTick <= ep.onsetTicks + ep.peakTicks) {
                ep.phase = 'peak';
            } else if (ep.currentTick <= ep.totalTicks) {
                ep.phase = 'decay';
            } else {
                // Recovery complete, remove episode
                delete this.activeEpisodes[id];
                return false;
            }

            // ── Capture TrainingEvent at PEAK phase (vitals at max deviation) ──
            if (ep.phase === 'peak' && !ep.peakSaved) {
                ep.peakSaved = true;

                const syntheticCurrent = {
                    temperature: this.activeStates[id].temperature,
                    heartRate: this.activeStates[id].heartRate,
                    activity: this.activeStates[id].activity || 75
                };
                const computedFeatures = await this.computeRollingFeatures(map._id, syntheticCurrent);

                if (computedFeatures) {
                    const trainingEvent = new TrainingEvent({
                        tenantId: map.userId,
                        animalId: map._id,
                        eventType: ep.type,
                        label: 1,
                        source: 'simulation_v3',
                        createdAt: new Date(),
                        features: computedFeatures,
                        metadata: {
                            intensity: ep.intensity,
                            phase: 'peak',
                            totalTicks: ep.totalTicks,
                            correlatedSignals: ep.correlatedSignals,
                            correlationStrength: ep.correlationStrength,
                            baselineSnapshot: {
                                temperature: ep.baselineSnapshot.temperature,
                                heartRate: ep.baselineSnapshot.heartRate
                            }
                        }
                    });
                    this.trainingEventBuffer.push(trainingEvent);
                }
            }

            return false;
        }

        // Randomly decide if we should start a new anomaly
        if (Math.random() < ANOMALY_INJECTION_RATE) {
            await this.startAnomalyEpisode(map);
            return false;
        } else {
            // 6️⃣ Sensor Noise Injection (5% chance). Do NOT create TrainingEvents. Unlabeled.
            if (Math.random() < 0.05) {
                return true; // isNoise = true
            }
            // Generate a Normal training label — only when animal is NOT in an active episode
            if (Math.random() < 0.08) {
                const syntheticCurrent = {
                    temperature: this.activeStates[id].temperature,
                    heartRate: this.activeStates[id].heartRate,
                    activity: this.activeStates[id].activity || 75
                };
                const computedFeatures = await this.computeRollingFeatures(map._id, syntheticCurrent);

                if (computedFeatures) {
                    const trainingEvent = new TrainingEvent({
                        tenantId: map.userId,
                        animalId: map._id,
                        eventType: 'normal',
                        label: 0,
                        source: 'simulation_v3',
                        createdAt: new Date(),
                        features: computedFeatures
                    });
                    this.trainingEventBuffer.push(trainingEvent);
                }
            }
            return false; // isNoise = false
        }
    }

    async simulateTelemetryTick() {
        logger.info({ action: 'SIMULATION_CYCLE_STARTED', service: 'simulation_engine' }, 'Starting simulation cycle');
        try {
            const animals = await LivestockMaster.find({
                livestock_id: { $exists: true },
                device_id: { $exists: true },
                userId: { $exists: true }
            });

            const activeMappings = animals.filter((animal, index) =>
                index % this.totalWorkers === this.workerIndex
            );

            logger.info({
                action: "SIMULATION_ANIMALS_LOADED",
                workerIndex: this.workerIndex,
                totalWorkers: this.totalWorkers,
                animalCount: activeMappings.length
            });

            if (activeMappings.length === 0) return; // Wait until UI mapping is finished

            for (const map of activeMappings) {
                const id = map.device_id;

                if (!this.activeStates[id]) {
                    const randomDistanceMeters = Math.floor(Math.random() * (800 - 200 + 1)) + 200;
                    const randomAngleRads = Math.random() * 2 * Math.PI;
                    const xOffsetMeters = randomDistanceMeters * Math.cos(randomAngleRads);
                    const yOffsetMeters = randomDistanceMeters * Math.sin(randomAngleRads);

                    this.activeStates[id] = {
                        temperature: 38.0,
                        heartRate: 60,
                        activity: 60 + Math.random() * 30, // continuous 60-90 baseline
                        battery: Math.floor(Math.random() * (100 - 40 + 1)) + 40,
                        signalStrength: -60,
                        lngOffset: xOffsetMeters / METERS_PER_DEGREE,
                        latOffset: yOffsetMeters / METERS_PER_DEGREE
                    };
                }

                const state = this.activeStates[id];

                // Decide Anomaly Generation
                const isNoise = await this.maybeInjectAnomaly(map);

                const activeEpisode = this.activeEpisodes[id];

                // Mutate Vitals based on Anomaly State
                if (activeEpisode) {
                    const ep = activeEpisode;

                    // calculate interpolation factor based on phase
                    let factor = 0;
                    if (ep.phase === 'onset') {
                        factor = ep.currentTick / ep.onsetTicks;
                    } else if (ep.phase === 'peak') {
                        factor = 1.0;
                    } else if (ep.phase === 'decay') {
                        const ticksIntoDecay = ep.currentTick - ep.onsetTicks - ep.peakTicks;
                        factor = 1.0 - (ticksIntoDecay / ep.decayTicks);
                    }

                    if (ep.targetAmplitude.temperature) {
                        const target = ep.baselineSnapshot.temperature + (ep.targetAmplitude.temperature - ep.baselineSnapshot.temperature) * factor;
                        state.temperature = this.drift(target, target - 0.2, target + 0.2, 0.1);
                    } else {
                        state.temperature = this.drift(state.temperature, 38.0, 39.2, 0.2); // Normal
                    }

                    if (ep.targetAmplitude.heartRate) {
                        const target = ep.baselineSnapshot.heartRate + (ep.targetAmplitude.heartRate - ep.baselineSnapshot.heartRate) * factor;
                        state.heartRate = Math.floor(this.drift(target, target - 5, target + 5, 2));
                    } else {
                        state.heartRate = Math.floor(this.drift(state.heartRate, 60, 80, 5)); // Normal
                    }

                    if (ep.targetAmplitude.activity === 'still') {
                        // Stillness: drop activity to 0-30 range based on intensity
                        const intensityFactor = ep.intensity === 'severe' ? 0.9 : (ep.intensity === 'moderate' ? 0.6 : 0.3);
                        state.activity = Math.max(0, 100 - (intensityFactor * 100) + (Math.random() * 20));
                    } else {
                        const gpsDrift = 5 / METERS_PER_DEGREE;
                        state.lngOffset += (Math.random() * gpsDrift * 2) - gpsDrift;
                        state.latOffset += (Math.random() * gpsDrift * 2) - gpsDrift;
                        state.activity = 60 + Math.random() * 30; // normal range
                    }
                } else {
                    // Normal Baseline Drift
                    state.temperature = this.drift(state.temperature, 38.0, 39.2, 0.2);
                    state.heartRate = Math.floor(this.drift(state.heartRate, 60, 80, 5));
                    const gpsDrift = 5 / METERS_PER_DEGREE;
                    state.lngOffset += (Math.random() * gpsDrift * 2) - gpsDrift;
                    state.latOffset += (Math.random() * gpsDrift * 2) - gpsDrift;
                    state.activity = 60 + Math.random() * 30; // normal range 60-90
                }

                // Isolate variables before injecting transient unlabeled noise
                let currentTemp = state.temperature;
                let currentHR = state.heartRate;
                let currentLngOffset = state.lngOffset;
                let currentLatOffset = state.latOffset;

                // 6️⃣ Stochastic Noise Injection (1-tick transients)
                if (isNoise) {
                    const noiseType = Math.random();
                    if (noiseType < 0.33) {
                        currentTemp += (Math.random() > 0.5 ? 1.5 : -1.5);
                    } else if (noiseType < 0.66) {
                        currentHR += Math.floor((Math.random() > 0.5 ? 25 : -25));
                    } else {
                        const spike = 50 / METERS_PER_DEGREE;
                        currentLngOffset += (Math.random() > 0.5 ? spike : -spike);
                        currentLatOffset += (Math.random() > 0.5 ? spike : -spike);
                    }
                }

                state.battery = state.battery > 5 ? state.battery - 0.1 : 100; // Slow drain
                state.signalStrength = Math.floor(this.drift(state.signalStrength, -90, -60, 3));

                // Validate boundaries (Annulus: 200m to 800m radial distance from center)
                const distFromCenterCoords = Math.sqrt(currentLngOffset ** 2 + currentLatOffset ** 2);
                const distFromCenterMeters = distFromCenterCoords * METERS_PER_DEGREE;

                if (distFromCenterMeters < 200 || distFromCenterMeters > 800) {
                    // Push them back to the valid boundary
                    const clampMeters = distFromCenterMeters < 200 ? 200 : 800;
                    const scaleFactor = clampMeters / distFromCenterMeters;
                    currentLngOffset *= scaleFactor;
                    currentLatOffset *= scaleFactor;

                    // Persist boundaries permanently if pushed back
                    state.lngOffset = currentLngOffset;
                    state.latOffset = currentLatOffset;
                }

                const currLng = TARGET_LOCATION[0] + currentLngOffset;
                const currLat = TARGET_LOCATION[1] + currentLatOffset;

                const statusArr = ['Active', 'Active', 'Active', 'Low Battery', 'Offline'];
                const randStatus = state.battery < 20 ? 'Low Battery' : (Math.random() > 0.95 ? 'Offline' : 'Active');

                // Build Time-Series Document
                const newTelemetry = new DeviceTelemetry({
                    deviceId: id,
                    animalId: map._id,
                    tenantId: map.userId,
                    temperature: currentTemp,
                    heartRate: currentHR,
                    activity: state.activity,
                    battery: Math.floor(state.battery),
                    signalStrength: state.signalStrength,
                    deviceStatus: randStatus,
                    location: {
                        type: 'Point',
                        coordinates: [currLng, currLat]
                    },
                    timestamp: new Date()
                });

                // Move to debug to reduce terminal noise (batch inserts will log globally)
                logger.debug({
                    action: 'TELEMETRY_GENERATED',
                    service: 'simulation_engine',
                    tenantId: map.userId,
                    animalId: map._id
                });

                this.telemetryBuffer.push(newTelemetry);
                // Also forward to real-time processors downstream
                DataManagementService.processTelemetryUpdate(newTelemetry);

                // Queue to BullMQ for ML Prediction Async
                try {
                    await queues.mlPredictions.add('predict', {
                        tenantId: map.userId,
                        animalId: map._id,
                        telemetry: {
                            temperature: currentTemp,
                            heartRate: currentHR,
                            activity: state.activity
                        }
                    });
                    logger.info({
                        action: 'PREDICTION_JOB_QUEUED',
                        service: 'simulation_engine',
                        tenantId: map.userId,
                        animalId: map._id
                    });
                } catch (e) {
                    logger.error({
                        action: 'PREDICTION_JOB_QUEUED',
                        result: 'error',
                        error: e.message,
                        service: 'simulation_engine'
                    });
                }
            }

            logger.debug({ action: 'SIMULATION_CYCLE_COMPLETED', service: 'simulation_engine' }, 'Simulation cycle complete');

        } catch (error) {
            logger.error({ action: 'SIMULATION_ERROR', result: 'error', error: error.message, service: 'simulation_engine' }, "[Hardware Simulation Service] Telemetry engine fault");
        }
    }
}

module.exports = new HardwareSimulationService();
