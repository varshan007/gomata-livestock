/**
 * DigitalTwinSimulator — GoMata Hidden-State Livestock Digital Twin v2
 * 
 * Production-grade replacement for HardwareSimulationService.
 * 
 * Architecture:
 *   EnvironmentModel     → Zone-level weather, THI, ammonia
 *   EpisodeScheduler     → Pre-plans infection/stress episodes per cow
 *   CowPhysiologyEngine  → 5 hidden-state evolution equations per cow
 *   SensorGenerator      → Converts hidden states to observable sensors
 *   HerdTransmissionModel → Contact-based disease spread (optional)
 *   InterventionEngine   → Strategy simulation (optional)
 *   DatasetExporter      → Batch CSV/JSON export (optional)
 * 
 * Modes:
 *   STREAMING — Real-time tick loop (replaces v1 behavior), emits to
 *               DeviceTelemetry + EventBus + BullMQ
 *   BATCH     — Generates full historical dataset to CSV
 * 
 * Data Flow (IDENTICAL to v1):
 *   tick() → DeviceTelemetry.insertMany() 
 *          → DataManagementService.processTelemetryUpdate()
 *          → queues.mlPredictions.add()
 */

'use strict';

const logger = require('../../utils/logger');
const DeviceTelemetry = require('../../models/DeviceTelemetry');
const LivestockMaster = require('../../models/LivestockMaster');
const TrainingEvent = require('../../models/TrainingEvent');
const DataManagementService = require('../DataManagementService');
const { queues } = require('../../config/bullmq');

const EnvironmentModel = require('./EnvironmentModel');
const CowPhysiologyEngine = require('./CowPhysiologyEngine');
const SensorGenerator = require('./SensorGenerator');
const EpisodeScheduler = require('./EpisodeScheduler');

// ── Constants ───────────────────────────────────────────────────────────────

const LOG_SERVICE = 'digital_twin_v2';
const TARGET_LOCATION = [77.161358, 28.634064]; // [lng, lat] — default farm center
const METERS_PER_DEGREE = 111139;

const DEFAULTS = {
    TICK_MS: parseInt(process.env.SIMULATION_TICK_MS || '20000'),  // 20s default
    TICK_MINUTES: 5,         // Each tick represents 5 sim-minutes
    BATCH_FLUSH_MS: 500,     // Buffer flush interval
    TRAINING_SAMPLE_RATE: 0.08, // 8% of ticks generate training labels
};

// ═════════════════════════════════════════════════════════════════════════════

class DigitalTwinSimulator {
    constructor() {
        this.intervalId = null;
        this.flushIntervalId = null;

        // ── Per-cow engines ──────────────────────────────────────────
        this.cowEngines = new Map();        // cowId → CowPhysiologyEngine
        this.sensorGenerators = new Map();  // cowId → SensorGenerator
        this.cowMetadata = new Map();       // cowId → { animal, gpsOffset }
        this.featureWindows = new Map();    // cowId → { temps[], hrs[], acts[], resps[], rums[] }

        // ── Shared subsystems ────────────────────────────────────────
        this.environment = new EnvironmentModel();
        this.episodeScheduler = null;
        this.schedule = null;

        // ── Tick counter ─────────────────────────────────────────────
        this.currentTick = 0;

        // ── Batch buffers (same pattern as v1) ───────────────────────
        this.telemetryBuffer = [];
        this.trainingEventBuffer = [];

        // ── Cluster support ──────────────────────────────────────────
        this.workerIndex = parseInt(process.env.WORKER_INDEX || '0');
        this.totalWorkers = parseInt(process.env.TOTAL_WORKERS || '1');

        // ── Metrics ──────────────────────────────────────────────────
        this.metrics = {
            totalTicks: 0,
            telemetryGenerated: 0,
            trainingEventsGenerated: 0,
            episodesTriggered: 0,
            transmissions: 0
        };
    }

    // ─────────────────────────────────────────────────────────────────────────
    // LIFECYCLE
    // ─────────────────────────────────────────────────────────────────────────

    async start() {
        if (this.intervalId) return;

        logger.info({
            action: 'DIGITAL_TWIN_STARTING',
            service: LOG_SERVICE,
            version: 'v2',
            workerIndex: this.workerIndex,
            totalWorkers: this.totalWorkers,
            tickMs: DEFAULTS.TICK_MS
        }, `🧬 [DigitalTwin v2] Starting Hidden-State Livestock Simulator`);

        // ── 1. Load animals from DB ──────────────────────────────────
        await this._initializeCowEngines();

        if (this.cowEngines.size === 0) {
            logger.warn({ service: LOG_SERVICE }, '[DigitalTwin v2] No animals found. Waiting...');
        }

        // ── 2. Pre-schedule episodes ─────────────────────────────────
        this._initializeEpisodeSchedule();

        // ── 3. Start tick loop ───────────────────────────────────────
        this.intervalId = setInterval(() => this._tick(), DEFAULTS.TICK_MS);

        // ── 4. Start batch flush ─────────────────────────────────────
        this.flushIntervalId = setInterval(() => this._flushBuffers(), DEFAULTS.BATCH_FLUSH_MS);

        logger.info({
            service: LOG_SERVICE,
            action: 'DIGITAL_TWIN_STARTED',
            cowCount: this.cowEngines.size,
            scheduleStats: this.schedule ? this.episodeScheduler.getScheduleStats(this.schedule) : null
        }, `🧬 [DigitalTwin v2] Started with ${this.cowEngines.size} cows`);
    }

    stop() {
        if (this.intervalId) clearInterval(this.intervalId);
        if (this.flushIntervalId) clearInterval(this.flushIntervalId);
        this.intervalId = null;
        this.flushIntervalId = null;
        logger.info({ service: LOG_SERVICE }, '[DigitalTwin v2] Stopped');
    }

    // ─────────────────────────────────────────────────────────────────────────
    // INITIALIZATION
    // ─────────────────────────────────────────────────────────────────────────

    async _initializeCowEngines() {
        try {
            const animals = await LivestockMaster.find({
                livestock_id: { $exists: true },
                device_id: { $exists: true },
                userId: { $exists: true }
            }).lean();

            // Cluster partitioning (same logic as v1)
            const myAnimals = animals.filter((_, idx) =>
                idx % this.totalWorkers === this.workerIndex
            );

            logger.info({
                service: LOG_SERVICE,
                action: 'ANIMALS_LOADED',
                total: animals.length,
                assigned: myAnimals.length,
                workerIndex: this.workerIndex
            });

            for (const animal of myAnimals) {
                const cowId = animal._id.toString();

                // Create physiology engine with individual variation
                const engine = new CowPhysiologyEngine(cowId, {
                    age: animal.age || (2 + Math.random() * 10),
                    breed: animal.breed || 'unknown'
                });

                // Create sensor generator using the engine's individual params
                const sensor = new SensorGenerator(engine.individualParams);

                // Generate GPS offset within farm radius
                const distM = 200 + Math.random() * 600; // 200-800m from center
                const angle = Math.random() * 2 * Math.PI;
                const lngOffset = (distM * Math.cos(angle)) / METERS_PER_DEGREE;
                const latOffset = (distM * Math.sin(angle)) / METERS_PER_DEGREE;

                this.cowEngines.set(cowId, engine);
                this.sensorGenerators.set(cowId, sensor);
                this.cowMetadata.set(cowId, {
                    animal,
                    lngOffset,
                    latOffset,
                    baseActivity: engine.individualParams.activityBaseline
                });
                this.featureWindows.set(cowId, {
                    temps: [], hrs: [], acts: [], resps: [], rums: []
                });
            }
        } catch (err) {
            logger.error({
                service: LOG_SERVICE,
                action: 'INIT_ERROR',
                error: err.message
            }, '[DigitalTwin v2] Failed to initialize cow engines');
        }
    }

    _initializeEpisodeSchedule() {
        if (this.cowEngines.size === 0) return;

        const cowIds = Array.from(this.cowEngines.keys());

        // Estimate total ticks for 180 days (or continuous if streaming)
        const estimatedTicks = Math.floor((180 * 24 * 60) / DEFAULTS.TICK_MINUTES);

        this.episodeScheduler = new EpisodeScheduler({
            totalTicks: estimatedTicks,
            tickMinutes: DEFAULTS.TICK_MINUTES,
            numCows: cowIds.length,
            farmType: 'dairy'  // Epidemiologically calibrated
        });

        this.schedule = this.episodeScheduler.generateSchedule(cowIds);

        const stats = this.episodeScheduler.getScheduleStats(this.schedule);
        logger.info({
            service: LOG_SERVICE,
            action: 'SCHEDULE_CREATED',
            ...stats
        }, `[DigitalTwin v2] Episode schedule: ${stats.totalInfections} infections, ${stats.totalStressWaves} stress waves`);
    }

    // ─────────────────────────────────────────────────────────────────────────
    // MAIN TICK — Core simulation loop
    // ─────────────────────────────────────────────────────────────────────────

    async _tick() {
        const tickStart = Date.now();
        this.currentTick++;
        this.metrics.totalTicks++;

        try {
            // Reload cows if empty (new animals may have been registered)
            if (this.cowEngines.size === 0) {
                await this._initializeCowEngines();
                if (this.cowEngines.size > 0) {
                    this._initializeEpisodeSchedule();
                }
                if (this.cowEngines.size === 0) return;
            }

            // ── 1. Environment snapshot ──────────────────────────────
            const animalsInZone = this.cowEngines.size;
            const env = this.environment.getEnvironment(
                this.currentTick, DEFAULTS.TICK_MINUTES, animalsInZone
            );
            const circadianTemp = this.environment.getCircadianTempOffset(env.hourOfDay);
            const circadianAct = this.environment.getCircadianActivityMultiplier(env.hourOfDay);

            // ── 2. Process each cow ──────────────────────────────────
            for (const [cowId, engine] of this.cowEngines) {
                const metadata = this.cowMetadata.get(cowId);
                const sensor = this.sensorGenerators.get(cowId);

                // ── 2a. Check for scheduled episodes ─────────────────
                if (this.schedule) {
                    const triggered = this.episodeScheduler.getTriggeredEpisodes(
                        cowId, this.currentTick, this.schedule
                    );
                    for (const ep of triggered) {
                        if (ep.type === 'infection' && engine.isSusceptible()) {
                            engine.seedInfection(ep);
                            this.metrics.episodesTriggered++;
                            logger.info({
                                service: LOG_SERVICE,
                                action: 'EPISODE_TRIGGERED',
                                cowId,
                                type: ep.diseaseType,
                                severity: ep.targetSeverity
                            });
                        }
                    }

                    // Stress wave boost
                    const stressBoost = this.episodeScheduler.getActiveStressBoost(
                        cowId, this.currentTick, this.schedule
                    );
                    env._stressBoost = stressBoost;
                }

                // ── 2b. Compute environmental stress ─────────────────
                const baseStress = this.environment.computeStressLoad(env);
                const totalStress = baseStress + (env._stressBoost || 0);

                // ── 2c. Evolve hidden states ─────────────────────────
                const hiddenState = engine.evolve(totalStress);

                // ── 2d. Generate sensor readings ─────────────────────
                const sensors = sensor.generate(
                    hiddenState, env, circadianTemp, circadianAct
                );

                // ── 2e. Generate GPS position (NaN-safe) ────────────
                const safeGpsRadius = isFinite(sensors.gpsRadius) ? sensors.gpsRadius : 100;
                const gpsJitter = safeGpsRadius / METERS_PER_DEGREE * 0.01;
                let currLng = TARGET_LOCATION[0] + metadata.lngOffset + (Math.random() - 0.5) * gpsJitter;
                let currLat = TARGET_LOCATION[1] + metadata.latOffset + (Math.random() - 0.5) * gpsJitter;

                // Final NaN guard for coordinates
                if (!isFinite(currLng)) currLng = TARGET_LOCATION[0] + metadata.lngOffset;
                if (!isFinite(currLat)) currLat = TARGET_LOCATION[1] + metadata.latOffset;

                // NaN-safe helper
                const safeNum = (v, fallback) => isFinite(v) ? v : fallback;

                // ── 2f. Build telemetry document ─────────────────────
                const animal = metadata.animal;
                const deviceStatus = Math.random() > 0.95 ? 'Offline' : 'Active';

                const telemetry = {
                    deviceId: animal.device_id,
                    animalId: animal._id,
                    tenantId: animal.userId,
                    temperature: safeNum(sensors.temperature, 38.5),
                    heartRate: safeNum(sensors.heartRate, 65),
                    respiration: safeNum(sensors.respiration, 26),
                    activity: Math.round(safeNum(sensors.activity, 0.75) * 100), // 0–1 → 0–100 for UI
                    rumination: safeNum(sensors.rumination, 35),
                    lyingTime: safeNum(sensors.lying, 25),
                    battery: Math.max(10, 100 - Math.floor(this.currentTick * 0.001)),
                    signalStrength: -50 - Math.floor(Math.random() * 30),
                    deviceStatus,
                    location: {
                        type: 'Point',
                        coordinates: [
                            parseFloat(currLng.toFixed(6)),
                            parseFloat(currLat.toFixed(6))
                        ]
                    },
                    // Environmental context
                    thi: safeNum(env.thi, 72),
                    ammonia: safeNum(env.ammonia, 8),
                    timestamp: new Date()
                };

                this.telemetryBuffer.push(telemetry);
                this.metrics.telemetryGenerated++;

                // ── 2g. Forward to real-time processors ──────────────
                // Same as v1 — keeps downstream pipeline intact
                DataManagementService.processTelemetryUpdate(telemetry);

                // ── 2h. Queue ML prediction ──────────────────────────
                try {
                    await queues.mlPredictions.add('predict', {
                        tenantId: animal.userId,
                        animalId: animal._id,
                        telemetry: {
                            temperature: sensors.temperature,
                            heartRate: sensors.heartRate,
                            activity: sensors.activity
                        }
                    });
                } catch (e) {
                    // Non-fatal — BullMQ might not be available
                }

                // ── 2i. Update feature window ────────────────────
                const win = this.featureWindows.get(cowId);
                if (win) {
                    const WINDOW = 72; // 6h at 5-min ticks
                    win.temps.push(sensors.temperature);
                    win.hrs.push(sensors.heartRate);
                    win.acts.push(sensors.activity);
                    win.resps.push(sensors.respiration);
                    win.rums.push(sensors.rumination);
                    if (win.temps.length > WINDOW) {
                        win.temps.shift(); win.hrs.shift(); win.acts.shift();
                        win.resps.shift(); win.rums.shift();
                    }
                }

                // ── 2j. Generate training labels (sampled) ───────────
                if (Math.random() < DEFAULTS.TRAINING_SAMPLE_RATE && win && win.temps.length >= 12) {
                    this._generateTrainingEvent(animal, sensors, hiddenState, env, engine, win, currLat, currLng);
                }
            }

            // ── 3. Log tick metrics periodically ─────────────────────
            if (this.currentTick % 50 === 0) {
                const durationMs = Date.now() - tickStart;
                logger.info({
                    service: LOG_SERVICE,
                    action: 'TICK_SUMMARY',
                    tick: this.currentTick,
                    cows: this.cowEngines.size,
                    durationMs,
                    telemetryTotal: this.metrics.telemetryGenerated,
                    episodesTriggered: this.metrics.episodesTriggered
                });
            }

        } catch (error) {
            logger.error({
                service: LOG_SERVICE,
                action: 'TICK_ERROR',
                tick: this.currentTick,
                error: error.message,
                stack: error.stack
            }, `[DigitalTwin v2] Tick error: ${error.message}`);
        }
    }

    // ─────────────────────────────────────────────────────────────────────────
    // TRAINING EVENT GENERATION — 10-Section Production Schema (v3)
    // ─────────────────────────────────────────────────────────────────────────

    _generateTrainingEvent(animal, sensors, hiddenState, env, engine, win, lat, lng) {
        const safe = (v, fb = 0) => isFinite(v) ? v : fb;
        const p = engine.individualParams;

        // ── Statistical helpers ───────────────────────────────
        const avg = arr => arr.length ? arr.reduce((s, v) => s + v, 0) / arr.length : 0;
        const std = arr => {
            const m = avg(arr);
            return arr.length > 1 ? Math.sqrt(arr.reduce((s, v) => s + (v - m) ** 2, 0) / (arr.length - 1)) : 0;
        };
        const slope = arr => {
            if (arr.length < 2) return 0;
            const n = arr.length;
            let sXY = 0, sX = 0, sY = 0, sX2 = 0;
            for (let i = 0; i < n; i++) { sX += i; sY += arr[i]; sXY += i * arr[i]; sX2 += i * i; }
            return (n * sXY - sX * sY) / (n * sX2 - sX * sX || 1);
        };
        const autocorr = arr => {
            if (arr.length < 3) return 0;
            const m = avg(arr);
            let num = 0, den = 0;
            for (let i = 1; i < arr.length; i++) num += (arr[i] - m) * (arr[i - 1] - m);
            for (let i = 0; i < arr.length; i++) den += (arr[i] - m) ** 2;
            return den > 0 ? num / den : 0;
        };

        // ── 6h window computations ────────────────────────────
        const tAvg = avg(win.temps), tStd = std(win.temps);
        const hAvg = avg(win.hrs), hStd = std(win.hrs);
        const aAvg = avg(win.acts), aStd = std(win.acts);

        // ── Episode tracking ───────────────────────────────
        const ticksPerDay = 1440 / DEFAULTS.TICK_MINUTES;
        const episodeDayIndex = engine.ticksSinceEpisodeStart > 0
            ? parseFloat((engine.ticksSinceEpisodeStart / ticksPerDay).toFixed(2)) : 0;
        const timeSinceStart = engine.ticksSinceEpisodeStart * DEFAULTS.TICK_MINUTES;
        const episodeId = engine.currentEpisode
            ? `${engine.cowId}_ep${engine.episodeCount}` : null;

        // ── Severity as numeric ──────────────────────────────
        const severityMap = { 'none': 0, 'mild': 1, 'moderate': 2, 'severe': 3 };
        const severityNum = severityMap[hiddenState.severityLevel] ?? 0;

        // ── Composite stress index (observable) ────────────────
        const ruminBaseline = safe(p.ruminationBaseline, 35);
        const ruminDrop = ruminBaseline > 0 ? safe((ruminBaseline - sensors.rumination) / ruminBaseline, 0) : 0;
        const stressIndex = safe(0.4 * sensors.heatStressIndex + 0.3 * Math.max(0, ruminDrop) + 0.3 * (1 - sensors.activity), 0);

        const now = new Date();

        // ── Forecast labels (schedule-only, leakage-safe) ─────────
        let forecastLabels = { infection_in_24h: 0, stress_in_24h: 0 };
        if (this.episodeScheduler && this.schedule) {
            forecastLabels = this.episodeScheduler.getForecastLabels(
                cowId, this.currentTick, this.schedule
            );
        }

        // ═══ 10-Section Production Document ════════════════════════
        const trainingEvent = {

            // 1️⃣ Multi-Tenant Context
            tenantId: animal.userId,
            farmId: animal.farmId || null,
            zoneId: animal.zoneId || null,
            animalId: animal._id,

            // 2️⃣ Temporal Context + Versioning
            timestamp: now,
            simulationVersion: 'digital_twin_v3',
            featureVersion: 'v4_windowed',
            episodeId,
            episodeDayIndex,
            timeSinceEpisodeStart: timeSinceStart,

            // 3️⃣ Static Animal Profile (snapshot)
            animalProfile: {
                parity: p.parity || 0,
                lactationStage: p.lactationStage || 'dry',
                bodyConditionScore: safe(p.bcs, 3.0),
                geneticHeatTolerance: p.heatTolerance || 'medium',
                previousDiseaseCount: engine.episodeCount
            },

            // 4️⃣ Raw Sensor Signals
            signals: {
                temperature_C: safe(sensors.temperature, 38.5),
                heartRate_bpm: safe(sensors.heartRate, 65),
                respiration_bpm: safe(sensors.respiration, 26),
                activity_index: safe(sensors.activity, 0.75),
                rumination_min: safe(sensors.rumination, 35),
                lying_min: safe(sensors.lying, 25),
                gps: {
                    lat: safe(lat, TARGET_LOCATION[1]),
                    lon: safe(lng, TARGET_LOCATION[0])
                }
            },

            // 5️⃣ Environmental Signals
            environment: {
                ambientTemp_C: safe(env.ambientTemp, 28),
                humidity_pct: safe(env.humidity, 55),
                thi: safe(env.thi, 72),
                ammonia_ppm: safe(env.ammonia, 8),
                airflow_rate: safe(env.airflow, 1.5),
                stocking_density_raw: safe(env.stockingDensity_raw, 0.1),
                stocking_density_normalized: safe(env.stockingDensity_normalized, 0.03)
            },

            // 6️⃣ Window-Based Engineered Features
            features: {
                temp_current: safe(sensors.temperature, 38.5),
                temp_6h_avg: parseFloat(safe(tAvg, 38.5).toFixed(2)),
                temp_6h_std: parseFloat(safe(tStd, 0).toFixed(3)),
                temp_6h_slope: parseFloat(safe(slope(win.temps), 0).toFixed(4)),
                temp_zscore: tStd > 0 ? parseFloat(safe((sensors.temperature - tAvg) / tStd, 0).toFixed(3)) : 0,

                hr_current: safe(sensors.heartRate, 65),
                hr_6h_avg: parseFloat(safe(hAvg, 65).toFixed(1)),
                hr_6h_std: parseFloat(safe(hStd, 0).toFixed(2)),
                hr_6h_slope: parseFloat(safe(slope(win.hrs), 0).toFixed(4)),
                hr_zscore: hStd > 0 ? parseFloat(safe((sensors.heartRate - hAvg) / hStd, 0).toFixed(3)) : 0,

                activity_current: safe(sensors.activity, 0.75),
                activity_6h_avg: parseFloat(safe(aAvg, 0.75).toFixed(3)),
                activity_6h_std: parseFloat(safe(aStd, 0).toFixed(4)),
                activity_6h_slope: parseFloat(safe(slope(win.acts), 0).toFixed(4)),
                activity_ratio: aAvg > 0.01 ? parseFloat(safe(sensors.activity / aAvg, 1).toFixed(3)) : 1,

                rumination_drop: parseFloat(safe(ruminDrop, 0).toFixed(3)),

                autocorrelation_temp: parseFloat(safe(autocorr(win.temps), 0).toFixed(4)),
                coefficient_variation_temp: tAvg > 0 ? parseFloat(safe(tStd / tAvg, 0).toFixed(4)) : 0,

                heat_stress_index: safe(sensors.heatStressIndex, 0),
                composite_stress_index: parseFloat(safe(stressIndex, 0).toFixed(3)),
                // Raw stress components (let ML learn weighting)
                heat_component: parseFloat(safe(1 / (1 + Math.exp(-(env.thi - 72) / 8)), 0.5).toFixed(4)),
                air_quality_component: parseFloat(safe(Math.max(0, Math.min(1, (env.ammonia - 5) / 30)), 0).toFixed(4)),
                crowding_component: safe(env.stockingDensity_normalized, 0),
                ventilation_component: parseFloat(safe(Math.min(1, env.airflow / 3.0), 0.5).toFixed(4))
            },

            // 7️⃣ Latent Biological States (NEVER model input)
            hiddenState: {
                infectionLoad: safe(hiddenState.infectionLoad, 0),
                stressLoad: safe(hiddenState.stressLoad, 0),
                immuneResponse: safe(hiddenState.immuneResponse, 0),
                compensationCapacity: safe(hiddenState.compensation, 1),
                fatigue: safe(hiddenState.fatigue, 0),
                compensationCollapse: !!hiddenState.compensationCollapse
            },

            // 8️⃣ Training Labels
            labels: {
                diseaseBinary: hiddenState.diseaseLabel || 0,
                infectionBinary: hiddenState.infectionBinary || 0,
                stressBinary: hiddenState.stressBinary || 0,
                mixedStateBinary: hiddenState.mixedStateBinary || 0,
                severityLevel: severityNum,
                episodePhase: hiddenState.episodePhase || 'healthy',
                diseaseType: hiddenState.diseaseType || 'none',
                // Forecast (from schedule only, NOT hidden state)
                infection_in_24h: forecastLabels.infection_in_24h,
                stress_in_24h: forecastLabels.stress_in_24h
            },

            // 9️⃣ Intervention Context
            interventionContext: {
                vaccinationActive: false,
                isolationActive: false,
                ventilationBoost: false,
                antibioticActive: false
            },

            // 🔟 Data Governance
            source: 'digital_twin_v3'
        };

        this.trainingEventBuffer.push(trainingEvent);
        this.metrics.trainingEventsGenerated++;
    }

    // ─────────────────────────────────────────────────────────────────────────
    // BUFFER FLUSH — Batch insert to MongoDB (same pattern as v1)
    // ─────────────────────────────────────────────────────────────────────────

    async _flushBuffers() {
        const telemetryCount = this.telemetryBuffer.length;
        const eventCount = this.trainingEventBuffer.length;

        if (telemetryCount === 0 && eventCount === 0) return;

        const startMs = Date.now();

        if (telemetryCount > 0) {
            const batch = this.telemetryBuffer.splice(0, telemetryCount);
            try {
                await DeviceTelemetry.insertMany(batch, { ordered: false, lean: true });
            } catch (e) {
                logger.error({
                    service: LOG_SERVICE,
                    action: 'BATCH_WRITE_ERROR',
                    error: e.message
                }, 'Telemetry batch write failed');
            }
        }

        if (eventCount > 0) {
            const batch = this.trainingEventBuffer.splice(0, eventCount);
            try {
                await TrainingEvent.insertMany(batch, { ordered: false, lean: true });
            } catch (e) {
                logger.error({
                    service: LOG_SERVICE,
                    action: 'BATCH_WRITE_ERROR',
                    error: e.message
                }, 'TrainingEvent batch write failed');
            }
        }

        const durationMs = Date.now() - startMs;

        if (telemetryCount > 10 || eventCount > 0) {
            logger.debug({
                service: LOG_SERVICE,
                action: 'BATCH_FLUSH',
                telemetryCount,
                eventCount,
                durationMs
            });
        }
    }

    // ─────────────────────────────────────────────────────────────────────────
    // PUBLIC API
    // ─────────────────────────────────────────────────────────────────────────

    getMetrics() {
        const engineStats = { healthy: 0, infected: 0, stressed: 0 };
        for (const [, engine] of this.cowEngines) {
            const snap = engine.getSnapshot();
            if (snap.diseaseLabel === 1) engineStats.infected++;
            else if (snap.stressLoad > 0.3) engineStats.stressed++;
            else engineStats.healthy++;
        }

        return {
            ...this.metrics,
            currentTick: this.currentTick,
            cowCount: this.cowEngines.size,
            ...engineStats
        };
    }

    /**
     * Get hidden state snapshots for all cows (for monitoring/debugging).
     * NOT exposed to ML models.
     */
    getHiddenStates() {
        const states = {};
        for (const [cowId, engine] of this.cowEngines) {
            states[cowId] = engine.getSnapshot();
        }
        return states;
    }
}

module.exports = DigitalTwinSimulator;
