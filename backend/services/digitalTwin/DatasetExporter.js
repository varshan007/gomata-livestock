/**
 * DatasetExporter — Batch Dataset Export for ML Training
 * 
 * GoMata Digital Twin Simulator v2
 * 
 * Exports three dataset types for ML training:
 *   1. Telemetry Table  — Per-reading raw sensor data + hidden labels
 *   2. Feature Table    — 6h engineered features + disease/severity/phase labels
 *   3. Herd-Level Table — Aggregate outbreak metrics + intervention outcomes  
 * 
 * Supports streaming CSV export for large datasets (52M+ rows).
 */

'use strict';

const fs = require('fs');
const path = require('path');
const logger = require('../../utils/logger');

const EnvironmentModel = require('./EnvironmentModel');
const CowPhysiologyEngine = require('./CowPhysiologyEngine');
const SensorGenerator = require('./SensorGenerator');
const EpisodeScheduler = require('./EpisodeScheduler');
const HerdTransmissionModel = require('./HerdTransmissionModel');
const InterventionEngine = require('./InterventionEngine');

const LOG_SERVICE = 'dataset_exporter';

// ── Feature Engineering Helpers ─────────────────────────────────────────────

function avg(arr) { return arr.length ? arr.reduce((s, v) => s + v, 0) / arr.length : 0; }
function std(arr) {
    const m = avg(arr);
    return arr.length > 1 ? Math.sqrt(arr.reduce((s, v) => s + (v - m) ** 2, 0) / (arr.length - 1)) : 0;
}
function slope(arr) {
    if (arr.length < 2) return 0;
    const n = arr.length;
    let sumXY = 0, sumX = 0, sumY = 0, sumX2 = 0;
    for (let i = 0; i < n; i++) {
        sumX += i; sumY += arr[i]; sumXY += i * arr[i]; sumX2 += i * i;
    }
    return (n * sumXY - sumX * sumY) / (n * sumX2 - sumX * sumX || 1);
}
function autocorrelation(arr) {
    if (arr.length < 3) return 0;
    const m = avg(arr);
    let num = 0, den = 0;
    for (let i = 1; i < arr.length; i++) {
        num += (arr[i] - m) * (arr[i - 1] - m);
    }
    for (let i = 0; i < arr.length; i++) {
        den += (arr[i] - m) ** 2;
    }
    return den > 0 ? num / den : 0;
}

// ═════════════════════════════════════════════════════════════════════════════

class DatasetExporter {
    /**
     * @param {Object} config
     * @param {number} config.numCows           - Number of cows to simulate
     * @param {number} config.days              - Simulation duration in days
     * @param {number} config.samplingMinutes   - Sampling interval (default 5)
     * @param {string} config.outputDir         - Output directory path
     * @param {string} [config.tenantId]        - Tenant ID for multi-tenant
     * @param {string} [config.farmId]          - Farm ID
     */
    constructor(config) {
        this.numCows = config.numCows || 100;
        this.days = config.days || 30;
        this.samplingMinutes = config.samplingMinutes || 5;
        this.outputDir = config.outputDir || path.join(__dirname, '../../data/exports');
        this.tenantId = config.tenantId || 'batch_export';
        this.farmId = config.farmId || 'FM-BATCH';

        this.totalTicks = Math.floor((this.days * 24 * 60) / this.samplingMinutes);
        this.featureWindowTicks = Math.floor(360 / this.samplingMinutes); // 6h window = 72 ticks
    }

    // ─────────────────────────────────────────────────────────────────────────
    // MAIN: Generate full dataset
    // ─────────────────────────────────────────────────────────────────────────

    async generateAll() {
        logger.info({
            service: LOG_SERVICE,
            action: 'EXPORT_START',
            numCows: this.numCows,
            days: this.days,
            totalTicks: this.totalTicks,
            estimatedRows: this.numCows * this.totalTicks
        }, `[DatasetExporter] Starting: ${this.numCows} cows × ${this.days} days = ${(this.numCows * this.totalTicks / 1e6).toFixed(1)}M rows`);

        // Ensure output dir exists
        fs.mkdirSync(this.outputDir, { recursive: true });

        const startTime = Date.now();

        // ── 1. Generate Telemetry + Feature datasets ─────────────────
        await this._generateTelemetryAndFeatures();

        // ── 2. Generate Herd-Level dataset ───────────────────────────
        await this._generateHerdDataset();

        // ── 3. Generate Intervention outcomes ────────────────────────
        await this._generateInterventionDataset();

        const elapsed = ((Date.now() - startTime) / 1000).toFixed(1);
        logger.info({
            service: LOG_SERVICE,
            action: 'EXPORT_COMPLETE',
            timeSec: elapsed
        }, `[DatasetExporter] Complete in ${elapsed}s`);

        return {
            telemetryFile: path.join(this.outputDir, 'telemetry.csv'),
            featureFile: path.join(this.outputDir, 'features_6h.csv'),
            herdFile: path.join(this.outputDir, 'herd_metrics.csv'),
            interventionFile: path.join(this.outputDir, 'intervention_outcomes.csv')
        };
    }

    // ─────────────────────────────────────────────────────────────────────────
    // TELEMETRY + FEATURE GENERATION
    // ─────────────────────────────────────────────────────────────────────────

    async _generateTelemetryAndFeatures() {
        const environment = new EnvironmentModel({
            startDate: new Date('2024-01-01T00:00:00Z')
        });

        // Create cow engines
        const cowIds = [];
        const engines = new Map();
        const sensors = new Map();
        const histories = new Map(); // For 6h feature engineering

        for (let i = 0; i < this.numCows; i++) {
            const cowId = `cow_${String(i).padStart(4, '0')}`;
            cowIds.push(cowId);
            const engine = new CowPhysiologyEngine(cowId);
            engines.set(cowId, engine);
            sensors.set(cowId, new SensorGenerator(engine.individualParams));
            histories.set(cowId, { temps: [], hrs: [], activities: [], resps: [], rums: [] });
        }

        // Schedule episodes
        const scheduler = new EpisodeScheduler({
            totalTicks: this.totalTicks,
            tickMinutes: this.samplingMinutes,
            numCows: this.numCows
        });
        const schedule = scheduler.generateSchedule(cowIds);

        // Open streaming CSV writers
        const telemetryPath = path.join(this.outputDir, 'telemetry.csv');
        const featurePath = path.join(this.outputDir, 'features_6h.csv');

        const telemetryCols = [
            'timestamp', 'tenant_id', 'farm_id', 'zone_id', 'cow_id',
            'temperature_C', 'heart_rate_bpm', 'respiration_bpm',
            'activity_index', 'rumination_minutes', 'lying_minutes',
            'gps_lat', 'gps_lon', 'thi', 'ammonia_ppm', 'airflow_rate',
            'stocking_density', 'infection_load', 'stress_load',
            'episode_phase', 'disease_label', 'severity_level'
        ];

        const featureCols = [
            'timestamp', 'cow_id', 'temp_current', 'temp_6h_avg', 'temp_6h_std',
            'temp_6h_slope', 'hr_current', 'hr_6h_avg', 'hr_zscore',
            'activity_ratio', 'rumination_drop', 'autocorrelation_temp',
            'cv_temp', 'hsi', 'stress_index',
            'disease_label', 'severity_level', 'phase_label'
        ];

        const telemetryStream = fs.createWriteStream(telemetryPath);
        const featureStream = fs.createWriteStream(featurePath);

        telemetryStream.write(telemetryCols.join(',') + '\n');
        featureStream.write(featureCols.join(',') + '\n');

        // ── Simulation loop ──────────────────────────────────────────
        const baseDate = new Date('2024-01-01T00:00:00Z');

        for (let tick = 0; tick < this.totalTicks; tick++) {
            const timestamp = new Date(baseDate.getTime() + tick * this.samplingMinutes * 60000);
            const env = environment.getEnvironment(tick, this.samplingMinutes, this.numCows);
            const circTemp = environment.getCircadianTempOffset(env.hourOfDay);
            const circAct = environment.getCircadianActivityMultiplier(env.hourOfDay);

            for (const cowId of cowIds) {
                const engine = engines.get(cowId);
                const sensor = sensors.get(cowId);
                const history = histories.get(cowId);

                // Check episodes
                const triggered = scheduler.getTriggeredEpisodes(cowId, tick, schedule);
                for (const ep of triggered) {
                    if (ep.type === 'infection' && engine.isSusceptible()) {
                        engine.seedInfection(ep);
                    }
                }

                const stressBoost = scheduler.getActiveStressBoost(cowId, tick, schedule);
                const baseStress = environment.computeStressLoad(env);
                const totalStress = baseStress + stressBoost;

                // Evolve & generate
                const hidden = engine.evolve(totalStress);
                const reading = sensor.generate(hidden, env, circTemp, circAct);

                // GPS (deterministic for batch)
                const gpsLat = 28.634 + (parseInt(cowId.split('_')[1]) * 0.0001);
                const gpsLon = 77.161 + (parseInt(cowId.split('_')[1]) * 0.00005);

                // Write telemetry row
                const telemetryRow = [
                    timestamp.toISOString(),
                    this.tenantId,
                    this.farmId,
                    `ZN-${this.farmId}-01`,
                    cowId,
                    reading.temperature,
                    reading.heartRate,
                    reading.respiration,
                    reading.activity,
                    reading.rumination,
                    reading.lying,
                    gpsLat.toFixed(6),
                    gpsLon.toFixed(6),
                    env.thi,
                    env.ammonia,
                    env.airflow,
                    env.stockingDensity,
                    hidden.infectionLoad,
                    hidden.stressLoad,
                    hidden.episodePhase,
                    hidden.diseaseLabel,
                    hidden.severityLevel
                ];
                telemetryStream.write(telemetryRow.join(',') + '\n');

                // Update history for feature engineering
                history.temps.push(reading.temperature);
                history.hrs.push(reading.heartRate);
                history.activities.push(reading.activity);
                history.resps.push(reading.respiration);
                history.rums.push(reading.rumination);

                // Trim to 6h window
                if (history.temps.length > this.featureWindowTicks) {
                    history.temps.shift();
                    history.hrs.shift();
                    history.activities.shift();
                    history.resps.shift();
                    history.rums.shift();
                }

                // Write feature row (only after enough data)
                if (history.temps.length >= 12) { // Minimum 1 hour of data
                    const t = history.temps;
                    const h = history.hrs;
                    const tAvg = avg(t), tStd = std(t);
                    const hAvg = avg(h), hStd = std(h);
                    const aAvg = avg(history.activities);
                    const rBaseline = engine.individualParams.ruminationBaseline;

                    const featureRow = [
                        timestamp.toISOString(),
                        cowId,
                        reading.temperature,
                        tAvg.toFixed(2),
                        tStd.toFixed(3),
                        slope(t).toFixed(4),
                        reading.heartRate,
                        hAvg.toFixed(1),
                        hStd > 0 ? ((reading.heartRate - hAvg) / hStd).toFixed(3) : '0',
                        aAvg > 0.01 ? (reading.activity / aAvg).toFixed(3) : '1',
                        rBaseline > 0 ? ((rBaseline - reading.rumination) / rBaseline).toFixed(3) : '0',
                        autocorrelation(t).toFixed(4),
                        tAvg > 0 ? (tStd / tAvg).toFixed(4) : '0',
                        reading.heatStressIndex,
                        hidden.stressLoad.toFixed(4),
                        hidden.diseaseLabel,
                        hidden.severityLevel,
                        hidden.episodePhase
                    ];
                    featureStream.write(featureRow.join(',') + '\n');
                }
            }

            // Log progress every day of simulation
            if (tick > 0 && tick % (this.totalTicks / Math.min(this.days, 30) | 0) === 0) {
                const dayNum = Math.floor(tick / (1440 / this.samplingMinutes));
                logger.info({
                    service: LOG_SERVICE,
                    action: 'GENERATE_PROGRESS',
                    day: dayNum,
                    totalDays: this.days,
                    tick,
                    totalTicks: this.totalTicks
                });
            }
        }

        telemetryStream.end();
        featureStream.end();

        logger.info({
            service: LOG_SERVICE,
            action: 'TELEMETRY_EXPORT_DONE',
            rows: this.numCows * this.totalTicks,
            file: telemetryPath
        });
    }

    // ─────────────────────────────────────────────────────────────────────────
    // HERD-LEVEL DATASET
    // ─────────────────────────────────────────────────────────────────────────

    async _generateHerdDataset() {
        const herdPath = path.join(this.outputDir, 'herd_metrics.csv');
        const cols = [
            'timestamp', 'herd_size', 'infected_count', 'infection_burden',
            'spread_velocity', 'herd_stability_index', 'forecast_peak_7day',
            'projected_loss', 'intervention_type'
        ];

        const stream = fs.createWriteStream(herdPath);
        stream.write(cols.join(',') + '\n');

        // Run herd simulation with transmission
        const environment = new EnvironmentModel();
        const herdSize = Math.min(this.numCows, 200);
        const cowIds = [];
        const engines = new Map();

        for (let i = 0; i < herdSize; i++) {
            const cowId = `herd_cow_${i}`;
            cowIds.push(cowId);
            engines.set(cowId, new CowPhysiologyEngine(cowId));
        }

        const scheduler = new EpisodeScheduler({
            totalTicks: Math.floor(90 * 288), // 90 days
            tickMinutes: 5,
            numCows: herdSize
        });
        const schedule = scheduler.generateSchedule(cowIds);

        const transmitter = new HerdTransmissionModel({ r0: 2.5, penDensity: 'medium' });

        // GPS metadata for distance calc
        const cowMeta = new Map();
        for (const cowId of cowIds) {
            const d = Math.random() * 30;
            const a = Math.random() * 2 * Math.PI;
            cowMeta.set(cowId, {
                animal: { zone_id: 'herd_zone' },
                lngOffset: d * Math.cos(a) / 111139,
                latOffset: d * Math.sin(a) / 111139
            });
        }

        const herdTicks = Math.floor(90 * 288); // 90 days
        const baseDate = new Date('2024-01-01T00:00:00Z');

        for (let tick = 0; tick < herdTicks; tick++) {
            const env = environment.getEnvironment(tick, 5, herdSize);
            const baseStress = environment.computeStressLoad(env);

            for (const cowId of cowIds) {
                const engine = engines.get(cowId);
                const triggered = scheduler.getTriggeredEpisodes(cowId, tick, schedule);
                for (const ep of triggered) {
                    if (ep.type === 'infection' && engine.isSusceptible()) {
                        engine.seedInfection(ep);
                    }
                }
                const stressBoost = scheduler.getActiveStressBoost(cowId, tick, schedule);
                engine.evolve(baseStress + stressBoost);
            }

            // Check herd transmission
            transmitter.checkTransmission(engines, cowMeta, tick);

            // Record metrics every hour (12 ticks)
            if (tick % 12 === 0) {
                const metrics = transmitter.computeHerdMetrics(engines, tick);
                const timestamp = new Date(baseDate.getTime() + tick * 5 * 60000);
                const projectedLoss = metrics.infectedCount * 500 * (metrics.infectedCount / herdSize);

                const row = [
                    timestamp.toISOString(),
                    herdSize,
                    metrics.infectedCount,
                    metrics.infectionBurden,
                    metrics.spreadVelocity,
                    metrics.herdStabilityIndex,
                    metrics.forecastPeak7Day,
                    Math.round(projectedLoss),
                    'none'
                ];
                stream.write(row.join(',') + '\n');
            }
        }

        stream.end();
        logger.info({ service: LOG_SERVICE, action: 'HERD_EXPORT_DONE', file: herdPath });
    }

    // ─────────────────────────────────────────────────────────────────────────
    // INTERVENTION DATASET
    // ─────────────────────────────────────────────────────────────────────────

    async _generateInterventionDataset() {
        const intervPath = path.join(this.outputDir, 'intervention_outcomes.csv');

        const engine = new InterventionEngine({ numScenarios: 1000 });
        const results = await engine.runAll();

        const cols = [
            'scenario_id', 'herd_size', 'vaccination_coverage',
            'isolation_strategy', 'r0', 'pen_density', 'initial_infected',
            'strategy_key', 'strategy_name',
            'peak_infected', 'peak_infected_fraction', 'outbreak_duration_days',
            'total_transmissions', 'economic_loss', 'intervention_cost',
            'risk_adjusted_cost', 'peak_reduction', 'duration_reduction',
            'economic_saving', 'net_benefit', 'cost_effectiveness_ratio', 'is_optimal'
        ];

        const stream = fs.createWriteStream(intervPath);
        stream.write(cols.join(',') + '\n');

        for (const r of results) {
            const row = [
                r.scenarioId, r.herdSize, r.vaccinationCoverage,
                r.isolationStrategy, r.r0, r.penDensity, r.initialInfected,
                r.strategyKey, `"${r.strategyName}"`,
                r.peakInfected, r.peakInfectedFraction, r.outbreakDurationDays,
                r.totalTransmissions, r.economicLoss, r.interventionCost,
                r.riskAdjustedCost, r.peakReduction || 0, r.durationReduction || 0,
                r.economicSaving || 0, r.netBenefit || 0,
                r.costEffectivenessRatio || 0, r.isOptimal ? 1 : 0
            ];
            stream.write(row.join(',') + '\n');
        }

        stream.end();
        logger.info({
            service: LOG_SERVICE,
            action: 'INTERVENTION_EXPORT_DONE',
            outcomes: results.length,
            file: intervPath
        });
    }
}

module.exports = DatasetExporter;
