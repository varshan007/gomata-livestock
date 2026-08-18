#!/usr/bin/env node

/**
 * generateBulkDatasetV4.js — Digital Twin v4 Contextual Causal Twin Bulk Generator
 * 
 * Generates ML training data with:
 *   - Production signals (milk, conductivity, weight, feed, water)
 *   - Management event timelines (vacc, antibiotic, feed change, transport, mixing)
 *   - Context-aware latent state evolution (susceptibility, BCS, recovery mods)
 *   - Separate trainingevents_v4 collection
 * 
 * Usage:
 *   node scripts/generateBulkDatasetV4.js --target 10000000 --cows 200 --days 365 --farm-type dairy
 */

'use strict';

process.env.TRAINING_SAMPLE_RATE = '0';

const mongoose = require('mongoose');

// ── Engine imports ──────────────────────────────────────────────────────────
const EnvironmentModel = require('../services/digitalTwin/EnvironmentModel');
const CowPhysiologyEngine = require('../services/digitalTwin/CowPhysiologyEngine');
const SensorGenerator = require('../services/digitalTwin/SensorGenerator');
const EpisodeScheduler = require('../services/digitalTwin/EpisodeScheduler');
const FarmProfile = require('../services/digitalTwin/FarmProfile');
const ProductionModel = require('../services/digitalTwin/ProductionModel');
const ManagementEventSimulator = require('../services/digitalTwin/ManagementEventSimulator');
const AnimalValidator = require('../services/digitalTwin/AnimalValidator');

// ── CLI args ────────────────────────────────────────────────────────────────
const args = process.argv.slice(2);
const getArg = (name, def) => {
    const idx = args.indexOf(`--${name}`);
    return idx >= 0 && args[idx + 1] ? args[idx + 1] : def;
};
const hasFlag = (name) => args.includes(`--${name}`);

const TARGET_EVENTS = parseInt(getArg('target', '10000000'));
const NUM_COWS = parseInt(getArg('cows', '200'));
const SIM_DAYS = parseInt(getArg('days', '365'));
const BATCH_SIZE = parseInt(getArg('batch', '5000'));
const FARM_TYPE = getArg('farm-type', 'dairy');
const TICK_MINUTES = 5;
const TICKS_PER_DAY = 1440 / TICK_MINUTES; // 288
const TOTAL_TICKS = SIM_DAYS * TICKS_PER_DAY;
const DRY_RUN = hasFlag('dry-run');
const MONGO_URI = process.env.MONGO_URI || 'mongodb://127.0.0.1:27017/livestock_monitoring';

const totalPossible = NUM_COWS * TOTAL_TICKS;
const SAMPLE_RATE = Math.min(1.0, TARGET_EVENTS / totalPossible);

// ── Helpers ─────────────────────────────────────────────────────────────────
const G = '\x1b[32m', C = '\x1b[36m', Y = '\x1b[33m', B = '\x1b[1m', X = '\x1b[0m', M = '\x1b[35m';
const safe = (v, fb = 0) => isFinite(v) ? v : fb;
const avg = arr => arr.length ? arr.reduce((a, b) => a + b, 0) / arr.length : 0;
const stddev = arr => {
    if (arr.length < 2) return 0;
    const m = avg(arr);
    return Math.sqrt(arr.reduce((s, v) => s + (v - m) ** 2, 0) / (arr.length - 1));
};
const lag1Autocorr = arr => {
    if (arr.length < 3) return 0;
    const m = avg(arr);
    let num = 0, den = 0;
    for (let i = 1; i < arr.length; i++) {
        num += (arr[i] - m) * (arr[i - 1] - m);
        den += (arr[i] - m) ** 2;
    }
    return den > 0 ? num / den : 0;
};

// ── Heat tolerance / lactation stress maps ──────────────────────────────────
const HEAT_TOL_FACTOR = { low: 1.3, medium: 1.0, high: 0.7 };
const LACT_STRESS_FACTOR = { early: 1.3, mid: 1.1, late: 1.0, dry: 0.9 };

// ═════════════════════════════════════════════════════════════════════════════

async function main() {
    const startTime = Date.now();

    console.log(`\n${B}${C}🧬 GoMata Digital Twin v4 — Contextual Causal Bulk Generator${X}`);
    console.log(`${C}   Target: ${(TARGET_EVENTS / 1e6).toFixed(1)}M events${X}`);
    console.log(`${C}   Cows:   ${NUM_COWS}${X}`);
    console.log(`${C}   Days:   ${SIM_DAYS} (${TOTAL_TICKS.toLocaleString()} ticks)${X}`);
    console.log(`${C}   Sample: ${(SAMPLE_RATE * 100).toFixed(1)}%${X}`);
    console.log(`${C}   Batch:  ${BATCH_SIZE.toLocaleString()}${X}`);
    console.log(`${C}   Farm:   ${FARM_TYPE}${X}`);
    console.log(`${C}   Mode:   ${DRY_RUN ? 'DRY RUN (no DB)' : 'LIVE → MongoDB (trainingevents_v4)'}${X}\n`);

    // ── Connect to MongoDB ──────────────────────────────────────────────
    let db = null, collection = null;
    if (!DRY_RUN) {
        await mongoose.connect(MONGO_URI);
        db = mongoose.connection.db;
        collection = db.collection('trainingevents_v4');
        console.log(`${G}✅ Connected to MongoDB (trainingevents_v4)${X}\n`);
    }

    // ── Tenant/Farm ─────────────────────────────────────────────────────
    const tenantId = new mongoose.Types.ObjectId();
    const farmId = new mongoose.Types.ObjectId();
    const zoneId = new mongoose.Types.ObjectId();

    // ── Create cow population ───────────────────────────────────────────
    const cowIds = [];
    const engines = new Map();
    const sensorGens = new Map();
    const prodModels = new Map();
    const cowMeta = new Map();
    const mgmtTimelines = new Map();
    const windows = new Map();

    const breeds = ['Gir', 'Sahiwal', 'Red Sindhi', 'Tharparkar', 'Holstein', 'Jersey'];
    const lactStages = ['early', 'mid', 'late', 'dry'];
    const heatTols = ['low', 'medium', 'high'];

    const mgmtSim = new ManagementEventSimulator(SIM_DAYS, TICKS_PER_DAY);

    let validationErrors = 0;
    let validationWarnings = 0;

    for (let i = 0; i < NUM_COWS; i++) {
        const cowId = `cow_${String(i).padStart(4, '0')}`;
        const age = 2 + Math.random() * 10;
        const breed = breeds[Math.floor(Math.random() * breeds.length)];
        const parity = Math.min(Math.floor(1 + Math.random() * 6), Math.floor(age - 2));
        const lactationStage = lactStages[Math.floor(Math.random() * lactStages.length)];
        const bcs = parseFloat((2.5 + Math.random() * 1.5).toFixed(1));
        const heatTolerance = heatTols[Math.floor(Math.random() * heatTols.length)];
        const baselineMilkYield = lactationStage === 'dry' ? 0 :
            (breed === 'Holstein' ? 30 + Math.random() * 10 :
                breed === 'Jersey' ? 20 + Math.random() * 8 :
                    10 + Math.random() * 8);
        const baselineWeight = breed === 'Holstein' ? 580 + Math.random() * 120 :
            breed === 'Jersey' ? 380 + Math.random() * 100 :
                320 + Math.random() * 100;

        // Days since calving (random for simulation)
        const calvingDaysAgo = lactationStage === 'dry' ? null :
            lactationStage === 'early' ? 10 + Math.random() * 80 :
                lactationStage === 'mid' ? 90 + Math.random() * 100 :
                    200 + Math.random() * 100;
        const calvingDate = calvingDaysAgo !== null
            ? new Date(Date.now() - calvingDaysAgo * 86400000)
            : null;

        const profile = {
            breed, parity, lactationStage, bodyConditionScore: bcs,
            geneticHeatTolerance: heatTolerance, previousDiseaseCount: 0,
            baselineMilkYield, baselineWeight, calvingDate, age
        };

        // Validate
        const validation = AnimalValidator.validate(profile);
        if (!validation.valid) validationErrors++;
        validationWarnings += validation.warnings.length;

        // Create engine + sensor + production + management
        const engine = new CowPhysiologyEngine(cowId, { age });
        const sensor = new SensorGenerator(engine.individualParams);
        const production = new ProductionModel(profile);
        const timeline = mgmtSim.generateTimeline();

        cowIds.push(cowId);
        engines.set(cowId, engine);
        sensorGens.set(cowId, sensor);
        prodModels.set(cowId, production);
        mgmtTimelines.set(cowId, timeline);

        cowMeta.set(cowId, {
            animalId: new mongoose.Types.ObjectId(),
            ...profile,
            baseLat: 28.6340 + (Math.random() - 0.5) * 0.01,
            baseLon: 77.1600 + (Math.random() - 0.5) * 0.01
        });

        windows.set(cowId, { temps: [], hrs: [], acts: [], rums: [] });
    }

    console.log(`${C}ℹ  Cows created: ${NUM_COWS} (${validationErrors} validation errors, ${validationWarnings} warnings)${X}`);

    // ── Schedule episodes ───────────────────────────────────────────────
    const farmProfile = FarmProfile.get(FARM_TYPE);
    const scheduler = new EpisodeScheduler({
        totalTicks: TOTAL_TICKS, tickMinutes: TICK_MINUTES,
        numCows: NUM_COWS, farmProfile
    });
    const schedule = scheduler.generateSchedule(cowIds);
    const stats = scheduler.getScheduleStats(schedule);
    console.log(`${C}ℹ  Episodes: ${stats.totalInfections} inf (${stats.infectionsPerCowPerYear}/cow/yr), ${stats.totalStressWaves} stress (${stats.stressPerCowPerYear}/cow/yr)${X}`);

    // ── Environment ─────────────────────────────────────────────────────
    const env = new EnvironmentModel();

    // ── Bulk generation loop ────────────────────────────────────────────
    let totalGenerated = 0;
    let batch = [];
    let lastReportTime = Date.now();
    let lastReportCount = 0;

    for (let tick = 1; tick <= TOTAL_TICKS; tick++) {
        const envSnap = env.getEnvironment(tick, TICK_MINUTES, NUM_COWS);
        const circT = env.getCircadianTempOffset(envSnap.hourOfDay);
        const circA = env.getCircadianActivityMultiplier(envSnap.hourOfDay);
        const baseStress = env.computeStressLoad(envSnap);
        const currentDay = tick / TICKS_PER_DAY;

        for (let c = 0; c < NUM_COWS; c++) {
            const cowId = cowIds[c];
            const engine = engines.get(cowId);
            const sensor = sensorGens.get(cowId);
            const prod = prodModels.get(cowId);
            const meta = cowMeta.get(cowId);
            const timeline = mgmtTimelines.get(cowId);
            const win = windows.get(cowId);

            // ── Episode triggers ────────────────────────────────────
            const triggered = scheduler.getTriggeredEpisodes(cowId, tick, schedule);
            for (const ep of triggered) {
                if (ep.type === 'infection' && engine.isSusceptible()) {
                    engine.seedInfection(ep);
                    // Trigger antibiotic treatment with 70% probability
                    mgmtSim.triggerAntibiotic(timeline, currentDay, 0.7);
                }
            }
            const stressBoost = scheduler.getActiveStressBoost(cowId, tick, schedule);

            // ── Management modifiers ────────────────────────────────
            const mgmt = mgmtSim.getActiveModifiers(timeline, tick);

            // ── Build v4 context for evolve() ───────────────────────
            const postpartumRisk = meta.calvingDate
                ? Math.exp(-Math.pow((currentDay * 86400000 - (Date.now() - meta.calvingDate.getTime())) / (20 * 86400000), 2))
                : 0;

            const context = {
                susceptibilityMod: (1 + 0.15 * meta.parity)
                    * (1 + 0.25 * engine.episodeCount)
                    * (1 + postpartumRisk)
                    * mgmt.susceptibilityMod,
                recoveryMod: mgmt.recoveryMod,
                stressSpike: stressBoost + mgmt.stressSpike,
                bcsCapacity: meta.bodyConditionScore / 3.0,
                heatToleranceFactor: HEAT_TOL_FACTOR[meta.geneticHeatTolerance] || 1.0,
                lactationStressFactor: LACT_STRESS_FACTOR[meta.lactationStage] || 1.0,
                vaccinationActive: mgmt.vaccinationActive,
                antibioticActive: mgmt.antibioticActive
            };

            // ── Evolve + generate sensors + production ──────────────
            const hidden = engine.evolve(baseStress, context);
            const reading = sensor.generate(hidden, envSnap, circT, circA);

            // Compute severity for production coupling
            const currentSeverity = hidden.diseaseLabel === 0 ? 0 :
                hidden.severityLevel === 'mild' ? 1 :
                    hidden.severityLevel === 'moderate' ? 2 : 3;
            const production = prod.generate(hidden, envSnap, tick, TICK_MINUTES, currentSeverity);

            // ── Update sliding window ───────────────────────────────
            win.temps.push(reading.temperature);
            win.hrs.push(reading.heartRate);
            win.acts.push(reading.activity);
            win.rums.push(reading.rumination);
            if (win.temps.length > 72) {
                win.temps.shift(); win.hrs.shift();
                win.acts.shift(); win.rums.shift();
            }

            // ── Sample ──────────────────────────────────────────────
            if (Math.random() >= SAMPLE_RATE) continue;
            if (win.temps.length < 12) continue;
            if (totalGenerated >= TARGET_EVENTS) continue;

            // ── Compute windowed features ───────────────────────────
            const tAvg = avg(win.temps), tStd = stddev(win.temps);
            const hAvg = avg(win.hrs), hStd = stddev(win.hrs);
            const aAvg = avg(win.acts), aStd = stddev(win.acts);
            const p = engine.individualParams;

            const n = win.temps.length;
            const tSlope = n > 2 ? (win.temps[n - 1] - win.temps[0]) / n : 0;
            const hSlope = n > 2 ? (win.hrs[n - 1] - win.hrs[0]) / n : 0;
            const aSlope = n > 2 ? (win.acts[n - 1] - win.acts[0]) / n : 0;

            const ruminBaseline = safe(p.ruminationBaseline, 35);
            const ruminDrop = ruminBaseline > 0 ? safe((ruminBaseline - reading.rumination) / ruminBaseline, 0) : 0;
            const stressIndex = safe(0.4 * reading.heatStressIndex + 0.3 * Math.max(0, ruminDrop) + 0.3 * (1 - reading.activity), 0);

            const gpsLat = meta.baseLat + (Math.random() - 0.5) * 0.001;
            const gpsLon = meta.baseLon + (Math.random() - 0.5) * 0.001;

            const now = new Date(Date.now() - (TOTAL_TICKS - tick) * TICK_MINUTES * 60000);
            const episodeDayIndex = engine.ticksSinceEpisodeStart > 0
                ? Math.floor(engine.ticksSinceEpisodeStart / TICKS_PER_DAY) : 0;

            // ── Build v4 document ───────────────────────────────────
            const doc = {
                tenantId, farmId, zoneId,
                animalId: meta.animalId,
                timestamp: now,
                simulationVersion: 'digital_twin_v4',
                featureVersion: 'v4_contextual',
                episodeId: hidden.episodePhase !== 'healthy'
                    ? `${cowId}_ep${hidden.episodeCount}` : null,
                episodeDayIndex,
                timeSinceEpisodeStart: engine.ticksSinceEpisodeStart * TICK_MINUTES,

                animalProfile: {
                    parity: meta.parity,
                    lactationStage: meta.lactationStage,
                    calvingDate: meta.calvingDate,
                    bodyConditionScore: meta.bodyConditionScore,
                    previousDiseaseCount: engine.episodeCount,
                    geneticHeatTolerance: meta.geneticHeatTolerance,
                    baselineMilkYield: meta.baselineMilkYield,
                    baselineWeight: meta.baselineWeight,
                    breed: meta.breed,
                    age: meta.age
                },

                signals: {
                    temperature_C: reading.temperature,
                    heartRate_bpm: reading.heartRate,
                    respiration_bpm: reading.respiration,
                    activity_index: reading.activity,
                    rumination_min: reading.rumination,
                    lying_min: reading.lying,
                    gps: { lat: gpsLat, lon: gpsLon }
                },

                production,

                environment: {
                    ambientTemp_C: envSnap.ambientTemp,
                    humidity_pct: envSnap.humidity,
                    thi: envSnap.thi,
                    ammonia_ppm: envSnap.ammonia,
                    airflow_rate: envSnap.airflow,
                    stockingDensity_raw: envSnap.stockingDensity_raw,
                    stockingDensity_normalized: envSnap.stockingDensity_normalized,
                    dayOfYear: envSnap.dayOfYear
                },

                managementEvents: {
                    vaccinationActive: mgmt.vaccinationActive,
                    antibioticActive: mgmt.antibioticActive,
                    transportActive: mgmt.transportActive,
                    feedChangeActive: mgmt.feedChangeActive,
                    groupMixingActive: mgmt.groupMixingActive,
                    ventilationBoost: mgmt.ventilationBoost
                },

                features: {
                    temp_current: reading.temperature,
                    temp_6h_avg: parseFloat(tAvg.toFixed(2)),
                    temp_6h_std: parseFloat(tStd.toFixed(3)),
                    temp_6h_slope: parseFloat(safe(tSlope, 0).toFixed(4)),
                    temp_zscore: tStd > 0 ? parseFloat(safe((reading.temperature - tAvg) / tStd, 0).toFixed(3)) : 0,
                    hr_current: reading.heartRate,
                    hr_6h_avg: parseFloat(hAvg.toFixed(1)),
                    hr_6h_std: parseFloat(hStd.toFixed(2)),
                    hr_6h_slope: parseFloat(safe(hSlope, 0).toFixed(4)),
                    hr_zscore: hStd > 0 ? parseFloat(safe((reading.heartRate - hAvg) / hStd, 0).toFixed(3)) : 0,
                    activity_current: reading.activity,
                    activity_6h_avg: parseFloat(aAvg.toFixed(3)),
                    activity_6h_std: parseFloat(aStd.toFixed(4)),
                    activity_6h_slope: parseFloat(safe(aSlope, 0).toFixed(4)),
                    activity_ratio: aAvg > 0 ? parseFloat(safe(reading.activity / aAvg, 1).toFixed(3)) : 1,
                    rumination_drop: parseFloat(safe(Math.max(0, ruminDrop), 0).toFixed(3)),
                    autocorrelation_temp: parseFloat(safe(lag1Autocorr(win.temps), 0).toFixed(4)),
                    coefficient_variation_temp: tAvg > 0 ? parseFloat(safe(tStd / tAvg, 0).toFixed(4)) : 0,
                    heat_stress_index: safe(reading.heatStressIndex, 0),
                    composite_stress_index: parseFloat(safe(stressIndex, 0).toFixed(3)),
                    heat_component: parseFloat(safe(1 / (1 + Math.exp(-(envSnap.thi - 72) / 8)), 0.5).toFixed(4)),
                    air_quality_component: parseFloat(safe(Math.max(0, Math.min(1, (envSnap.ammonia - 5) / 30)), 0).toFixed(4)),
                    crowding_component: safe(envSnap.stockingDensity_normalized, 0),
                    ventilation_component: parseFloat(safe(Math.min(1, envSnap.airflow / 3.0), 0.5).toFixed(4)),
                    // v4 production deviation features
                    milk_deviation: meta.baselineMilkYield > 0
                        ? parseFloat((production.milkYield / meta.baselineMilkYield).toFixed(3)) : 0,
                    feed_deviation: parseFloat((production.feedIntake / 18).toFixed(3)),
                    weight_deviation: parseFloat((production.bodyWeight / meta.baselineWeight).toFixed(4)),
                    conductivity_deviation: parseFloat((production.milkConductivity / 5.0).toFixed(3))
                },

                hiddenState: {
                    infectionLoad: hidden.infectionLoad,
                    stressLoad: hidden.stressLoad,
                    immuneResponse: hidden.immuneResponse,
                    compensationCapacity: hidden.compensation,
                    fatigue: hidden.fatigue,
                    compensationCollapse: !!hidden.compensationCollapse
                },

                labels: {
                    diseaseBinary: hidden.diseaseLabel,
                    infectionBinary: hidden.infectionBinary || 0,
                    stressBinary: hidden.stressBinary || 0,
                    mixedStateBinary: hidden.mixedStateBinary || 0,
                    severityLevel: hidden.diseaseLabel === 0 ? 0 :
                        hidden.severityLevel === 'mild' ? 1 :
                            hidden.severityLevel === 'moderate' ? 2 : 3,
                    episodePhase: hidden.episodePhase,
                    diseaseType: hidden.diseaseType,
                    ...scheduler.getForecastLabels(cowId, tick, schedule)
                },

                interventionContext: {
                    vaccinationActive: mgmt.vaccinationActive,
                    isolationActive: false,
                    ventilationBoost: mgmt.ventilationBoost,
                    antibioticActive: mgmt.antibioticActive
                },

                source: 'digital_twin_v4'
            };

            batch.push(doc);
            totalGenerated++;

            // ── Flush batch ─────────────────────────────────────────
            if (batch.length >= BATCH_SIZE) {
                if (!DRY_RUN) {
                    await collection.insertMany(batch, { ordered: false });
                }
                batch = [];

                const now2 = Date.now();
                if (now2 - lastReportTime > 5000) {
                    const rate = (totalGenerated - lastReportCount) / ((now2 - lastReportTime) / 1000);
                    const pct = (totalGenerated / TARGET_EVENTS * 100).toFixed(1);
                    const eta = rate > 0 ? ((TARGET_EVENTS - totalGenerated) / rate / 60).toFixed(1) : '?';

                    process.stdout.write(
                        `\r  ${G}${(totalGenerated / 1e6).toFixed(2)}M${X} / ${(TARGET_EVENTS / 1e6).toFixed(1)}M` +
                        `  ${C}${pct}%${X}  ${Y}${Math.round(rate).toLocaleString()}/s${X}` +
                        `  ETA: ${M}${eta} min${X}  tick ${tick}/${TOTAL_TICKS}   `
                    );

                    lastReportTime = now2;
                    lastReportCount = totalGenerated;
                }
            }

            if (totalGenerated >= TARGET_EVENTS) break;
        }

        if (totalGenerated >= TARGET_EVENTS) break;
    }

    // ── Flush remaining ─────────────────────────────────────────────────
    if (batch.length > 0 && !DRY_RUN) {
        await collection.insertMany(batch, { ordered: false });
    }

    const elapsed = ((Date.now() - startTime) / 1000).toFixed(1);
    const rateOverall = Math.round(totalGenerated / elapsed);

    console.log(`\n\n${B}${G}═══ V4 GENERATION COMPLETE ═══${X}`);
    console.log(`  Events:   ${B}${totalGenerated.toLocaleString()}${X}`);
    console.log(`  Duration: ${B}${elapsed}s${X} (${(elapsed / 60).toFixed(1)} min)`);
    console.log(`  Rate:     ${B}${rateOverall.toLocaleString()}/s${X}`);
    console.log(`  Cows:     ${NUM_COWS}`);
    console.log(`  Days:     ${SIM_DAYS}`);
    console.log(`  Farm:     ${FARM_TYPE}`);
    console.log(`  Version:  digital_twin_v4`);
    console.log(`  Mode:     ${DRY_RUN ? 'DRY RUN' : 'MongoDB (trainingevents_v4)'}`);

    // ── Distribution guardrail ──────────────────────────────────────────
    if (!DRY_RUN && collection) {
        console.log(`\n${C}📊 Distribution check...${X}`);
        const total = await collection.countDocuments({ source: 'digital_twin_v4' });
        if (total > 0) {
            const healthyCount = await collection.countDocuments({ source: 'digital_twin_v4', 'labels.episodePhase': 'healthy' });
            const stressCount = await collection.countDocuments({ source: 'digital_twin_v4', 'labels.episodePhase': { $in: ['stress_onset', 'stress_peak', 'stress_recovery'] } });
            const infCount = await collection.countDocuments({ source: 'digital_twin_v4', 'labels.infectionBinary': 1, 'labels.mixedStateBinary': { $ne: 1 } });
            const mixedCount = await collection.countDocuments({ source: 'digital_twin_v4', 'labels.mixedStateBinary': 1 });
            const collapseCount = await collection.countDocuments({ source: 'digital_twin_v4', 'hiddenState.compensationCollapse': true });
            const mgmtActive = await collection.countDocuments({
                source: 'digital_twin_v4', $or: [
                    { 'managementEvents.vaccinationActive': true },
                    { 'managementEvents.antibioticActive': true },
                    { 'managementEvents.transportActive': true },
                    { 'managementEvents.feedChangeActive': true }
                ]
            });

            const pct = (n) => (n / total * 100).toFixed(1);
            console.log(`    Healthy:    ${pct(healthyCount)}%  ${pct(healthyCount) >= 55 ? G + '✅' : Y + '⚠️'}${X}  target ≥55%`);
            console.log(`    Stress:     ${pct(stressCount)}%  target 15-25%`);
            console.log(`    Infection:  ${pct(infCount)}%  target 10-20%`);
            console.log(`    Mixed:      ${pct(mixedCount)}%  target 5-12%`);
            console.log(`    Collapse:   ${pct(collapseCount)}%  target <8%`);
            console.log(`    Mgmt active: ${pct(mgmtActive)}%`);
        }

        // Create indexes
        console.log(`\n${C}Creating indexes...${X}`);
        await collection.createIndex({ source: 1, timestamp: -1 });
        await collection.createIndex({ animalId: 1, timestamp: -1 });
        await collection.createIndex({ 'labels.episodePhase': 1, source: 1 });
        await collection.createIndex({ 'labels.diseaseType': 1, source: 1 });
        await collection.createIndex({ tenantId: 1, animalId: 1, timestamp: -1 });
        await collection.createIndex({ 'managementEvents.vaccinationActive': 1 });
        await collection.createIndex({ 'managementEvents.antibioticActive': 1 });
        console.log(`${G}✅ Indexes created${X}`);

        const count = await collection.countDocuments({ source: 'digital_twin_v4' });
        console.log(`\n  ${B}Total v4 events in DB: ${count.toLocaleString()}${X}`);

        await mongoose.disconnect();
    }

    console.log('');
}

main().catch(err => {
    console.error('Fatal:', err);
    process.exit(1);
});
