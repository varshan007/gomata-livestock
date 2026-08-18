#!/usr/bin/env node
/**
 * GoMata Bulk Dataset Generator
 * 
 * Generates millions of training events offline using the Digital Twin
 * physics engine in a tight loop (no timers). Writes directly to MongoDB
 * with bulk inserts.
 * 
 * Usage:
 *   node scripts/generateBulkDataset.js                    # Default: 10M events
 *   node scripts/generateBulkDataset.js --target 1000000   # 1M events
 *   node scripts/generateBulkDataset.js --cows 200 --days 365
 *   node scripts/generateBulkDataset.js --dry-run           # No DB, just count
 * 
 * Estimated: 10M events in ~2-4 hours depending on hardware.
 */

'use strict';

const mongoose = require('mongoose');
const path = require('path');

// ── Engine imports ──────────────────────────────────────────────────────────
const EnvironmentModel = require('../services/digitalTwin/EnvironmentModel');
const CowPhysiologyEngine = require('../services/digitalTwin/CowPhysiologyEngine');
const SensorGenerator = require('../services/digitalTwin/SensorGenerator');
const EpisodeScheduler = require('../services/digitalTwin/EpisodeScheduler');
const FarmProfile = require('../services/digitalTwin/FarmProfile');

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
const TICK_MINUTES = 5;
const TICKS_PER_DAY = 1440 / TICK_MINUTES; // 288
const TOTAL_TICKS = SIM_DAYS * TICKS_PER_DAY;
const DRY_RUN = hasFlag('dry-run');
const FARM_TYPE = getArg('farm-type', 'dairy');
const MONGO_URI = process.env.MONGO_URI || 'mongodb://127.0.0.1:27017/livestock_monitoring';

// Sampling rate: auto-compute to hit target
// Total possible = NUM_COWS × TOTAL_TICKS
const totalPossible = NUM_COWS * TOTAL_TICKS;
const SAMPLE_RATE = Math.min(1.0, TARGET_EVENTS / totalPossible);

// ── Color helpers ───────────────────────────────────────────────────────────
const G = '\x1b[32m', C = '\x1b[36m', Y = '\x1b[33m', B = '\x1b[1m', X = '\x1b[0m', M = '\x1b[35m';

// ── Stats helpers ───────────────────────────────────────────────────────────
const gaussNoise = (std) => {
    let u = 0, v = 0;
    while (u === 0) u = Math.random();
    while (v === 0) v = Math.random();
    return std * Math.sqrt(-2.0 * Math.log(u)) * Math.cos(2.0 * Math.PI * v);
};

const avg = arr => arr.length > 0 ? arr.reduce((s, v) => s + v, 0) / arr.length : 0;
const stddev = arr => {
    if (arr.length < 2) return 0;
    const m = avg(arr);
    return Math.sqrt(arr.reduce((s, v) => s + (v - m) ** 2, 0) / (arr.length - 1));
};

const lag1Autocorr = arr => {
    if (arr.length < 3) return 0;
    const m = avg(arr);
    let num = 0, den = 0;
    for (let i = 1; i < arr.length; i++) num += (arr[i] - m) * (arr[i - 1] - m);
    for (let i = 0; i < arr.length; i++) den += (arr[i] - m) ** 2;
    return den > 0 ? num / den : 0;
};

const safe = (v, fallback = 0) => isFinite(v) ? v : fallback;

// ═════════════════════════════════════════════════════════════════════════════
//  MAIN
// ═════════════════════════════════════════════════════════════════════════════

async function main() {
    const startTime = Date.now();

    console.log(`\n${B}${C}🧬 GoMata Bulk Dataset Generator${X}`);
    console.log(`${C}   Target:  ${(TARGET_EVENTS / 1e6).toFixed(1)}M events${X}`);
    console.log(`${C}   Cows:    ${NUM_COWS}${X}`);
    console.log(`${C}   Days:    ${SIM_DAYS}${X}`);
    console.log(`${C}   Ticks:   ${TOTAL_TICKS.toLocaleString()} per cow${X}`);
    console.log(`${C}   Sample:  ${(SAMPLE_RATE * 100).toFixed(1)}%${X}`);
    console.log(`${C}   Est:     ${Math.round(totalPossible * SAMPLE_RATE).toLocaleString()} events${X}`);
    console.log(`${C}   Batch:   ${BATCH_SIZE.toLocaleString()}${X}`);
    console.log(`${C}   Mode:    ${DRY_RUN ? 'DRY RUN (no DB)' : 'LIVE → MongoDB'}${X}`);
    console.log(`${C}   Farm:    ${FARM_TYPE}${X}\n`);

    // ── Connect to MongoDB ──────────────────────────────────────────────
    let db = null;
    let collection = null;

    if (!DRY_RUN) {
        await mongoose.connect(MONGO_URI);
        db = mongoose.connection.db;
        collection = db.collection('trainingevents');
        console.log(`${G}✅ Connected to MongoDB${X}\n`);
    }

    // ── Generate farms/tenants ───────────────────────────────────────────
    const tenantId = new mongoose.Types.ObjectId();
    const farmId = new mongoose.Types.ObjectId();
    const zoneId = new mongoose.Types.ObjectId();

    // ── Create cow population ───────────────────────────────────────────
    const cowIds = [];
    const engines = new Map();
    const sensorGens = new Map();
    const cowMeta = new Map();
    const windows = new Map(); // sliding windows per cow

    const breeds = ['Gir', 'Sahiwal', 'Red Sindhi', 'Tharparkar', 'Holstein', 'Jersey'];
    const lactStages = ['early', 'mid', 'late', 'dry'];
    const heatTols = ['low', 'medium', 'high'];

    for (let i = 0; i < NUM_COWS; i++) {
        const cowId = `cow_${String(i).padStart(4, '0')}`;
        const age = 2 + Math.random() * 10;
        const engine = new CowPhysiologyEngine(cowId, { age });
        const sensor = new SensorGenerator(engine.individualParams);

        cowIds.push(cowId);
        engines.set(cowId, engine);
        sensorGens.set(cowId, sensor);

        cowMeta.set(cowId, {
            animalId: new mongoose.Types.ObjectId(),
            breed: breeds[Math.floor(Math.random() * breeds.length)],
            parity: Math.floor(1 + Math.random() * 6),
            lactationStage: lactStages[Math.floor(Math.random() * lactStages.length)],
            bcs: parseFloat((2.5 + Math.random() * 1.5).toFixed(1)),
            heatTolerance: heatTols[Math.floor(Math.random() * heatTols.length)],
            baseLat: 28.6340 + (Math.random() - 0.5) * 0.01,
            baseLon: 77.1600 + (Math.random() - 0.5) * 0.01
        });

        windows.set(cowId, { temps: [], hrs: [], acts: [], rums: [] });
    }

    // ── Schedule episodes ───────────────────────────────────────────────
    const farmProfile = FarmProfile.get(FARM_TYPE);
    const scheduler = new EpisodeScheduler({
        totalTicks: TOTAL_TICKS,
        tickMinutes: TICK_MINUTES,
        numCows: NUM_COWS,
        farmProfile
    });
    const schedule = scheduler.generateSchedule(cowIds);
    const stats = scheduler.getScheduleStats(schedule);
    console.log(`${C}ℹ  Farm profile: ${stats.farmType}${X}`);
    console.log(`${C}ℹ  Episodes: ${stats.totalInfections} infections (${stats.infectionsPerCowPerYear}/cow/yr), ${stats.totalStressWaves} stress (${stats.stressPerCowPerYear}/cow/yr)${X}`);

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

        for (let c = 0; c < NUM_COWS; c++) {
            const cowId = cowIds[c];
            const engine = engines.get(cowId);
            const sensor = sensorGens.get(cowId);
            const meta = cowMeta.get(cowId);
            const win = windows.get(cowId);

            // ── Episode triggers ────────────────────────────────────
            const triggered = scheduler.getTriggeredEpisodes(cowId, tick, schedule);
            for (const ep of triggered) {
                if (ep.type === 'infection' && engine.isSusceptible()) {
                    engine.seedInfection(ep);
                }
            }
            const stressBoost = scheduler.getActiveStressBoost(cowId, tick, schedule);
            const totalStress = baseStress + stressBoost;

            // ── Evolve + generate sensors ───────────────────────────
            const hidden = engine.evolve(totalStress);
            const reading = sensor.generate(hidden, envSnap, circT, circA);

            // ── Update sliding window (72 ticks = 6 hours) ──────────
            win.temps.push(reading.temperature);
            win.hrs.push(reading.heartRate);
            win.acts.push(reading.activity);
            win.rums.push(reading.rumination);
            if (win.temps.length > 72) {
                win.temps.shift(); win.hrs.shift();
                win.acts.shift(); win.rums.shift();
            }

            // ── Sample for training event ───────────────────────────
            if (Math.random() >= SAMPLE_RATE) continue;
            if (win.temps.length < 12) continue;
            if (totalGenerated >= TARGET_EVENTS) continue;

            // ── Compute windowed features ───────────────────────────
            const tAvg = avg(win.temps);
            const tStd = stddev(win.temps);
            const hAvg = avg(win.hrs);
            const hStd = stddev(win.hrs);
            const aAvg = avg(win.acts);
            const aStd = stddev(win.acts);
            const p = engine.individualParams;

            // Slopes
            const n = win.temps.length;
            const tSlope = n > 2 ? (win.temps[n - 1] - win.temps[0]) / n : 0;
            const hSlope = n > 2 ? (win.hrs[n - 1] - win.hrs[0]) / n : 0;
            const aSlope = n > 2 ? (win.acts[n - 1] - win.acts[0]) / n : 0;

            const ruminBaseline = safe(p.ruminationBaseline, 35);
            const ruminDrop = ruminBaseline > 0 ? safe((ruminBaseline - reading.rumination) / ruminBaseline, 0) : 0;
            const stressIndex = safe(0.4 * reading.heatStressIndex + 0.3 * Math.max(0, ruminDrop) + 0.3 * (1 - reading.activity), 0);

            // GPS with jitter
            const gpsLat = meta.baseLat + (Math.random() - 0.5) * 0.001;
            const gpsLon = meta.baseLon + (Math.random() - 0.5) * 0.001;

            // ── Build document ──────────────────────────────────────
            const now = new Date(Date.now() - (TOTAL_TICKS - tick) * TICK_MINUTES * 60000);

            const ticksPerDay = 1440 / TICK_MINUTES;
            const episodeDayIndex = engine.ticksSinceEpisodeStart > 0
                ? Math.floor(engine.ticksSinceEpisodeStart / ticksPerDay)
                : 0;

            const doc = {
                tenantId,
                farmId,
                zoneId,
                animalId: meta.animalId,
                timestamp: now,
                simulationVersion: 'digital_twin_v3',
                featureVersion: 'v4_windowed',
                episodeId: hidden.episodePhase !== 'healthy'
                    ? `${cowId}_ep${hidden.episodeCount}`
                    : null,
                episodeDayIndex,
                timeSinceEpisodeStart: engine.ticksSinceEpisodeStart * TICK_MINUTES,

                animalProfile: {
                    parity: meta.parity,
                    lactationStage: meta.lactationStage,
                    bodyConditionScore: meta.bcs,
                    geneticHeatTolerance: meta.heatTolerance,
                    previousDiseaseCount: hidden.episodeCount
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

                environment: {
                    ambientTemp_C: envSnap.ambientTemp,
                    humidity_pct: envSnap.humidity,
                    thi: envSnap.thi,
                    ammonia_ppm: envSnap.ammonia,
                    airflow_rate: envSnap.airflow,
                    stocking_density_raw: envSnap.stockingDensity_raw,
                    stocking_density_normalized: envSnap.stockingDensity_normalized
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
                    // Raw stress components (let ML learn weighting)
                    heat_component: parseFloat(safe(1 / (1 + Math.exp(-(envSnap.thi - 72) / 8)), 0.5).toFixed(4)),
                    air_quality_component: parseFloat(safe(Math.max(0, Math.min(1, (envSnap.ammonia - 5) / 30)), 0).toFixed(4)),
                    crowding_component: safe(envSnap.stockingDensity_normalized, 0),
                    ventilation_component: parseFloat(safe(Math.min(1, envSnap.airflow / 3.0), 0.5).toFixed(4))
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
                    // Forecast (from schedule only, NOT hidden state)
                    ...scheduler.getForecastLabels(cowId, tick, schedule)
                },

                interventionContext: {
                    vaccinationActive: false,
                    isolationActive: false,
                    ventilationBoost: false,
                    antibioticActive: false
                },

                source: 'digital_twin_v3'
            };

            batch.push(doc);
            totalGenerated++;

            // ── Flush batch ─────────────────────────────────────────
            if (batch.length >= BATCH_SIZE) {
                if (!DRY_RUN) {
                    await collection.insertMany(batch, { ordered: false });
                }
                batch = [];

                // Progress report every 5 seconds
                const now2 = Date.now();
                if (now2 - lastReportTime > 5000) {
                    const elapsed = (now2 - startTime) / 1000;
                    const rate = (totalGenerated - lastReportCount) / ((now2 - lastReportTime) / 1000);
                    const pct = (totalGenerated / TARGET_EVENTS * 100).toFixed(1);
                    const eta = rate > 0 ? ((TARGET_EVENTS - totalGenerated) / rate / 60).toFixed(1) : '?';

                    process.stdout.write(
                        `\r  ${G}${(totalGenerated / 1e6).toFixed(2)}M${X} / ${(TARGET_EVENTS / 1e6).toFixed(1)}M` +
                        `  ${C}${pct}%${X}` +
                        `  ${Y}${Math.round(rate).toLocaleString()}/s${X}` +
                        `  ETA: ${M}${eta} min${X}` +
                        `  tick ${tick}/${TOTAL_TICKS}   `
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
    const rateOverall = Math.round(totalGenerated / (elapsed));

    console.log(`\n\n${B}${G}═══ GENERATION COMPLETE ═══${X}`);
    console.log(`  Events:   ${B}${totalGenerated.toLocaleString()}${X}`);
    console.log(`  Duration: ${B}${elapsed}s${X} (${(elapsed / 60).toFixed(1)} min)`);
    console.log(`  Rate:     ${B}${rateOverall.toLocaleString()}/s${X}`);
    console.log(`  Cows:     ${NUM_COWS}`);
    console.log(`  Days:     ${SIM_DAYS}`);
    console.log(`  Farm:     ${FARM_TYPE}`);
    console.log(`  Mode:     ${DRY_RUN ? 'DRY RUN' : 'MongoDB'}`);

    // ── Distribution guardrail ──────────────────────────────────────────
    if (!DRY_RUN && collection) {
        console.log(`\n${C}📊 Distribution check...${X}`);
        const total = await collection.countDocuments({ source: 'digital_twin_v3' });
        if (total > 0) {
            const healthyCount = await collection.countDocuments({ source: 'digital_twin_v3', 'labels.episodePhase': 'healthy' });
            const stressCount = await collection.countDocuments({ source: 'digital_twin_v3', 'labels.episodePhase': { $in: ['stress_onset', 'stress_peak', 'stress_recovery'] } });
            const infCount = await collection.countDocuments({ source: 'digital_twin_v3', 'labels.infectionBinary': 1, 'labels.mixedStateBinary': { $ne: 1 } });
            const mixedCount = await collection.countDocuments({ source: 'digital_twin_v3', 'labels.mixedStateBinary': 1 });
            const collapseCount = await collection.countDocuments({ source: 'digital_twin_v3', 'hiddenState.compensationCollapse': true });

            const healthyPct = (healthyCount / total * 100).toFixed(1);
            const stressPct = (stressCount / total * 100).toFixed(1);
            const infPct = (infCount / total * 100).toFixed(1);
            const mixedPct = (mixedCount / total * 100).toFixed(1);
            const collapsePct = (collapseCount / total * 100).toFixed(1);

            console.log(`    Healthy:    ${healthyPct}%  ${healthyPct >= 55 ? G + '✅' : Y + '⚠️'}${X}  target ≥55%`);
            console.log(`    Stress:     ${stressPct}%  ${stressPct >= 15 && stressPct <= 25 ? G + '✅' : Y + '⚠️'}${X}  target 15-25%`);
            console.log(`    Infection:  ${infPct}%  ${infPct >= 10 && infPct <= 20 ? G + '✅' : Y + '⚠️'}${X}  target 10-20%`);
            console.log(`    Mixed:      ${mixedPct}%  ${mixedPct >= 5 && mixedPct <= 12 ? G + '✅' : Y + '⚠️'}${X}  target 5-12%`);
            console.log(`    Collapse:   ${collapsePct}%  ${collapsePct < 8 ? G + '✅' : Y + '⚠️'}${X}  target <8%`);
        }

        // Create indexes
        console.log(`\n${C}Creating indexes...${X}`);
        await collection.createIndex({ source: 1, timestamp: -1 });
        await collection.createIndex({ animalId: 1, timestamp: -1 });
        await collection.createIndex({ 'labels.episodePhase': 1, source: 1 });
        await collection.createIndex({ 'labels.diseaseType': 1, source: 1 });
        await collection.createIndex({ tenantId: 1, animalId: 1, timestamp: -1 });
        console.log(`${G}✅ Indexes created${X}`);

        const count = await collection.countDocuments({ source: 'digital_twin_v3' });
        console.log(`\n  ${B}Total v3 events in DB: ${count.toLocaleString()}${X}`);

        await mongoose.disconnect();
    }

    console.log('');
}

main().catch(err => {
    console.error('Fatal:', err);
    process.exit(1);
});
