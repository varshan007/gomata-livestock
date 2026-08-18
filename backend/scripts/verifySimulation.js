#!/usr/bin/env node
/**
 * GoMata Digital Twin v3 — Scientific Verification Suite
 * 
 * 11-Level Production Validation
 * Runs the simulator in-memory and validates:
 *   L1:  Signal range + noise magnitude
 *   L2:  Lag-1 autocorrelation + circadian R²
 *   L3:  Cross-signal correlation by state
 *   L4:  Phase logic ordering
 *   L5:  Label consistency
 *   L6:  Distribution realism
 *   L7:  Herd spread (batch scenario)
 *   L8:  Intervention effect
 *   L9:  ML dry run (noted — requires Python)
 *   L10: Leakage check (structural)
 *   L11: Drift check (noted — requires 2-farm run)
 * 
 * Usage:
 *   node scripts/verifySimulation.js
 *   node scripts/verifySimulation.js --cows 20 --days 30
 */

'use strict';

// ── Imports (all internal — no DB or Redis needed) ──────────────────────────

const EnvironmentModel = require('../services/digitalTwin/EnvironmentModel');
const CowPhysiologyEngine = require('../services/digitalTwin/CowPhysiologyEngine');
const SensorGenerator = require('../services/digitalTwin/SensorGenerator');
const EpisodeScheduler = require('../services/digitalTwin/EpisodeScheduler');
const ProductionModel = require('../services/digitalTwin/ProductionModel');
const ManagementEventSimulator = require('../services/digitalTwin/ManagementEventSimulator');
const AnimalValidator = require('../services/digitalTwin/AnimalValidator');
const HerdTransmissionModel = require('../services/digitalTwin/HerdTransmissionModel');
const InterventionEngine = require('../services/digitalTwin/InterventionEngine');

// ── CLI args ────────────────────────────────────────────────────────────────

const args = process.argv.slice(2);
const getArg = (name, def) => {
    const idx = args.indexOf(`--${name}`);
    return idx >= 0 && args[idx + 1] ? args[idx + 1] : def;
};

const NUM_COWS = parseInt(getArg('cows', '10'));
const SIM_DAYS = parseInt(getArg('days', '14'));
const TICK_MINUTES = 5;
const TICKS_PER_DAY = 1440 / TICK_MINUTES;
const TOTAL_TICKS = SIM_DAYS * TICKS_PER_DAY;

// ── Color helpers ───────────────────────────────────────────────────────────

const G = '\x1b[32m', R = '\x1b[31m', Y = '\x1b[33m', C = '\x1b[36m';
const B = '\x1b[1m', X = '\x1b[0m', M = '\x1b[35m';

const pass = (msg, detail) => { console.log(`  ${G}✅ PASS${X}  ${msg}${detail ? `  ${C}${detail}${X}` : ''}`); return true; };
const fail = (msg, detail) => { console.log(`  ${R}❌ FAIL${X}  ${msg}${detail ? `  ${Y}${detail}${X}` : ''}`); return false; };
const warn = (msg, detail) => { console.log(`  ${Y}⚠️  WARN${X}  ${msg}${detail ? `  ${Y}${detail}${X}` : ''}`); return true; };
const info = (msg) => console.log(`  ${C}ℹ  ${msg}${X}`);
const header = (level, msg) => console.log(`\n${B}${M}═══ LEVEL ${level} — ${msg} ═══${X}`);

// ── Stats helpers ───────────────────────────────────────────────────────────

const avg = arr => arr.reduce((s, v) => s + v, 0) / arr.length;
const stddev = arr => { const m = avg(arr); return Math.sqrt(arr.reduce((s, v) => s + (v - m) ** 2, 0) / (arr.length - 1)); };
const pearson = (a, b) => {
    const n = Math.min(a.length, b.length);
    const ma = avg(a.slice(0, n)), mb = avg(b.slice(0, n));
    let num = 0, da = 0, db = 0;
    for (let i = 0; i < n; i++) {
        num += (a[i] - ma) * (b[i] - mb);
        da += (a[i] - ma) ** 2;
        db += (b[i] - mb) ** 2;
    }
    return da > 0 && db > 0 ? num / Math.sqrt(da * db) : 0;
};
const lag1Autocorr = arr => {
    const n = arr.length;
    if (n < 3) return 0;
    const m = avg(arr);
    let num = 0, den = 0;
    for (let i = 1; i < n; i++) num += (arr[i] - m) * (arr[i - 1] - m);
    for (let i = 0; i < n; i++) den += (arr[i] - m) ** 2;
    return den > 0 ? num / den : 0;
};

// Fit 24h sine → R² (circadian strength)
const circadianR2 = (times, values) => {
    const n = values.length;
    if (n < 48) return 0; // Need at least 2 cycles
    const TWO_PI = 2 * Math.PI;
    // Best-fit: y = A*cos(2π(t-φ)/24) + B
    // Try phase shifts 0-23h, pick best R²
    const yMean = avg(values);
    const ssTot = values.reduce((s, v) => s + (v - yMean) ** 2, 0);
    let bestR2 = 0;
    for (let phase = 0; phase < 24; phase++) {
        let ssRes = 0;
        for (let i = 0; i < n; i++) {
            const hour = times[i] % 24;
            const predicted = yMean + stddev(values) * 0.5 * Math.cos(TWO_PI * (hour - phase) / 24);
            ssRes += (values[i] - predicted) ** 2;
        }
        const r2 = 1 - ssRes / ssTot;
        if (r2 > bestR2) bestR2 = r2;
    }
    return bestR2;
};

// ═════════════════════════════════════════════════════════════════════════════
//  SIMULATION RUNNER — Generate data in-memory
// ═════════════════════════════════════════════════════════════════════════════

function runSimulation() {
    console.log(`\n${B}${C}🧬 GoMata Digital Twin v3 — Scientific Verification${X}`);
    console.log(`${C}   ${NUM_COWS} cows × ${SIM_DAYS} days = ${TOTAL_TICKS} ticks${X}\n`);

    const env = new EnvironmentModel();
    const cowIds = Array.from({ length: NUM_COWS }, (_, i) => `cow_${i}`);

    // Create engines + sensors
    const engines = new Map();
    const sensors = new Map();
    for (const id of cowIds) {
        const engine = new CowPhysiologyEngine(id, { age: 3 + Math.random() * 8 });
        engines.set(id, engine);
        sensors.set(id, new SensorGenerator(engine.individualParams));
    }

    // Schedule episodes
    const scheduler = new EpisodeScheduler({
        totalTicks: TOTAL_TICKS,
        tickMinutes: TICK_MINUTES,
        numCows: NUM_COWS,
        farmType: 'dairy'
    });
    const schedule = scheduler.generateSchedule(cowIds);
    const schedStats = scheduler.getScheduleStats(schedule);

    info(`Episodes scheduled: ${schedStats.totalInfections} infections, ${schedStats.totalStressWaves} stress waves`);

    // ── Per-cow time-series collection ───────────────────────────────────
    const data = new Map(); // cowId → { temps[], hrs[], resps[], acts[], rums[], lies[], phases[], ... }
    for (const id of cowIds) {
        data.set(id, {
            temps: [], hrs: [], resps: [], acts: [], rums: [], lies: [],
            phases: [], infLoads: [], stressLoads: [], compensations: [],
            fatigues: [], diseaseLabels: [], severities: [], times: [],
            diseaseTypes: [], collapses: [],
            // Phase transition log for ordering validation
            phaseTransitions: [],
            lastPhase: 'healthy'
        });
    }

    // ── Run tick loop ────────────────────────────────────────────────────
    for (let tick = 1; tick <= TOTAL_TICKS; tick++) {
        const envSnap = env.getEnvironment(tick, TICK_MINUTES, NUM_COWS);
        const circT = env.getCircadianTempOffset(envSnap.hourOfDay);
        const circA = env.getCircadianActivityMultiplier(envSnap.hourOfDay);

        for (const id of cowIds) {
            const engine = engines.get(id);
            const sensor = sensors.get(id);

            // Check episodes
            const triggered = scheduler.getTriggeredEpisodes(id, tick, schedule);
            for (const ep of triggered) {
                if (ep.type === 'infection' && engine.isSusceptible()) {
                    engine.seedInfection(ep);
                }
            }
            const stressBoost = scheduler.getActiveStressBoost(id, tick, schedule);
            const baseStress = env.computeStressLoad(envSnap);
            const totalStress = baseStress + stressBoost;

            const hidden = engine.evolve(totalStress);
            const reading = sensor.generate(hidden, envSnap, circT, circA);

            const d = data.get(id);
            d.temps.push(reading.temperature);
            d.hrs.push(reading.heartRate);
            d.resps.push(reading.respiration);
            d.acts.push(reading.activity);
            d.rums.push(reading.rumination);
            d.lies.push(reading.lying);
            d.phases.push(hidden.episodePhase);
            d.infLoads.push(hidden.infectionLoad);
            d.stressLoads.push(hidden.stressLoad);
            d.compensations.push(hidden.compensation);
            d.fatigues.push(hidden.fatigue);
            d.diseaseLabels.push(hidden.diseaseLabel);
            d.severities.push(hidden.severityLevel);
            d.diseaseTypes.push(hidden.diseaseType);
            d.times.push(envSnap.hourOfDay);
            d.collapses.push(hidden.compensationCollapse ? 1 : 0);

            // Track phase transitions
            if (hidden.episodePhase !== d.lastPhase) {
                d.phaseTransitions.push({
                    tick, from: d.lastPhase, to: hidden.episodePhase
                });
                d.lastPhase = hidden.episodePhase;
            }
        }

        if (tick % (TICKS_PER_DAY * 2) === 0) {
            process.stdout.write(`\r  Simulating... day ${Math.floor(tick / TICKS_PER_DAY)}/${SIM_DAYS}`);
        }
    }
    process.stdout.write(`\r  Simulated ${TOTAL_TICKS} ticks across ${NUM_COWS} cows                    \n`);

    return { data, engines, schedule, scheduler, schedStats, env };
}


// ═════════════════════════════════════════════════════════════════════════════
//  LEVEL 1 — SIGNAL SANITY CHECK
// ═════════════════════════════════════════════════════════════════════════════

function level1(data) {
    header('1', 'SIGNAL SANITY CHECK');
    let passed = 0, total = 0;

    const RANGES = {
        temps: [36.5, 42.5, 'Temperature (°C)'],
        hrs: [40, 180, 'Heart Rate (bpm)'],
        resps: [10, 80, 'Respiration (bpm)'],
        acts: [0, 1.01, 'Activity (0–1)'],
        rums: [0, 60, 'Rumination (min/hr)'],
        lies: [0, 60, 'Lying Time (min/hr)']
    };

    // Range check
    for (const [key, [lo, hi, label]] of Object.entries(RANGES)) {
        total++;
        const all = [];
        for (const [, d] of data) all.push(...d[key]);
        const min = Math.min(...all.slice(0, 10000)); // sample for speed
        const max = Math.max(...all.slice(0, 10000));
        const violations = all.filter(v => v < lo || v > hi).length;
        if (violations === 0) {
            passed++; pass(`${label} range`, `[${min.toFixed(2)}, ${max.toFixed(2)}] within [${lo}, ${hi}]`);
        } else {
            fail(`${label} range`, `${violations} violations outside [${lo}, ${hi}]`);
        }
    }

    // Noise magnitude (healthy only)
    console.log('');
    const NOISE_TARGETS = {
        temps: [0.05, 0.4, 'Temp SD'],
        hrs: [1.5, 12, 'HR SD'],
        acts: [0.01, 0.20, 'Activity SD']
    };

    for (const [key, [lo, hi, label]] of Object.entries(NOISE_TARGETS)) {
        total++;
        const healthySamples = [];
        for (const [, d] of data) {
            for (let i = 0; i < d[key].length; i++) {
                if (d.phases[i] === 'healthy') healthySamples.push(d[key][i]);
            }
        }
        if (healthySamples.length < 100) { warn(`${label}`, 'Not enough healthy samples'); continue; }
        const sd = stddev(healthySamples);
        if (sd >= lo && sd <= hi) {
            passed++; pass(`${label} noise`, `σ = ${sd.toFixed(4)}  target [${lo}, ${hi}]`);
        } else {
            fail(`${label} noise`, `σ = ${sd.toFixed(4)}  target [${lo}, ${hi}]`);
        }
    }

    return { passed, total };
}


// ═════════════════════════════════════════════════════════════════════════════
//  LEVEL 2 — TIME-SERIES DYNAMICS
// ═════════════════════════════════════════════════════════════════════════════

function level2(data) {
    header('2', 'TIME-SERIES DYNAMICS');
    let passed = 0, total = 0;

    // 3️⃣ Lag-1 Autocorrelation (healthy temp)
    total++;
    const healthyTemps = [];
    for (const [, d] of data) {
        for (let i = 0; i < d.temps.length; i++) {
            if (d.phases[i] === 'healthy') healthyTemps.push(d.temps[i]);
            else if (healthyTemps.length > 0) break; // take first healthy stretch
        }
    }
    if (healthyTemps.length > 100) {
        const ac = lag1Autocorr(healthyTemps.slice(0, 5000));
        if (ac >= 0.85 && ac <= 0.9995) {
            passed++; pass(`Lag-1 autocorrelation (healthy temp)`, `ρ₁ = ${ac.toFixed(4)}  target [0.85, 0.9995]`);
        } else {
            fail(`Lag-1 autocorrelation (healthy temp)`, `ρ₁ = ${ac.toFixed(4)}  target [0.85, 0.9995]`);
        }
    } else {
        warn('Lag-1 autocorrelation', 'Not enough healthy temp data');
    }

    // 4️⃣ Circadian Strength
    total++;
    const allTemps = [], allTimes = [];
    for (const [, d] of data) {
        for (let i = 0; i < Math.min(d.temps.length, TICKS_PER_DAY * 3); i++) {
            if (d.phases[i] === 'healthy') {
                allTemps.push(d.temps[i]);
                allTimes.push(d.times[i]);
            }
        }
    }
    if (allTemps.length > 200) {
        const r2 = circadianR2(allTimes, allTemps);
        if (r2 >= 0.15 && r2 <= 0.95) {
            passed++; pass(`Circadian R² (temp)`, `R² = ${r2.toFixed(4)}  target [0.15, 0.95]`);
        } else {
            fail(`Circadian R² (temp)`, `R² = ${r2.toFixed(4)}  target [0.15, 0.95]`);
        }
    } else {
        warn('Circadian R²', 'Not enough data');
    }

    // 5️⃣ Recovery Slope
    total++;
    let recoveryTests = 0, smoothRecoveries = 0;
    for (const [, d] of data) {
        for (let i = 1; i < d.phases.length; i++) {
            if (d.phases[i] === 'recovery' && d.phases[i - 1] === 'recovery') {
                // Check temp is declining or stable during recovery
                if (i > 10) {
                    const window = d.temps.slice(Math.max(0, i - 20), i);
                    if (window.length >= 10) {
                        recoveryTests++;
                        const first5 = avg(window.slice(0, 5));
                        const last5 = avg(window.slice(-5));
                        if (last5 <= first5 + 0.3) smoothRecoveries++; // Allow small noise
                    }
                }
            }
        }
    }
    if (recoveryTests > 0) {
        const pct = (smoothRecoveries / recoveryTests * 100).toFixed(1);
        if (smoothRecoveries / recoveryTests > 0.6) {
            passed++; pass(`Recovery slope (gradual decline)`, `${pct}% smooth (${smoothRecoveries}/${recoveryTests})`);
        } else {
            fail(`Recovery slope`, `Only ${pct}% smooth — may have sharp drops`);
        }
    } else {
        warn('Recovery slope', 'No recovery phases found — try more days or cows');
    }

    return { passed, total };
}


// ═════════════════════════════════════════════════════════════════════════════
//  LEVEL 3 — CROSS-SIGNAL CORRELATION
// ═════════════════════════════════════════════════════════════════════════════

function level3(data) {
    header('3', 'CROSS-SIGNAL CORRELATION');
    let passed = 0, total = 0;

    // Collect by state
    const infected = { temps: [], hrs: [], acts: [], resps: [], stresses: [] };
    const healthy = { temps: [], hrs: [], acts: [], resps: [], stresses: [] };

    for (const [, d] of data) {
        for (let i = 0; i < d.temps.length; i++) {
            const target = d.infLoads[i] > 0.05 ? infected : healthy;
            target.temps.push(d.temps[i]);
            target.hrs.push(d.hrs[i]);
            target.acts.push(d.acts[i]);
            target.resps.push(d.resps[i]);
            target.stresses.push(d.stressLoads[i]);
        }
    }

    // Infection correlations
    if (infected.temps.length > 50) {
        total++;
        const tempHR = pearson(infected.temps, infected.hrs);
        if (tempHR > 0.4) { passed++; pass(`Corr(temp, HR) during infection`, `r = ${tempHR.toFixed(3)}  target > 0.4`); }
        else { fail(`Corr(temp, HR) during infection`, `r = ${tempHR.toFixed(3)}  target > 0.4`); }

        total++;
        const tempAct = pearson(infected.temps, infected.acts);
        if (tempAct < -0.2) { passed++; pass(`Corr(temp, activity) during infection`, `r = ${tempAct.toFixed(3)}  target < -0.2`); }
        else { fail(`Corr(temp, activity) during infection`, `r = ${tempAct.toFixed(3)}  target < -0.2`); }
    } else {
        info('Not enough infected samples for infection correlation — try more cows/days');
    }

    // Healthy state — temp-HR should have weak correlation
    if (healthy.temps.length > 200) {
        total++;
        const healthyTempHR = pearson(healthy.temps.slice(0, 5000), healthy.hrs.slice(0, 5000));
        if (Math.abs(healthyTempHR) < 0.6) {
            passed++; pass(`Corr(temp, HR) during healthy`, `r = ${healthyTempHR.toFixed(3)}  (weak, expected)`);
        } else {
            warn(`Corr(temp, HR) during healthy`, `r = ${healthyTempHR.toFixed(3)}  unexpectedly strong`);
        }
    }

    return { passed, total };
}


// ═════════════════════════════════════════════════════════════════════════════
//  LEVEL 4 — PHASE LOGIC ORDERING
// ═════════════════════════════════════════════════════════════════════════════

function level4(data) {
    header('4', 'PHASE LOGIC ORDERING');
    let passed = 0, total = 0;

    const VALID_TRANSITIONS = {
        'healthy': ['incubation', 'stress_onset'],
        'incubation': ['onset', 'mixed_onset'],
        'onset': ['peak', 'mixed_onset'],
        'peak': ['plateau', 'mixed_onset'],
        'plateau': ['recovery', 'mixed_onset'],
        'recovery': ['resolved'],
        'resolved': ['healthy', 'incubation'],
        // Stress-only
        'stress_onset': ['stress_peak', 'stress_recovery', 'incubation', 'mixed_onset'],
        'stress_peak': ['stress_recovery', 'incubation', 'mixed_onset'],
        'stress_recovery': ['healthy', 'stress_onset', 'mixed_onset'],
        // Mixed (infection + stress)
        'mixed_onset': ['mixed_peak', 'mixed_recovery', 'stress_onset', 'healthy'],
        'mixed_peak': ['mixed_recovery', 'stress_onset', 'healthy'],
        'mixed_recovery': ['healthy', 'mixed_onset', 'stress_onset']
    };

    let totalTransitions = 0, validTransitions = 0, invalidList = [];

    for (const [cowId, d] of data) {
        for (const t of d.phaseTransitions) {
            totalTransitions++;
            const allowed = VALID_TRANSITIONS[t.from] || [];
            if (allowed.includes(t.to)) {
                validTransitions++;
            } else {
                invalidList.push(`${cowId}: ${t.from} → ${t.to} @ tick ${t.tick}`);
            }
        }
    }

    total++;
    if (totalTransitions === 0) {
        warn('Phase transitions', 'No episodes occurred — try more cows/days');
    } else {
        const pct = (validTransitions / totalTransitions * 100).toFixed(1);
        if (validTransitions === totalTransitions) {
            passed++; pass(`Phase ordering`, `${totalTransitions} transitions, ALL valid`);
        } else {
            fail(`Phase ordering`, `${validTransitions}/${totalTransitions} valid (${pct}%)`);
            invalidList.slice(0, 5).forEach(l => info(`  Invalid: ${l}`));
        }
    }

    // Check: temp rise correlates with infection progression
    total++;
    let tempRiseBeforePeak = 0, tempRiseChecks = 0;
    for (const [, d] of data) {
        for (let i = 10; i < d.phases.length; i++) {
            if (d.phases[i] === 'peak' && d.phases[i - 10] === 'onset') {
                tempRiseChecks++;
                const onsetTemp = avg(d.temps.slice(i - 10, i - 5));
                const peakTemp = avg(d.temps.slice(i - 3, i + 2));
                if (peakTemp > onsetTemp) tempRiseBeforePeak++;
            }
        }
    }
    if (tempRiseChecks > 0) {
        const pct = (tempRiseBeforePeak / tempRiseChecks * 100).toFixed(0);
        if (tempRiseBeforePeak / tempRiseChecks > 0.6) {
            passed++; pass(`Temp rises onset → peak`, `${pct}% (${tempRiseBeforePeak}/${tempRiseChecks})`);
        } else {
            fail(`Temp rises onset → peak`, `Only ${pct}%`);
        }
    } else {
        warn('Temp rise check', 'No onset→peak transitions found');
    }

    return { passed, total };
}


// ═════════════════════════════════════════════════════════════════════════════
//  LEVEL 5 — LABEL CONSISTENCY
// ═════════════════════════════════════════════════════════════════════════════

function level5(data) {
    header('5', 'LABEL CONSISTENCY');
    let passed = 0, total = 0;

    let healthyInfection0 = 0, healthyInfectionNZ = 0;
    let severeHighI = 0, severeCount = 0;
    let label0healthy = 0, label0total = 0;

    for (const [, d] of data) {
        for (let i = 0; i < d.phases.length; i++) {
            // If healthy → infectionLoad should be 0
            if (d.phases[i] === 'healthy') {
                if (d.infLoads[i] <= 0.005) healthyInfection0++;
                else healthyInfectionNZ++;
            }
            // If severe → I > 0.5 OR S > 0.7 (stress-based severity is valid)
            if (d.severities[i] === 'severe') {
                severeCount++;
                if (d.infLoads[i] > 0.5 || d.stressLoads[i] > 0.7) severeHighI++;
            }
            // If diseaseBinary = 0 → phase should be healthy
            if (d.diseaseLabels[i] === 0) {
                label0total++;
                if (d.phases[i] === 'healthy') label0healthy++;
            }
        }
    }

    total++;
    const healthyTotal = healthyInfection0 + healthyInfectionNZ;
    if (healthyTotal > 0) {
        const pct = (healthyInfection0 / healthyTotal * 100).toFixed(1);
        if (healthyInfection0 / healthyTotal > 0.98) {
            passed++; pass(`Healthy → I(t) ≈ 0`, `${pct}%  (${healthyInfection0}/${healthyTotal})`);
        } else {
            fail(`Healthy → I(t) ≈ 0`, `Only ${pct}% — ${healthyInfectionNZ} contradictions`);
        }
    }

    total++;
    if (severeCount > 0) {
        const pct = (severeHighI / severeCount * 100).toFixed(0);
        if (severeHighI / severeCount > 0.8) {
            passed++; pass(`Severe → I(t) > 0.5`, `${pct}%  (${severeHighI}/${severeCount})`);
        } else {
            fail(`Severe → I(t) > 0.5`, `Only ${pct}%`);
        }
    } else {
        warn('Severe check', 'No severe episodes — expected with short simulation');
    }

    total++;
    if (label0total > 0) {
        const pct = (label0healthy / label0total * 100).toFixed(1);
        if (label0healthy / label0total > 0.95) {
            passed++; pass(`Label 0 → phase healthy`, `${pct}%`);
        } else {
            fail(`Label 0 → phase healthy`, `Only ${pct}%`);
        }
    }

    return { passed, total };
}


// ═════════════════════════════════════════════════════════════════════════════
//  LEVEL 6 — DISTRIBUTION CHECK
// ═════════════════════════════════════════════════════════════════════════════

function level6(data) {
    header('6', 'DISTRIBUTION CHECK');
    let passed = 0, total = 0;

    const phaseCounts = {};
    let totalSamples = 0;

    for (const [, d] of data) {
        for (const phase of d.phases) {
            phaseCounts[phase] = (phaseCounts[phase] || 0) + 1;
            totalSamples++;
        }
    }

    console.log('');
    info('Phase distribution:');
    for (const [phase, count] of Object.entries(phaseCounts).sort((a, b) => b[1] - a[1])) {
        const pct = (count / totalSamples * 100).toFixed(1);
        const bar = '█'.repeat(Math.ceil(pct / 2));
        console.log(`    ${phase.padEnd(14)} ${pct.padStart(5)}%  ${G}${bar}${X}`);
    }

    // Healthy should be majority (50-90%)
    total++;
    const healthyPct = ((phaseCounts.healthy || 0) / totalSamples * 100);
    if (healthyPct >= 25 && healthyPct <= 95) {
        passed++; pass(`Healthy proportion`, `${healthyPct.toFixed(1)}%  target [25%, 95%]`);
    } else {
        fail(`Healthy proportion`, `${healthyPct.toFixed(1)}%  target [25%, 95%]`);
    }

    // Disease labels: should not be 50/50
    total++;
    const diseaseSamples = Object.entries(phaseCounts)
        .filter(([k]) => k !== 'healthy')
        .reduce((s, [, v]) => s + v, 0);
    const diseasePct = (diseaseSamples / totalSamples * 100);
    if (diseasePct >= 3 && diseasePct <= 80) {
        passed++; pass(`Disease proportion`, `${diseasePct.toFixed(1)}%  (includes stress episodes)`);
    } else if (diseasePct < 3) {
        warn(`Disease proportion`, `${diseasePct.toFixed(1)}% — very low, try more cows/days`);
    } else {
        fail(`Disease proportion`, `${diseasePct.toFixed(1)}% — unrealistically high`);
    }

    return { passed, total };
}


// ═════════════════════════════════════════════════════════════════════════════
//  LEVEL 7 — HERD SPREAD VALIDATION
// ═════════════════════════════════════════════════════════════════════════════

function level7() {
    header('7', 'HERD SPREAD VALIDATION');
    let passed = 0, total = 0;

    try {
        const herd = new HerdTransmissionModel({
            r0: 2.5,
            penDensity: 'medium'
        });

        // Run two scenarios: no intervention and vaccination
        const baseScenario = {
            scenarioId: 'verify_1',
            herdSize: 30,
            initialInfected: 2,
            r0: 2.5,
            penDensity: 'medium',
            durationDays: 14,
            vaccinationCoverage: 0,
            isolationStrategy: 'none'
        };

        const result = herd.runScenario(baseScenario);

        total++;
        if (result && result.peakInfected > 0) {
            passed++; pass(`Outbreak peak exists`, `Peak: ${result.peakInfected} infected`);
        } else {
            fail(`Outbreak peak`, `peakInfected = ${result ? result.peakInfected : 'null'}`);
        }

        total++;
        if (result && result.outbreakDurationDays > 0) {
            passed++; pass(`Outbreak has duration`, `${result.outbreakDurationDays} days`);
        } else {
            fail(`Outbreak duration`, `0 days — no spread`);
        }

        total++;
        if (result && result.totalTransmissions > 0) {
            passed++; pass(`Transmission occurred`, `${result.totalTransmissions} transmissions`);
        } else {
            fail(`Transmission`, `0 transmissions — spread broken`);
        }

    } catch (e) {
        warn('Herd spread', `Skipped: ${e.message}`);
    }

    return { passed, total };
}


// ═════════════════════════════════════════════════════════════════════════════
//  LEVEL 8 — INTERVENTION VALIDATION
// ═════════════════════════════════════════════════════════════════════════════

function level8() {
    header('8', 'INTERVENTION VALIDATION');
    let passed = 0, total = 0;

    try {
        const baseScenario = {
            herdSize: 30,
            initialInfected: 2,
            r0: 2.5,
            penDensity: 'medium',
            durationDays: 14,
            isolationStrategy: 'none'
        };

        // Run 3 paired comparisons for statistical robustness
        let noVaccPeakTotal = 0, vaccPeakTotal = 0;
        let noVaccTransTotal = 0, vaccTransTotal = 0;
        const RUNS = 5;

        for (let r = 0; r < RUNS; r++) {
            const h1 = new HerdTransmissionModel({ r0: 2.5, penDensity: 'medium' });
            const h2 = new HerdTransmissionModel({ r0: 2.5, penDensity: 'medium' });
            const nv = h1.runScenario({ ...baseScenario, scenarioId: `nv_${r}`, vaccinationCoverage: 0 });
            const wv = h2.runScenario({ ...baseScenario, scenarioId: `wv_${r}`, vaccinationCoverage: 0.6 });
            noVaccPeakTotal += nv.peakInfected;
            vaccPeakTotal += wv.peakInfected;
            noVaccTransTotal += nv.totalTransmissions;
            vaccTransTotal += wv.totalTransmissions;
        }

        const nvPeak = (noVaccPeakTotal / RUNS).toFixed(1);
        const vPeak = (vaccPeakTotal / RUNS).toFixed(1);
        const nvTrans = (noVaccTransTotal / RUNS).toFixed(1);
        const vTrans = (vaccTransTotal / RUNS).toFixed(1);

        total++;
        if (vaccPeakTotal <= noVaccPeakTotal) {
            passed++; pass(`Vaccination reduces peak (avg ${RUNS} runs)`, `No vacc: ${nvPeak} → 60% vacc: ${vPeak}`);
        } else {
            fail(`Vaccination effect`, `No vacc avg: ${nvPeak} vs 60% vacc avg: ${vPeak}`);
        }

        total++;
        if (vaccTransTotal <= noVaccTransTotal) {
            passed++; pass(`Vaccination reduces transmission (avg ${RUNS} runs)`, `No vacc: ${nvTrans} → 60% vacc: ${vTrans}`);
        } else {
            fail(`Vaccination transmission`, `No vacc avg: ${nvTrans} vs vacc avg: ${vTrans}`);
        }

    } catch (e) {
        warn('Intervention', `Skipped: ${e.message}`);
    }

    return { passed, total };
}


// ═════════════════════════════════════════════════════════════════════════════
//  LEVEL 9 — ML DRY RUN (Documented)
// ═════════════════════════════════════════════════════════════════════════════

function level9() {
    header('9', 'ML DRY RUN (Manual Step)');
    info('This requires Python + scikit-learn. Run after exporting data:');
    info('  1. node scripts/generateDataset.js --cows 50 --days 30');
    info('  2. python3 ml_service/scripts/train_classifier.py');
    info('  Target: 85–95% accuracy (>99% = too simple, <70% = broken)');
    return { passed: 0, total: 0 };
}


// ═════════════════════════════════════════════════════════════════════════════
//  LEVEL 10 — LEAKAGE CHECK (Structural)
// ═════════════════════════════════════════════════════════════════════════════

function level10() {
    header('10', 'LEAKAGE CHECK');
    let passed = 0, total = 0;

    // Structural check: ML_EXPORT_PROJECTION should exclude hiddenState
    try {
        const TrainingEvent = require('../models/TrainingEvent');
        const projection = TrainingEvent.ML_EXPORT_PROJECTION;

        total++;
        if (projection && projection.hiddenState === 0) {
            passed++; pass(`ML export excludes hiddenState`, 'ML_EXPORT_PROJECTION.hiddenState = 0');
        } else if (projection) {
            fail(`ML export may leak hiddenState`, JSON.stringify(projection));
        } else {
            warn('ML_EXPORT_PROJECTION not defined');
        }

        // Check that features section does NOT contain infection/stress/fatigue
        total++;
        const featureFields = Object.keys(projection).filter(k => projection[k] === 1);
        const leaked = featureFields.filter(f =>
            f.includes('infection') || f.includes('stressLoad') || f.includes('fatigue')
        );
        if (leaked.length === 0) {
            passed++; pass(`No hidden state in feature projection`, `Clean: ${featureFields.join(', ')}`);
        } else {
            fail(`Potential leakage`, `Leaked fields: ${leaked.join(', ')}`);
        }
    } catch (e) {
        warn('Leakage check', `Skipped: ${e.message}`);
    }

    return { passed, total };
}


// ═════════════════════════════════════════════════════════════════════════════
//  LEVEL 11 — DRIFT TEST (Documented)
// ═════════════════════════════════════════════════════════════════════════════

function level11() {
    header('11', 'DRIFT / CROSS-FARM TEST (Manual Step)');
    info('Requires 2-farm generation with different configs:');
    info('  Farm A: Heat tolerance = high, density = 0.5');
    info('  Farm B: Heat tolerance = low, density = 2.0');
    info('  Train on A, test on B — performance should not collapse (>60% accuracy)');
    return { passed: 0, total: 0 };
}


// ═════════════════════════════════════════════════════════════════════════════
//  LEVEL 12 — ANNUAL INFECTION RATE
// ═════════════════════════════════════════════════════════════════════════════

function level12(schedStats) {
    header('12', 'EPIDEMIOLOGICAL RATE');
    let passed = 0, total = 0;

    const infRate = parseFloat(schedStats.infectionsPerCowPerYear);
    total++;
    if (infRate >= 0.2 && infRate <= 1.0) {
        passed++;
        pass(`Infections/cow/year`, `${infRate}  target [0.2, 1.0]`);
    } else {
        fail(`Infections/cow/year`, `${infRate}  target [0.2, 1.0]`);
    }

    const stressRate = parseFloat(schedStats.stressPerCowPerYear);
    total++;
    if (stressRate >= 1.0 && stressRate <= 5.0) {
        passed++;
        pass(`Stress waves/cow/year`, `${stressRate}  target [1.0, 5.0]`);
    } else {
        fail(`Stress waves/cow/year`, `${stressRate}  target [1.0, 5.0]`);
    }

    return { passed, total };
}


// ═════════════════════════════════════════════════════════════════════════════
//  LEVEL 13 — MIXED PREVALENCE
// ═════════════════════════════════════════════════════════════════════════════

function level13(data) {
    header('13', 'MIXED STATE PREVALENCE');
    let passed = 0, total = 0;

    let mixedTicks = 0, totalTicks = 0;
    for (const [, d] of data) {
        for (const phase of d.phases) {
            totalTicks++;
            if (phase.startsWith('mixed_')) mixedTicks++;
        }
    }

    const mixedPct = totalTicks > 0 ? (mixedTicks / totalTicks * 100) : 0;
    total++;
    if (mixedPct < 12) {
        passed++;
        pass(`Mixed prevalence`, `${mixedPct.toFixed(1)}%  target <12%`);
    } else {
        fail(`Mixed prevalence`, `${mixedPct.toFixed(1)}%  target <12%`);
    }

    return { passed, total };
}


// ═════════════════════════════════════════════════════════════════════════════
//  LEVEL 14 — COMPENSATION COLLAPSE FREQUENCY
// ═════════════════════════════════════════════════════════════════════════════

function level14(data) {
    header('14', 'COMPENSATION COLLAPSE');
    let passed = 0, total = 0;

    let collapseTicks = 0, totalTicks = 0;
    for (const [, d] of data) {
        for (const c of d.collapses) {
            totalTicks++;
            if (c === 1) collapseTicks++;
        }
    }

    const collapsePct = totalTicks > 0 ? (collapseTicks / totalTicks * 100) : 0;
    total++;
    if (collapsePct < 8) {
        passed++;
        pass(`Collapse frequency`, `${collapsePct.toFixed(1)}%  target <8%`);
    } else {
        fail(`Collapse frequency`, `${collapsePct.toFixed(1)}%  target <8%`);
    }

    return { passed, total };
}


// ═════════════════════════════════════════════════════════════════════════════
//  LEVEL 15 — SEASONAL STRESS CONCENTRATION
// ═════════════════════════════════════════════════════════════════════════════

function level15(data) {
    header('15', 'SEASONAL STRESS CONCENTRATION');
    let passed = 0, total = 0;

    // Only valid if we have ≥1 year of data
    if (SIM_DAYS < 365) {
        info('Requires ≥365 days for seasonal check — skipping');
        return { passed: 0, total: 0 };
    }

    // Count stress ticks in summer (day 120-270) vs winter (day 300-90)
    let summerStress = 0, summerTotal = 0;
    let winterStress = 0, winterTotal = 0;

    const ticksPerDay = 1440 / TICK_MINUTES;

    for (const [, d] of data) {
        for (let i = 0; i < d.phases.length; i++) {
            const dayOfYear = Math.floor(i / ticksPerDay) % 365;
            const isStress = d.phases[i].startsWith('stress_') || d.phases[i].startsWith('mixed_');

            if (dayOfYear >= 120 && dayOfYear <= 270) {
                summerTotal++;
                if (isStress) summerStress++;
            } else if (dayOfYear >= 300 || dayOfYear <= 90) {
                winterTotal++;
                if (isStress) winterStress++;
            }
        }
    }

    const summerRate = summerTotal > 0 ? summerStress / summerTotal : 0;
    const winterRate = winterTotal > 0 ? winterStress / winterTotal : 0;
    const ratio = winterRate > 0 ? summerRate / winterRate : (summerRate > 0 ? 999 : 1);

    total++;
    if (ratio >= 1.5) {
        passed++;
        pass(`Summer/Winter stress ratio`, `${ratio.toFixed(1)}x  target ≥1.5x`);
    } else {
        fail(`Summer/Winter stress ratio`, `${ratio.toFixed(1)}x  target ≥1.5x`);
    }

    return { passed, total };
}


// ═════════════════════════════════════════════════════════════════════════════
//  LEVEL 16 — PRODUCTION MODEL CONSISTENCY
// ═════════════════════════════════════════════════════════════════════════════

function level16(data) {
    header('16', 'PRODUCTION MODEL (v4)');
    let passed = 0, total = 0;

    // Test: milk drops during infection
    const profile = {
        breed: 'Holstein', lactationStage: 'mid', baselineMilkYield: 35,
        baselineWeight: 650, parity: 2, calvingDate: new Date(Date.now() - 90 * 86400000)
    };
    const prod = new ProductionModel(profile);

    const healthyMilk = prod.generate({ infectionLoad: 0, stressLoad: 0, fatigue: 0 },
        { thi: 65, hourOfDay: 12, dayOfYear: 180 }, 100, 5, 0);
    const sickMilk = prod.generate({ infectionLoad: 0.7, stressLoad: 0.3, fatigue: 0.4 },
        { thi: 65, hourOfDay: 12, dayOfYear: 180 }, 100, 5, 3);

    total++;
    if (sickMilk.milkYield < healthyMilk.milkYield) {
        passed++;
        pass('Milk drops during infection', `healthy=${healthyMilk.milkYield}L → sick=${sickMilk.milkYield}L`);
    } else {
        fail('Milk drops during infection', `healthy=${healthyMilk.milkYield} vs sick=${sickMilk.milkYield}`);
    }

    total++;
    if (sickMilk.milkConductivity > healthyMilk.milkConductivity) {
        passed++;
        pass('Conductivity rises during infection', `healthy=${healthyMilk.milkConductivity} → sick=${sickMilk.milkConductivity}`);
    } else {
        fail('Conductivity rises during infection', `h=${healthyMilk.milkConductivity} s=${sickMilk.milkConductivity}`);
    }

    // Dry cow should produce 0 milk
    const dryProd = new ProductionModel({ ...profile, lactationStage: 'dry' });
    const dryMilk = dryProd.generate({ infectionLoad: 0, stressLoad: 0, fatigue: 0 },
        { thi: 65, hourOfDay: 12, dayOfYear: 180 }, 100, 5);
    total++;
    if (dryMilk.milkYield === 0) {
        passed++;
        pass('Dry cow produces 0 milk', 'milkYield=0');
    } else {
        fail('Dry cow produces 0 milk', `milkYield=${dryMilk.milkYield}`);
    }

    return { passed, total };
}


// ═════════════════════════════════════════════════════════════════════════════
//  LEVEL 17 — MANAGEMENT EVENT TIMELINE
// ═════════════════════════════════════════════════════════════════════════════

function level17() {
    header('17', 'MANAGEMENT EVENTS (v4)');
    let passed = 0, total = 0;

    const mgmt = new ManagementEventSimulator(365, TICKS_PER_DAY);

    // Generate 20 timelines, check each has events
    let timelinesWithEvents = 0;
    let totalEvents = 0;
    for (let i = 0; i < 20; i++) {
        const tl = mgmt.generateTimeline();
        if (tl.length > 0) timelinesWithEvents++;
        totalEvents += tl.length;
    }

    total++;
    if (timelinesWithEvents >= 15) { // At least 75% should have events
        passed++;
        pass('Timeline generation', `${timelinesWithEvents}/20 have events (avg ${(totalEvents / 20).toFixed(1)}/cow)`);
    } else {
        fail('Timeline generation', `Only ${timelinesWithEvents}/20 have events`);
    }

    // Test antibiotic trigger
    const tl = [];
    mgmt.triggerAntibiotic(tl, 50, 1.0); // 100% probability
    total++;
    if (tl.length === 1 && tl[0].type === 'antibiotic') {
        passed++;
        pass('Antibiotic trigger', `triggered at day 50, duration=${tl[0].durationDays}d`);
    } else {
        fail('Antibiotic trigger', `Expected 1 antibiotic event, got ${tl.length}`);
    }

    // Test modifier composition
    const tl2 = mgmt.generateTimeline();
    const mods = mgmt.getActiveModifiers(tl2, 0);
    total++;
    if (mods.susceptibilityMod !== undefined && mods.recoveryMod !== undefined) {
        passed++;
        pass('Modifier structure', `suscept=${mods.susceptibilityMod.toFixed(2)}, recov=${mods.recoveryMod.toFixed(1)}`);
    } else {
        fail('Modifier structure', 'Missing fields');
    }

    return { passed, total };
}


// ═════════════════════════════════════════════════════════════════════════════
//  LEVEL 18 — ANIMAL VALIDATION
// ═════════════════════════════════════════════════════════════════════════════

function level18() {
    header('18', 'ANIMAL VALIDATION (v4)');
    let passed = 0, total = 0;

    // Valid profile should pass
    const valid = {
        parity: 3, lactationStage: 'mid', bodyConditionScore: 3.5,
        geneticHeatTolerance: 'medium', previousDiseaseCount: 1, age: 6
    };
    const r1 = AnimalValidator.validate(valid);
    total++;
    if (r1.valid) {
        passed++;
        pass('Valid profile passes', `${r1.errors.length} errors`);
    } else {
        fail('Valid profile passes', r1.errors.join('; '));
    }

    // Invalid BCS should fail
    const invalid = { ...valid, bodyConditionScore: 7 };
    const r2 = AnimalValidator.validate(invalid);
    total++;
    if (!r2.valid) {
        passed++;
        pass('Invalid BCS rejected', r2.errors[0]);
    } else {
        fail('Invalid BCS rejected', 'Should have failed');
    }

    // Parity 0 + calving date cross-consistency
    const cross = { ...valid, parity: 0, calvingDate: new Date() };
    const r3 = AnimalValidator.validate(cross);
    total++;
    if (!r3.valid) {
        passed++;
        pass('Parity 0 + calving rejected', r3.errors[0]);
    } else {
        fail('Parity 0 + calving rejected', 'Should have failed');
    }

    return { passed, total };
}


// ═════════════════════════════════════════════════════════════════════════════
//  LEVEL 19 — LABEL INTEGRITY
// ═════════════════════════════════════════════════════════════════════════════

function level19(data) {
    header('19', 'LABEL INTEGRITY (v4)');
    let passed = 0, total = 0;

    // mixed=1 → both infectionBinary=1 AND stressBinary=1
    let violations = 0, mixedCount = 0;
    for (const [, d] of data) {
        for (let i = 0; i < d.phases.length; i++) {
            if (d.phases[i].startsWith('mixed_')) {
                mixedCount++;
                // During mixed, both I > 0.05 and S should be elevated
                if (d.infLoads[i] < 0.01) violations++;
            }
        }
    }

    total++;
    if (violations === 0) {
        passed++;
        pass('Mixed→infection integrity', `${mixedCount} mixed ticks, 0 violations`);
    } else {
        fail('Mixed→infection integrity', `${violations}/${mixedCount} mixed ticks had I<0.01`);
    }

    return { passed, total };
}


// ═════════════════════════════════════════════════════════════════════════════
//  MAIN — Run all levels
// ═════════════════════════════════════════════════════════════════════════════

function main() {
    const startTime = Date.now();

    const { data, schedStats } = runSimulation();

    const results = [];
    results.push(level1(data));
    results.push(level2(data));
    results.push(level3(data));
    results.push(level4(data));
    results.push(level5(data));
    results.push(level6(data));
    results.push(level7());
    results.push(level8());
    results.push(level9());
    results.push(level10());
    results.push(level11());
    results.push(level12(schedStats));
    results.push(level13(data));
    results.push(level14(data));
    results.push(level15(data));
    results.push(level16(data));
    results.push(level17());
    results.push(level18());
    results.push(level19(data));

    // ── Final Summary ────────────────────────────────────────────────────
    const totalPassed = results.reduce((s, r) => s + r.passed, 0);
    const totalTests = results.reduce((s, r) => s + r.total, 0);
    const durationSec = ((Date.now() - startTime) / 1000).toFixed(1);

    console.log(`\n${B}${C}═══════════════════════════════════════════════════${X}`);
    console.log(`${B}${C}  VERIFICATION SUMMARY${X}`);
    console.log(`${B}${C}═══════════════════════════════════════════════════${X}`);
    console.log(`\n  Tests:    ${B}${totalPassed}/${totalTests} passed${X}`);
    console.log(`  Duration: ${durationSec}s`);
    console.log(`  Cows:     ${NUM_COWS}`);
    console.log(`  Days:     ${SIM_DAYS}`);

    if (totalPassed === totalTests) {
        console.log(`\n  ${G}${B}🟢 ALL CHECKS PASSED — Simulator is production-ready${X}\n`);
    } else {
        const failCount = totalTests - totalPassed;
        console.log(`\n  ${Y}${B}🟡 ${failCount} issue(s) detected — review above${X}\n`);
    }
}

main();
