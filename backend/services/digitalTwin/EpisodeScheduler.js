/**
 * EpisodeScheduler — Epidemiologically Calibrated Episode Planner
 * 
 * GoMata Digital Twin Simulator v3 — Calibrated Edition
 * 
 * Uses Poisson-sampled annual incidence rates and seasonal weighting
 * to generate realistic infection and stress timelines per cow.
 * 
 * Mixed states are NOT scheduled — they emerge naturally when
 * infection overlaps with stress, handled by CowPhysiologyEngine.
 * 
 * Calibrated Targets (Dairy):
 *   Healthy:         ≥ 55%
 *   Stress only:     15–25%
 *   Infection only:  10–20%
 *   Mixed (emergent): 5–12%
 */

'use strict';

const FarmProfile = require('./FarmProfile');

// ── Constants ───────────────────────────────────────────────────────────────

const DISEASE_TYPES = ['brd', 'mastitis', 'laminitis', 'generic'];
const SEVERITY_LEVELS = ['mild', 'moderate', 'severe'];

// ── Helpers ─────────────────────────────────────────────────────────────────

function getRandomInt(min, max) {
    return Math.floor(Math.random() * (max - min + 1)) + min;
}

function weightedPick(items, weights) {
    const total = weights.reduce((s, w) => s + w, 0);
    let r = Math.random() * total;
    for (let i = 0; i < items.length; i++) {
        r -= weights[i];
        if (r <= 0) return items[i];
    }
    return items[items.length - 1];
}

/**
 * Poisson random variate (Knuth algorithm).
 * @param {number} lambda - Expected value
 * @returns {number} Poisson-distributed integer
 */
function poissonSample(lambda) {
    if (lambda <= 0) return 0;
    const L = Math.exp(-lambda);
    let k = 0;
    let p = 1;
    do {
        k++;
        p *= Math.random();
    } while (p > L);
    return k - 1;
}

/**
 * Seasonal factor: peaks in summer (day 150–250).
 * Uses sigmoid-amplified sine curve for sharp seasonal contrast.
 * f(day) = 0.2 + 0.8 * sigmoid(4 * sin(2π(day - 180) / 365))
 * Winter: ~0.2 (minimal stress), Summer: ~1.0 (peak stress)
 */
function seasonalFactor(dayOfYear) {
    const sinValue = Math.sin(2 * Math.PI * (dayOfYear - 180) / 365);
    const sigmoid = 1 / (1 + Math.exp(-4 * sinValue));
    return 0.2 + 0.8 * sigmoid;
}

// ═════════════════════════════════════════════════════════════════════════════

class EpisodeScheduler {
    /**
     * @param {Object} config
     * @param {number} config.totalTicks     - Total simulation ticks
     * @param {number} config.tickMinutes    - Minutes per tick (default 5)
     * @param {number} config.numCows        - Total number of cows
     * @param {string} [config.farmType]     - 'dairy' or 'beef' (default: 'dairy')
     * @param {Object} [config.farmProfile]  - Custom profile (overrides farmType)
     */
    constructor(config) {
        this.totalTicks = config.totalTicks;
        this.tickMinutes = config.tickMinutes || 5;
        this.numCows = config.numCows || 100;

        // Derived
        this.totalDays = (this.totalTicks * this.tickMinutes) / 1440;
        this.ticksPerDay = 1440 / this.tickMinutes; // 288 for 5-min ticks

        // Farm profile
        this.profile = config.farmProfile || FarmProfile.get(config.farmType || 'dairy');

        // Scale λ by simulation duration relative to 1 year
        this.yearScale = this.totalDays / 365;
    }

    // ─────────────────────────────────────────────────────────────────────────
    // GENERATE SCHEDULE — Create episode timeline for all cows
    // ─────────────────────────────────────────────────────────────────────────

    generateSchedule(cowIds) {
        const schedule = new Map();

        for (const cowId of cowIds) {
            const episodes = this._planCowEpisodes(cowId);
            schedule.set(cowId, episodes);
        }

        return schedule;
    }

    // ─────────────────────────────────────────────────────────────────────────
    // PER-COW EPISODE PLANNING (Poisson + Seasonal + Gap Enforcement)
    // ─────────────────────────────────────────────────────────────────────────

    _planCowEpisodes(cowId) {
        const episodes = [];
        const p = this.profile;

        // ── 1. Infection episodes (Poisson-sampled) ─────────────────
        const lambdaInf = p.lambda_infection * this.yearScale;
        const numInfections = poissonSample(lambdaInf);

        const infectionStarts = this._distributeWithGap(
            numInfections,
            p.infectionGapDays,
            this.totalDays
        );

        for (const startDay of infectionStarts) {
            const severity = weightedPick(SEVERITY_LEVELS, p.severityWeights);
            const diseaseType = weightedPick(p.diseaseTypes, p.diseaseWeights);

            const durationDays = severity === 'severe'
                ? getRandomInt(p.infectionDuration.min + 3, p.infectionDuration.max)
                : severity === 'moderate'
                    ? getRandomInt(p.infectionDuration.min + 1, p.infectionDuration.max - 2)
                    : getRandomInt(p.infectionDuration.min, p.infectionDuration.min + 3);

            const startTick = Math.floor(startDay * this.ticksPerDay);
            const estimatedDurationTicks = durationDays * this.ticksPerDay;

            episodes.push({
                type: 'infection',
                startTick,
                estimatedDurationTicks,
                diseaseType,
                targetSeverity: severity,
                cowId,
                triggered: false
            });
        }

        // ── 2. Stress waves (Poisson + Seasonal weighting) ──────────
        const lambdaStress = p.lambda_stress * this.yearScale;
        const numStressWaves = poissonSample(lambdaStress);

        const stressStarts = this._distributeSeasonalWithGap(
            numStressWaves,
            p.stressGapDays,
            this.totalDays
        );

        for (const startDay of stressStarts) {
            const durationDays = getRandomInt(p.stressDuration.min, p.stressDuration.max);
            const stressIntensity = weightedPick(
                SEVERITY_LEVELS,
                p.stressIntensityWeights
            );

            const startTick = Math.floor(startDay * this.ticksPerDay);

            episodes.push({
                type: 'heat_stress',
                startTick,
                durationTicks: durationDays * this.ticksPerDay,
                intensity: stressIntensity,
                cowId,
                triggered: false
            });
        }

        // ── Sort by start tick ───────────────────────────────────────
        episodes.sort((a, b) => a.startTick - b.startTick);

        return episodes;
    }

    // ─────────────────────────────────────────────────────────────────────────
    // GAP ENFORCEMENT — Minimum spacing between episodes
    // ─────────────────────────────────────────────────────────────────────────

    /**
     * Distribute N events across totalDays with minimum gap.
     * Uses rejection sampling to enforce gaps.
     */
    _distributeWithGap(count, gapDays, totalDays) {
        if (count === 0) return [];

        const starts = [];
        const maxAttempts = count * 20;
        let attempts = 0;

        while (starts.length < count && attempts < maxAttempts) {
            attempts++;
            const day = Math.random() * totalDays * 0.95; // avoid tail

            // Check gap from all existing
            const tooClose = starts.some(d => Math.abs(d - day) < gapDays);
            if (!tooClose) {
                starts.push(day);
            }
        }

        return starts.sort((a, b) => a - b);
    }

    /**
     * Distribute stress waves with seasonal weighting + gap enforcement.
     * Stress concentrated in summer (day 150–250).
     */
    _distributeSeasonalWithGap(count, gapDays, totalDays) {
        if (count === 0) return [];

        const starts = [];
        const maxAttempts = count * 50;
        let attempts = 0;

        while (starts.length < count && attempts < maxAttempts) {
            attempts++;

            // Sample candidate day uniformly
            const candidateDay = Math.random() * totalDays * 0.95;
            const dayInYear = candidateDay % 365;

            // Accept/reject based on seasonal probability
            const pAccept = seasonalFactor(dayInYear);
            if (Math.random() > pAccept) continue;

            // Check gap
            const tooClose = starts.some(d => Math.abs(d - candidateDay) < gapDays);
            if (!tooClose) {
                starts.push(candidateDay);
            }
        }

        return starts.sort((a, b) => a - b);
    }

    // ─────────────────────────────────────────────────────────────────────────
    // CHECK TICK — What episodes should trigger at this tick?
    // ─────────────────────────────────────────────────────────────────────────

    getTriggeredEpisodes(cowId, currentTick, schedule) {
        const cowEpisodes = schedule.get(cowId);
        if (!cowEpisodes) return [];

        const triggered = [];

        for (const ep of cowEpisodes) {
            if (ep.triggered) continue;
            if (ep.startTick <= currentTick) {
                ep.triggered = true;
                triggered.push(ep);
            }
        }

        return triggered;
    }

    /**
     * Check if a stress wave is active for a cow at the given tick.
     */
    getActiveStressBoost(cowId, currentTick, schedule) {
        const cowEpisodes = schedule.get(cowId);
        if (!cowEpisodes) return 0;

        let maxBoost = 0;

        for (const ep of cowEpisodes) {
            if (ep.type !== 'heat_stress') continue;
            if (currentTick >= ep.startTick && currentTick < ep.startTick + ep.durationTicks) {
                const intensityMap = { mild: 0.15, moderate: 0.30, severe: 0.50 };
                const boost = intensityMap[ep.intensity] || 0.20;

                // Bell-curve within episode: peak at center
                const progress = (currentTick - ep.startTick) / ep.durationTicks;
                const envelope = Math.sin(Math.PI * progress); // 0→1→0
                maxBoost = Math.max(maxBoost, boost * envelope);
            }
        }

        return maxBoost;
    }

    /**
     * Compute forecast labels from schedule (leakage-safe).
     */
    getForecastLabels(cowId, currentTick, schedule) {
        const cowEpisodes = schedule.get(cowId);
        if (!cowEpisodes) return { infection_in_24h: 0, stress_in_24h: 0 };

        const lookAheadTicks = Math.floor(1440 / this.tickMinutes); // 24h
        const futureEnd = currentTick + lookAheadTicks;

        let infection_in_24h = 0;
        let stress_in_24h = 0;

        for (const ep of cowEpisodes) {
            if (ep.startTick > currentTick && ep.startTick <= futureEnd) {
                if (ep.type === 'infection') infection_in_24h = 1;
                if (ep.type === 'heat_stress') stress_in_24h = 1;
            }
            if (ep.type === 'heat_stress' && ep.durationTicks) {
                const epEnd = ep.startTick + ep.durationTicks;
                if (ep.startTick <= futureEnd && epEnd > currentTick) {
                    stress_in_24h = 1;
                }
            }
            if (ep.type === 'infection' && ep.estimatedDurationTicks) {
                const epEnd = ep.startTick + ep.estimatedDurationTicks;
                if (ep.startTick <= futureEnd && epEnd > currentTick) {
                    infection_in_24h = 1;
                }
            }
        }

        return { infection_in_24h, stress_in_24h };
    }

    // ─────────────────────────────────────────────────────────────────────────
    // BATCH SCENARIO GENERATION
    // ─────────────────────────────────────────────────────────────────────────

    generateOutbreakScenarios(numScenarios = 1000, opts = {}) {
        const scenarios = [];
        const herdMin = opts.herdSizeMin || 50;
        const herdMax = opts.herdSizeMax || 500;

        for (let i = 0; i < numScenarios; i++) {
            const herdSize = getRandomInt(herdMin, herdMax);
            const vaccinationCoverage = weightedPick(
                [0, 0.3, 0.6, 0.8],
                [0.25, 0.25, 0.25, 0.25]
            );
            const initialInfected = Math.max(1, Math.floor(herdSize * (0.01 + Math.random() * 0.04)));
            const r0 = 1.5 + Math.random() * 2.5;

            scenarios.push({
                scenarioId: i,
                herdSize,
                vaccinationCoverage,
                initialInfected,
                r0,
                isolationStrategy: weightedPick(
                    ['none', 'immediate', 'delayed_24h', 'delayed_48h'],
                    [0.3, 0.3, 0.2, 0.2]
                ),
                penDensity: weightedPick(['low', 'medium', 'high'], [0.3, 0.4, 0.3]),
                durationDays: getRandomInt(30, 90)
            });
        }

        return scenarios;
    }

    // ─────────────────────────────────────────────────────────────────────────
    // SCHEDULE STATS
    // ─────────────────────────────────────────────────────────────────────────

    getScheduleStats(schedule) {
        let totalInfections = 0;
        let totalStressWaves = 0;
        const severityCounts = { mild: 0, moderate: 0, severe: 0 };

        for (const [, episodes] of schedule) {
            for (const ep of episodes) {
                if (ep.type === 'infection') {
                    totalInfections++;
                    severityCounts[ep.targetSeverity]++;
                } else {
                    totalStressWaves++;
                }
            }
        }

        return {
            totalInfections,
            totalStressWaves,
            averageInfectionsPerCow: (totalInfections / schedule.size).toFixed(2),
            averageStressPerCow: (totalStressWaves / schedule.size).toFixed(2),
            infectionsPerCowPerYear: (totalInfections / schedule.size / (this.totalDays / 365)).toFixed(2),
            stressPerCowPerYear: (totalStressWaves / schedule.size / (this.totalDays / 365)).toFixed(2),
            severityCounts,
            farmType: this.profile.farmType
        };
    }
}

module.exports = EpisodeScheduler;
