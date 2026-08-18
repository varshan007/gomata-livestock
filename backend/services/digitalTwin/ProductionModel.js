/**
 * ProductionModel — Simulates production signals from hidden state + animal profile
 * 
 * GoMata Digital Twin v4 — Contextual Causal Twin
 * 
 * All production outputs are deterministic functions of latent state + animal config.
 * This ensures zero label leakage: production is an EFFECT of hidden state, not a cause.
 * 
 * Equations:
 *   Milk(t)  = baseMilk × (1-0.75I^1.3) × (1-0.35S) × (1-0.2F) × (1-0.12×sev) × LC(t)
 *   LC(t)    = a × t^b × exp(-c×t)  (Wood's lactation curve)
 *   Conductivity = baseline × (1 + 0.4I)  (mastitis indicator)
 *   Weight   = baseWeight × (1 - 0.05I - 0.03S) + noise
 *   FeedIntake = baseFeed × (1 - 0.3I - 0.2S) × circadianFeed
 *   WaterIntake = baseWater × (1 + 0.3×HSI) × (1 - 0.25I)
 */

'use strict';

// ── Default baselines by breed/type ─────────────────────────────────────────

const BREED_DEFAULTS = {
    'Holstein': { milk: 35, weight: 650, feed: 22, water: 100 },
    'Jersey': { milk: 25, weight: 450, feed: 18, water: 80 },
    'Gir': { milk: 12, weight: 400, feed: 16, water: 70 },
    'Sahiwal': { milk: 15, weight: 450, feed: 17, water: 75 },
    'Red Sindhi': { milk: 10, weight: 350, feed: 15, water: 65 },
    'Tharparkar': { milk: 11, weight: 380, feed: 15, water: 65 },
    'default': { milk: 18, weight: 450, feed: 18, water: 80 }
};

// ── Helpers ─────────────────────────────────────────────────────────────────

function gaussNoise(std) {
    let u = 0, v = 0;
    while (u === 0) u = Math.random();
    while (v === 0) v = Math.random();
    return std * Math.sqrt(-2.0 * Math.log(u)) * Math.cos(2.0 * Math.PI * v);
}

const safe = (v, fallback = 0) => isFinite(v) ? v : fallback;
const clamp = (v, min, max) => Math.max(min, Math.min(max, v));

// ═════════════════════════════════════════════════════════════════════════════

class ProductionModel {
    /**
     * @param {Object} animalProfile
     * @param {string} animalProfile.breed
     * @param {string} animalProfile.lactationStage - early|mid|late|dry
     * @param {number} animalProfile.baselineMilkYield - L/day (overrides breed)
     * @param {number} animalProfile.baselineWeight - kg
     * @param {number} animalProfile.parity
     * @param {Date|null} animalProfile.calvingDate
     */
    constructor(animalProfile) {
        const breed = animalProfile.breed || 'default';
        const defaults = BREED_DEFAULTS[breed] || BREED_DEFAULTS['default'];

        this.baseMilk = animalProfile.baselineMilkYield || defaults.milk;
        this.baseWeight = animalProfile.baselineWeight || defaults.weight;
        this.baseFeed = defaults.feed;
        this.baseWater = defaults.water;
        this.lactationStage = animalProfile.lactationStage || 'mid';
        this.parity = animalProfile.parity || 1;
        this.calvingDate = animalProfile.calvingDate || null;

        // Parity adjustment: milk peaks at parity 3-4
        this.parityMilkFactor = this.parity <= 1 ? 0.75 :
            this.parity <= 3 ? 1.0 :
                this.parity <= 5 ? 0.95 : 0.85;
    }

    /**
     * Generate production signals for one tick.
     * 
     * @param {Object} hiddenState - { infectionLoad, stressLoad, fatigue, compensation }
     * @param {Object} env - { thi, hourOfDay, dayOfYear }
     * @param {number} tickIndex - Current tick
     * @param {number} tickMinutes - Minutes per tick
     * @param {number} [severityLevel] - Discrete severity 0-3 for direct milk coupling
     * @returns {Object} Production signals
     */
    generate(hiddenState, env, tickIndex, tickMinutes = 5, severityLevel = 0) {
        const I = safe(hiddenState.infectionLoad, 0);
        const S = safe(hiddenState.stressLoad, 0);
        const F = safe(hiddenState.fatigue, 0);

        const hourOfDay = safe(env.hourOfDay, 12);
        const hsi = safe(1 / (1 + Math.exp(-(env.thi - 72) / 8)), 0.5);

        // ── Days in milk (from calving date or estimated) ────────────
        const daysInMilk = this._daysInMilk(tickIndex, tickMinutes);

        // ── 1. Milk Yield (L/day, sampled 2-3× per day → per tick) ──
        const milkYield = this._computeMilkYield(I, S, F, daysInMilk, hourOfDay, severityLevel);

        // ── 2. Milk Conductivity (mS/cm) ────────────────────────────
        // Baseline ~5 mS/cm, rises with mastitis (infection)
        const milkConductivity = this._computeConductivity(I);

        // ── 3. Body Weight (kg) ──────────────────────────────────────
        const bodyWeight = this._computeBodyWeight(I, S);

        // ── 4. Feed Intake (kg DM/day → per tick rate) ──────────────
        const feedIntake = this._computeFeedIntake(I, S, hourOfDay);

        // ── 5. Water Intake (L/day → per tick rate) ─────────────────
        const waterIntake = this._computeWaterIntake(I, hsi, hourOfDay);

        return {
            milkYield: parseFloat(safe(milkYield, 0).toFixed(2)),
            milkConductivity: parseFloat(safe(milkConductivity, 5.0).toFixed(2)),
            bodyWeight: parseFloat(safe(bodyWeight, this.baseWeight).toFixed(1)),
            feedIntake: parseFloat(safe(feedIntake, 0).toFixed(2)),
            waterIntake: parseFloat(safe(waterIntake, 0).toFixed(2))
        };
    }

    // ─────────────────────────────────────────────────────────────────────────
    // PRIVATE METHODS
    // ─────────────────────────────────────────────────────────────────────────

    _daysInMilk(tickIndex, tickMinutes) {
        if (this.lactationStage === 'dry') return -1;
        if (this.calvingDate) {
            const now = new Date();
            return Math.floor((now - this.calvingDate) / (1000 * 60 * 60 * 24));
        }
        // Estimate from lactation stage
        const stageMap = { early: 30, mid: 120, late: 250, dry: -1 };
        return stageMap[this.lactationStage] || 120;
    }

    /**
     * Milk yield using Wood's lactation curve + disease impact.
     * 
     * DUAL penalty ensures strict monotonic decrease by severity:
     *   1. Nonlinear I^1.3 for continuous infection load coupling
     *   2. Direct (1 - 0.12*sev) for discrete severity guarantee
     * 
     * Milk = baseMilk × parityFactor × (1-0.75I^1.3) × (1-0.35S) × (1-0.2F) × (1-0.12*sev) × LC(t)
     */
    _computeMilkYield(I, S, F, daysInMilk, hourOfDay, severityLevel = 0) {
        if (this.lactationStage === 'dry' || daysInMilk < 0) return 0;

        // Wood's lactation curve: y = a * t^b * exp(-c*t)
        const a = 0.30, b = 0.25, c = 0.003;
        const t = Math.max(1, daysInMilk);
        const lc = a * Math.pow(t, b) * Math.exp(-c * t);
        const tPeak = b / c;
        const lcPeak = a * Math.pow(tPeak, b) * Math.exp(-c * tPeak);
        const lcNorm = lcPeak > 0 ? lc / lcPeak : 1.0;

        // NONLINEAR infection penalty: I^1.3 (convex→steeper at high I)
        const infectionPenalty = 1 - 0.75 * Math.pow(Math.max(0, I), 1.3);
        const stressPenalty = 1 - 0.35 * S;
        const fatiguePenalty = 1 - 0.2 * F;

        // DIRECT severity penalty: guarantees monotonic 0>1>2>3
        // Sev 0: 100%, Sev 1: 88%, Sev 2: 76%, Sev 3: 64%
        const sevPenalty = 1 - 0.12 * Math.min(3, Math.max(0, severityLevel));

        const diseaseFactor = infectionPenalty * stressPenalty * fatiguePenalty * sevPenalty;

        // Milking pattern: 2 peaks per day (6am, 6pm), never drops to zero
        const milkingCycle = 0.7 + 0.3 * Math.cos(2 * Math.PI * (hourOfDay - 6) / 12);

        const yield_ = this.baseMilk * this.parityMilkFactor * clamp(lcNorm, 0.05, 1.0) *
            clamp(diseaseFactor, 0.02, 1.0) * milkingCycle;

        // Reduced noise (0.1) to prevent stochastic severity crossovers
        return clamp(yield_ + gaussNoise(0.1), 0, this.baseMilk * 1.3);
    }

    /**
     * Milk electrical conductivity — rises with mastitis.
     * Baseline ~5 mS/cm, increases with infection load.
     */
    _computeConductivity(I) {
        if (this.lactationStage === 'dry') return 0;
        const baseline = 5.0;
        const infectionEffect = 1 + 0.4 * I; // Up to 40% increase
        return clamp(baseline * infectionEffect + gaussNoise(0.2), 3.5, 9.0);
    }

    /**
     * Body weight — drops with infection and stress.
     */
    _computeBodyWeight(I, S) {
        const diseaseLoss = 1 - 0.05 * I - 0.03 * S;
        return clamp(this.baseWeight * diseaseLoss + gaussNoise(2.0),
            this.baseWeight * 0.8, this.baseWeight * 1.1);
    }

    /**
     * Feed intake — drops with illness, has circadian pattern.
     */
    _computeFeedIntake(I, S, hourOfDay) {
        const diseaseFactor = (1 - 0.3 * I) * (1 - 0.2 * S);

        // Circadian: cows eat more during daylight (6am-8pm)
        const feedCircadian = (hourOfDay >= 6 && hourOfDay <= 20) ? 1.0 : 0.3;

        const intake = this.baseFeed * clamp(diseaseFactor, 0.2, 1.0) * feedCircadian;
        return clamp(intake + gaussNoise(0.5), 0, this.baseFeed * 1.3);
    }

    /**
     * Water intake — rises with heat stress, drops with illness.
     */
    _computeWaterIntake(I, hsi, hourOfDay) {
        const heatFactor = 1 + 0.3 * hsi; // More water when hot
        const illnessFactor = 1 - 0.25 * I;

        // Circadian: drink more during day
        const waterCircadian = (hourOfDay >= 6 && hourOfDay <= 20) ? 1.0 : 0.4;

        const intake = this.baseWater * heatFactor * clamp(illnessFactor, 0.3, 1.0) * waterCircadian;
        return clamp(intake + gaussNoise(2.0), 0, this.baseWater * 2.0);
    }
}

module.exports = ProductionModel;
