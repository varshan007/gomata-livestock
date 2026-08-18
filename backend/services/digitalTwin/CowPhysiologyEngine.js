/**
 * CowPhysiologyEngine — Per-Cow Hidden-State Physiological Simulator
 * 
 * GoMata Digital Twin Simulator v2
 * 
 * Maintains 5 latent state variables per cow that evolve via differential
 * equations every tick. These hidden states are the TRUE biological reality
 * from which all observable sensor readings are derived.
 * 
 * Hidden States:
 *   I(t) — Infection Load      [0, 1]   SIR-style logistic growth
 *   R(t) — Immune Response     [0, 1]   Delayed activation, natural decay
 *   S(t) — Stress Load         [0, 1.5] Environment-driven
 *   C(t) — Compensation        [0, 1]   Ability to mask symptoms
 *   F(t) — Fatigue             [0, 1]   Accumulated exhaustion
 * 
 * These states are NEVER exposed to ML models — only used as labels.
 */

'use strict';

// ── Evolution Parameters (biologically calibrated) ──────────────────────────

const PARAMS = {
    // Infection dynamics
    ALPHA: 0.15,       // Logistic growth rate
    BETA: 0.08,        // Immune clearance rate
    INFECTION_SEED_MIN: 0.02,  // Minimum starting infection load
    INFECTION_SEED_MAX: 0.08,  // Maximum starting infection load

    // Immune dynamics
    GAMMA: 0.10,       // Immune activation rate
    DELTA: 0.02,       // Immune natural decay
    TAU_TICKS: 72,     // Immune lag — 72 ticks × 5 min = 6 hours

    // Compensation dynamics
    ETA_1: 0.03,       // Compensation drain from infection
    ETA_2: 0.02,       // Compensation drain from stress
    COMP_RECOVERY: 0.005, // Natural compensation recovery rate

    // Fatigue dynamics
    KAPPA: 0.01,       // Fatigue from stress
    LAMBDA: 0.02,      // Fatigue from infection
    MU: 0.005,         // Fatigue natural recovery

    // Clamp bounds
    INFECTION_RESOLVED_THRESHOLD: 0.005, // Below this = resolved
};

// ── Individual Variation Generator ──────────────────────────────────────────

function gaussianNoise(mean = 0, std = 1) {
    // Box-Muller transform (guarded against u1=0 → log(0)=-Infinity)
    const u1 = Math.max(1e-10, Math.random());
    const u2 = Math.random();
    const result = mean + std * Math.sqrt(-2 * Math.log(u1)) * Math.cos(2 * Math.PI * u2);
    return isFinite(result) ? result : mean; // Fallback to mean if NaN/Inf
}

// ═════════════════════════════════════════════════════════════════════════════

class CowPhysiologyEngine {
    /**
     * @param {string} cowId       - Unique cow identifier
     * @param {Object} [opts]      - Individual variation parameters
     * @param {number} [opts.age]  - Age in years (affects immune response)
     * @param {string} [opts.breed]
     */
    constructor(cowId, opts = {}) {
        this.cowId = cowId;

        // ── Individual variation (no two cows are identical) ─────────
        this.individualParams = this._generateIndividualParams(opts);

        // ── Hidden state vector ──────────────────────────────────────
        this.state = {
            I: 0.0,    // Infection load
            R: 0.0,    // Immune response
            S: 0.0,    // Stress load
            C: 1.0,    // Compensation capacity (starts full)
            F: 0.0,    // Fatigue
        };

        // ── Infection history buffer for immune lag (τ) ──────────────
        this.infectionHistory = new Array(PARAMS.TAU_TICKS + 1).fill(0.0);

        // ── Episode tracking ─────────────────────────────────────────
        this.currentEpisode = null;     // Active infection episode
        this.episodePhase = 'healthy';  // healthy | incubation | onset | peak | plateau | recovery | resolved
        this.ticksSinceEpisodeStart = 0;
        this.peakInfection = 0;

        // ── Cumulative metrics ───────────────────────────────────────
        this.totalInfectionBurden = 0;
        this.episodeCount = 0;
    }

    /**
     * @param {number} environmentalStress - S(t) from EnvironmentModel
     * @param {Object} [context] - v4 user data modifiers
     * @param {number} [context.susceptibilityMod] - Infection susceptibility multiplier (parity, vacc, postpartum)
     * @param {number} [context.recoveryMod] - Recovery rate multiplier (antibiotics)
     * @param {number} [context.stressSpike] - Additive stress from management events
     * @param {number} [context.bcsCapacity] - Immune capacity from BCS (BCS/3.0)
     * @param {number} [context.heatToleranceFactor] - Heat stress modifier {low:1.3, med:1.0, high:0.7}
     * @param {number} [context.lactationStressFactor] - Lactation stress modifier
     * @param {boolean} [context.vaccinationActive] - Whether vaccination is actively protecting
     * @param {boolean} [context.antibioticActive] - Whether antibiotic treatment is active
     * @returns {Object} Current hidden state snapshot
     */
    evolve(environmentalStress, context = {}) {
        const p = this.individualParams;
        const s = this.state;

        // NaN-safe helper: replaces NaN/Infinity with fallback
        const safe = (v, fallback = 0) => isFinite(v) ? v : fallback;

        // Sanitize inputs
        const sI = safe(s.I, 0);
        const sR = safe(s.R, 0);
        const sC = safe(s.C, 1);
        const sF = safe(s.F, 0);
        const alpha = safe(p.alpha, PARAMS.ALPHA);
        const beta = safe(p.beta, PARAMS.BETA);
        const gamma = safe(p.gamma, PARAMS.GAMMA);
        const delta = safe(p.delta, PARAMS.DELTA);

        // ── v4 Context modifiers (default = neutral) ─────────────────
        const susceptMod = safe(context.susceptibilityMod, 1.0);
        const recoveryMod = safe(context.recoveryMod, 1.0);
        const stressSpike = safe(context.stressSpike, 0);
        const bcsCapacity = safe(context.bcsCapacity, 1.0);
        const heatTolFactor = safe(context.heatToleranceFactor, 1.0);
        const lactStressFactor = safe(context.lactationStressFactor, 1.0);
        const vaccActive = !!context.vaccinationActive;
        const abxActive = !!context.antibioticActive;

        // ── 1. Infection Load Evolution ──────────────────────────────
        // Vaccination directly reduces infection force by 60%
        const vaccReduction = vaccActive ? 0.4 : 1.0; // 60% reduction when vaccinated
        let effectiveAlpha = alpha * susceptMod * vaccReduction;

        // Antibiotic effect on growth: crush infection growth by 80%
        // Without this, logistic growth at I≈0.5 (its peak) overwhelms decay
        if (abxActive) {
            effectiveAlpha *= 0.2;
        }

        // I(t+1) = I(t) + α_eff·I(t)·(1−I(t)) − β·R_eff·I(t)
        const infectionGrowth = effectiveAlpha * sI * (1 - sI);
        const immuneClearance = beta * sR * sI;
        let newI = safe(sI + infectionGrowth - immuneClearance, 0);

        // Antibiotic direct suppression: 10% decay per tick compounds over time
        // At I=0.8: tick 1→0.72, tick 10→0.28, tick 20→0.10 (clearance in ~30 ticks = 2.5h)
        if (abxActive && newI > 0) {
            newI *= 0.90;
        }

        // Infection can't go below 0 or above 1
        newI = Math.max(0, Math.min(1.0, newI));

        // Auto-resolve when infection drops below threshold
        if (newI < PARAMS.INFECTION_RESOLVED_THRESHOLD && sI > 0 && this.episodePhase !== 'healthy') {
            newI = 0;
        }

        // ── 2. Immune Response (BCS-scaled + stress-suppressed) ──────
        // R(t+1) = R(t) + γ_eff·I(t−τ)·exp(−k·S)·bcsCapacity − δ·R(t)
        const delayedInfection = safe(this.infectionHistory[0], 0);
        const envStress = safe(environmentalStress, 0);
        const immuneSuppression = Math.exp(-2.0 * envStress);
        const gammaEff = gamma * recoveryMod;  // v4: antibiotics boost γ
        const immuneActivation = gammaEff * delayedInfection * immuneSuppression * bcsCapacity;
        const immuneDecay = delta * sR;
        let newR = safe(sR + immuneActivation - immuneDecay, 0);

        // Antibiotic immune boost: +0.005 per tick (~0.25 over 50 ticks)
        if (abxActive) {
            newR += 0.005;
        }
        newR = Math.max(0, Math.min(1.0, newR));

        // ── 3. Stress Load (environment × user factors + mgmt spikes) ─
        const modifiedStress = envStress * heatTolFactor * lactStressFactor + stressSpike;
        const newS = Math.max(0, Math.min(1.0, modifiedStress));

        // ── 4. Compensation Capacity ─────────────────────────────────
        // C(t+1) = C(t) − η₁·I(t) − η₂·S(t) + recovery when healthy
        let newC = safe(sC - PARAMS.ETA_1 * newI - PARAMS.ETA_2 * newS, 1);
        // Slowly recover when infection is low
        if (newI < 0.1) {
            newC += PARAMS.COMP_RECOVERY * (1 - newI);
        }
        newC = Math.max(0, Math.min(1.0, safe(newC, 1)));

        // ── 5. Fatigue ───────────────────────────────────────────────
        // F(t+1) = F(t) + κ·S(t) + λ·I(t) − μ·F(t)
        let newF = safe(sF + PARAMS.KAPPA * newS + PARAMS.LAMBDA * newI - PARAMS.MU * sF, 0);
        newF = Math.max(0, Math.min(1.0, newF));

        // ── 6. Compensation Collapse Detection ──────────────────────
        // Stricter guard: requires severe infection + high stress + sustained 12h
        // Makes collapse rare (<8%) and clinically meaningful
        const collapseCondition = (newI > 0.6 && newS > 0.5);
        if (collapseCondition) {
            this.collapseAboveThresholdTicks = (this.collapseAboveThresholdTicks || 0) + 1;
        } else {
            this.collapseAboveThresholdTicks = 0;
        }
        // 144 ticks = 12 hours at 5-min interval
        this.compensationCollapse = (collapseCondition && this.collapseAboveThresholdTicks >= 144);

        // ── 7. Track stress duration for mixed-state logic ───────────
        if (newS > 0.4) {
            this.stressAboveThresholdTicks = (this.stressAboveThresholdTicks || 0) + 1;
        } else {
            this.stressAboveThresholdTicks = 0;
        }
        if (newI > 0.05) {
            this.infectionAboveThresholdTicks = (this.infectionAboveThresholdTicks || 0) + 1;
        } else {
            this.infectionAboveThresholdTicks = 0;
        }

        // ── Update infection history buffer (shift left, push new) ───
        this.infectionHistory.shift();
        this.infectionHistory.push(newI);

        // ── Update state ─────────────────────────────────────────────
        s.I = newI;
        s.R = newR;
        s.S = newS;
        s.C = newC;
        s.F = newF;

        // ── Track episode phase ──────────────────────────────────────
        this._updateEpisodePhase();

        // ── Cumulative metrics ───────────────────────────────────────
        this.totalInfectionBurden += newI;

        return this.getSnapshot();
    }

    // ─────────────────────────────────────────────────────────────────────────
    // INFECTION SEEDING — Called by EpisodeScheduler
    // ─────────────────────────────────────────────────────────────────────────

    /**
     * Seed an infection episode. Sets a small initial I(0) which grows
     * logistically via the evolution equations — NOT a random spike.
     * 
     * @param {Object} episode - Episode metadata
     * @param {string} episode.diseaseType   - 'brd' | 'mastitis' | 'laminitis' | 'generic'
     * @param {string} episode.targetSeverity - 'mild' | 'moderate' | 'severe'
     */
    seedInfection(episode) {
        // Seed infection load based on target severity
        const severitySeeds = {
            mild: { min: 0.015, max: 0.04, alphaScale: 0.7 },
            moderate: { min: 0.03, max: 0.06, alphaScale: 1.0 },
            severe: { min: 0.05, max: 0.10, alphaScale: 1.4 }
        };

        const seed = severitySeeds[episode.targetSeverity] || severitySeeds.moderate;

        this.state.I = seed.min + Math.random() * (seed.max - seed.min);
        this.individualParams.alpha = PARAMS.ALPHA * seed.alphaScale * (this.individualParams.alphaMultiplier || 1.0);

        this.currentEpisode = {
            ...episode,
            startTick: 0,
            seedLoad: this.state.I
        };
        this.episodePhase = 'incubation';
        this.ticksSinceEpisodeStart = 0;
        this.peakInfection = 0;
        this.episodeCount++;
    }

    /**
     * Seed infection from herd transmission (contact-based).
     * Lower initial load than scheduled episodes.
     * 
     * @param {number} sourceInfection - Source cow's infection load
     * @param {number} transmissionProb - Probability that was rolled
     */
    seedFromTransmission(sourceInfection, transmissionProb) {
        if (this.state.I > 0.01) return; // Already infected

        const seed = sourceInfection * transmissionProb * 0.3;
        this.state.I = Math.max(0.01, Math.min(0.05, seed));

        this.currentEpisode = {
            diseaseType: 'transmitted',
            targetSeverity: 'moderate',
            source: 'herd_transmission',
            startTick: 0,
            seedLoad: this.state.I
        };
        this.episodePhase = 'incubation';
        this.ticksSinceEpisodeStart = 0;
        this.peakInfection = 0;
        this.episodeCount++;
    }

    // ─────────────────────────────────────────────────────────────────────────
    // EPISODE PHASE TRACKING
    // ─────────────────────────────────────────────────────────────────────────

    _updateEpisodePhase() {
        const I = this.state.I;
        const S = this.state.S;

        if (this.currentEpisode) {
            this.ticksSinceEpisodeStart++;

            if (I > this.peakInfection) {
                this.peakInfection = I;
            }
        }

        // ── Duration thresholds for mixed state ──────────────────────
        // Require sustained states to avoid noisy label transitions
        const MIN_INFECTION_TICKS = 72;  // 6 hours at 5-min ticks
        const MIN_STRESS_TICKS = 72;     // 6 hours at 5-min ticks (tightened for v4 seasonal)
        const isSustainedInfection = (this.infectionAboveThresholdTicks || 0) >= MIN_INFECTION_TICKS;
        const isSustainedStress = (this.stressAboveThresholdTicks || 0) >= MIN_STRESS_TICKS;

        // ── Mixed episode detection ─────────────────────────────────
        // When both infection AND stress are sustained → mixed state
        // v4: S threshold at 0.65 — antibiotic delay creates longer infection peaks
        if (I > 0.05 && S > 0.65 && isSustainedInfection && isSustainedStress) {
            if (!this.episodePhase.startsWith('mixed_')) {
                this.episodePhase = 'mixed_onset';
                this.mixedEpisodeActive = true;
                return;
            }
            if (this.episodePhase === 'mixed_onset' && (I >= 0.3 || S >= 0.7)) {
                this.episodePhase = 'mixed_peak';
                return;
            }
            if (this.episodePhase === 'mixed_peak' && I < 0.2 && S < 0.5) {
                this.episodePhase = 'mixed_recovery';
                return;
            }
            return; // Stay in current mixed phase
        }

        // Exiting mixed state
        if (this.episodePhase.startsWith('mixed_')) {
            if (I < 0.05 && S < 0.3) {
                this.episodePhase = 'healthy';
                this.mixedEpisodeActive = false;
            } else if (I < 0.05) {
                // Stress persists, infection cleared → stress track
                this.episodePhase = 'stress_onset';
                this.stressEpisodeActive = true;
                this.mixedEpisodeActive = false;
            }
            // else infection persists → let infection track take over below
            return;
        }

        // ── Stress-only episode logic (no infection) ─────────────────
        if (I < 0.01) {
            if (this.episodePhase === 'healthy' && S > 0.6) {
                this.episodePhase = 'stress_onset';
                this.stressEpisodeActive = true;
                this.ticksSinceEpisodeStart = 0;
                return;
            }

            if (this.episodePhase === 'stress_onset') {
                if (S > 0.75) {
                    this.episodePhase = 'stress_peak';
                } else if (S < 0.35) {
                    this.episodePhase = 'stress_recovery';
                }
                return;
            }

            if (this.episodePhase === 'stress_peak') {
                if (S < 0.5) {
                    this.episodePhase = 'stress_recovery';
                }
                return;
            }

            if (this.episodePhase === 'stress_recovery') {
                if (S < 0.25) {
                    this.episodePhase = 'healthy';
                    this.stressEpisodeActive = false;
                } else if (S > 0.6) {
                    this.episodePhase = 'stress_onset';
                }
                return;
            }
        }

        // ── Infection episode logic ──────────────────────────────────
        if (this.stressEpisodeActive && I >= 0.05) {
            this.stressEpisodeActive = false;
        }

        if (this.episodePhase === 'healthy') {
            return;
        }

        if (this.episodePhase === 'incubation') {
            const minIncubationTicks = 4;
            if (this.ticksSinceEpisodeStart >= minIncubationTicks && (I >= 0.10 || this.state.C < 0.7)) {
                this.episodePhase = 'onset';
            }
        } else if (this.episodePhase === 'onset') {
            if (I >= 0.30) {
                this.episodePhase = 'peak';
            }
        } else if (this.episodePhase === 'peak') {
            if (this.state.R > 0.3 && I < this.peakInfection * 0.9) {
                this.episodePhase = 'plateau';
            }
        } else if (this.episodePhase === 'plateau') {
            if (I < 0.15) {
                this.episodePhase = 'recovery';
            }
        } else if (this.episodePhase === 'recovery') {
            if (I < PARAMS.INFECTION_RESOLVED_THRESHOLD) {
                this.episodePhase = 'resolved';
                this.currentEpisode = null;
            }
        } else if (this.episodePhase === 'resolved') {
            this.episodePhase = 'healthy';
        }
    }

    // ─────────────────────────────────────────────────────────────────────────
    // SNAPSHOT — Current state for labeling and diagnostics
    // ─────────────────────────────────────────────────────────────────────────

    getSnapshot() {
        // NaN-safe: ensure all values are finite numbers
        const safe = (v, fallback = 0) => isFinite(v) ? v : fallback;

        const I = safe(this.state.I, 0);
        const R = safe(this.state.R, 0);
        const S = safe(this.state.S, 0);
        const C = safe(this.state.C, 1);
        const F = safe(this.state.F, 0);

        // ── Multi-state detection ───────────────────────────────────
        const isStressEpisode = this.episodePhase.startsWith('stress_');
        const isMixedEpisode = this.episodePhase.startsWith('mixed_');
        let infectionBinary = I >= 0.05 ? 1 : 0;
        let stressBinary = S >= 0.4 ? 1 : 0;
        const mixedStateBinary = isMixedEpisode ? 1 : 0;

        // ── Label integrity: mixed MUST imply both infection AND stress ──
        if (mixedStateBinary === 1) {
            infectionBinary = 1;
            stressBinary = 1;
        }

        // ── Severity (combined infection + stress) ──────────────────
        let severity = 'none';
        if (I >= 0.6) severity = 'severe';
        else if (I >= 0.3) severity = 'moderate';
        else if (I >= 0.05) severity = 'mild';

        if ((isStressEpisode || isMixedEpisode) && severity === 'none') {
            if (S >= 0.8) severity = 'severe';
            else if (S >= 0.6) severity = 'moderate';
            else if (S >= 0.4) severity = 'mild';
        }
        // Mixed: take max severity
        if (isMixedEpisode) {
            const stressSev = S >= 0.8 ? 3 : S >= 0.6 ? 2 : S >= 0.4 ? 1 : 0;
            const infSev = I >= 0.6 ? 3 : I >= 0.3 ? 2 : I >= 0.05 ? 1 : 0;
            severity = ['none', 'mild', 'moderate', 'severe'][Math.max(stressSev, infSev)];
        }

        // ── Disease label (backward compatible) ─────────────────────
        let diseaseLabel = 0;
        if (I >= 0.05 || this.episodePhase !== 'healthy') {
            diseaseLabel = 1;
        }

        // ── Disease type ────────────────────────────────────────────
        let diseaseType = 'none';
        if (isMixedEpisode) {
            diseaseType = 'mixed_infection_stress';
        } else if (this.currentEpisode?.diseaseType) {
            diseaseType = this.currentEpisode.diseaseType;
        } else if (isStressEpisode) {
            diseaseType = 'heat_stress';
        }

        return {
            cowId: this.cowId,
            // Hidden states (for training labels only)
            infectionLoad: parseFloat(I.toFixed(4)),
            immuneResponse: parseFloat(R.toFixed(4)),
            stressLoad: parseFloat(S.toFixed(4)),
            compensation: parseFloat(C.toFixed(4)),
            fatigue: parseFloat(F.toFixed(4)),
            compensationCollapse: !!this.compensationCollapse,
            // Labels
            episodePhase: this.episodePhase,
            diseaseLabel,
            infectionBinary,
            stressBinary,
            mixedStateBinary,
            severityLevel: severity,
            diseaseType,
            // Metrics
            peakInfection: parseFloat(safe(this.peakInfection, 0).toFixed(4)),
            episodeCount: this.episodeCount,
            ticksSinceEpisodeStart: this.ticksSinceEpisodeStart
        };
    }
    // ─────────────────────────────────────────────────────────────────────────
    // INDIVIDUAL VARIATION — No two cows are identical
    // ─────────────────────────────────────────────────────────────────────────

    _generateIndividualParams(opts) {
        const age = opts.age || (2 + Math.random() * 10); // 2-12 years
        const ageImmuneFactor = age < 3 ? 0.7 : (age > 8 ? 0.8 : 1.0); // Young/old = weaker immune

        // ── Static risk metadata (per-cow, never changes) ────────────
        const parity = age < 2.5 ? 0 : Math.min(7, Math.floor(age / 1.8) + Math.floor(Math.random() * 2));
        const lactationStages = ['dry', 'early', 'mid', 'late'];
        const lactationStage = parity === 0 ? 'dry' : lactationStages[Math.floor(Math.random() * lactationStages.length)];
        const bcs = parseFloat((2.5 + Math.random() * 1.5).toFixed(1)); // Body Condition Score 2.5–4.0
        const heatTolerances = ['low', 'medium', 'high'];
        const heatTolerance = heatTolerances[Math.floor(Math.random() * heatTolerances.length)];
        const breed = opts.breed || 'unknown';

        return {
            // Growth rate multiplier (some cows get sicker faster)
            alpha: PARAMS.ALPHA * (0.8 + Math.random() * 0.4),
            alphaMultiplier: 0.8 + Math.random() * 0.4,

            // Immune response (individual + age effect)
            beta: PARAMS.BETA * ageImmuneFactor * (0.85 + Math.random() * 0.3),
            gamma: PARAMS.GAMMA * ageImmuneFactor * (0.9 + Math.random() * 0.2),
            delta: PARAMS.DELTA * (0.9 + Math.random() * 0.2),

            // Body temperature baseline (individual variation ±0.3°C)
            tempBaseline: 38.5 + gaussianNoise(0, 0.15),

            // Heart rate baseline (individual variation ±8 bpm)
            hrBaseline: 65 + gaussianNoise(0, 4),

            // Respiration baseline (individual variation ±4 bpm)
            respBaseline: 26 + gaussianNoise(0, 2),

            // Activity baseline (normalized 0–1 internally)
            activityBaseline: 0.75 + gaussianNoise(0, 0.08),

            // Rumination baseline (minutes per hour)
            ruminationBaseline: 35 + gaussianNoise(0, 5),

            // Lying time baseline (minutes per hour)
            lyingBaseline: 25 + gaussianNoise(0, 3),

            // GPS movement radius baseline (meters)
            movementRadiusBaseline: 100 + gaussianNoise(0, 20),

            // Stress sensitivity (some cows stress more easily)
            stressSensitivity: 1.0 + gaussianNoise(0, 0.15),

            // Compensation capacity decay rate
            compDecayRate: 1.0 + gaussianNoise(0, 0.1),

            // ── Static metadata (for ML training labels) ─────────────
            age,
            breed,
            parity,
            lactationStage,
            bcs,
            heatTolerance
        };
    }

    /**
     * Check if this cow is susceptible to new infection.
     */
    isSusceptible() {
        return this.state.I < 0.01 && this.episodePhase === 'healthy';
    }

    /**
     * Check if this cow is infectious (can transmit).
     */
    isInfectious() {
        return this.state.I >= 0.05;
    }

    /**
     * Reset to healthy state (for intervention testing).
     */
    resetToHealthy() {
        this.state.I = 0;
        this.state.R = 0;
        this.state.C = 1.0;
        this.state.F = 0;
        this.currentEpisode = null;
        this.episodePhase = 'healthy';
        this.peakInfection = 0;
        this.infectionHistory.fill(0);
    }
}

module.exports = CowPhysiologyEngine;
