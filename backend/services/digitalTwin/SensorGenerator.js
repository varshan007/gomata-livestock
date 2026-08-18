/**
 * SensorGenerator — Observable Signal Generator from Hidden States
 * 
 * GoMata Digital Twin Simulator v2
 * 
 * Converts hidden physiological states (I, R, S, C, F) into observable
 * sensor readings using biologically calibrated equations. Each sensor
 * value is a deterministic function of the hidden states plus Gaussian noise.
 * 
 * Sensors Generated:
 *   - Temperature (°C)      — Fever driven by I and S, masked by C
 *   - Heart Rate (bpm)      — Elevated by I, S, F, and temp changes
 *   - Respiration (bpm)     — Stress and infection driven
 *   - Activity (0–1)        — Normalized, suppressed by I, F, S
 *   - Rumination (min/hr)   — Drops with infection and stress
 *   - Lying Time (min/hr)   — Increases with infection and fatigue
 *   - GPS Movement (meters) — Radius shrinks with infection
 *   - HSI (0–1)             — Observable heat stress index (from THI + temp response)
 * 
 * Calibration Targets:
 *   - Lag-1 autocorrelation ≈ 0.97
 *   - CV ≈ 0.10–0.15
 *   - Temp-HR correlation ≈ 0.8 during infection
 *   - Resp-THI correlation ≈ 0.9 during heat stress
 */

'use strict';

// ── Signal Coefficients (biologically calibrated) ───────────────────────────

const COEFF = {
    // Temperature: T = T_base + circadian + a₁·I·(1−C) + a₂·S·(1−C) + noise
    TEMP: {
        a1: 3.0,       // Infection → temp (max +3°C fever)
        a2: 0.8,       // Stress → temp (max +0.8°C)
        noise_std: 0.12, // Gaussian noise σ = 0.12°C
        smoothing: 0.92  // Exponential smoothing for autocorrelation
    },

    // Heart Rate: HR = HR_base + b₁·I + b₂·S + b₃·F + b₄·dT/dt + noise
    HR: {
        b1: 50,        // Infection → HR (max +50 bpm)
        b2: 25,        // Stress → HR (max +25 bpm)
        b3: 15,        // Fatigue → HR (max +15 bpm)
        b4: 30,        // Temp rate of change → HR (°C/tick × 30)
        noise_std: 2.5,
        smoothing: 0.90
    },

    // Respiration: Resp = Resp_base + c₁·S + c₂·I + noise
    RESP: {
        c1: 30,        // Stress → resp (max +30 bpm)
        c2: 20,        // Infection → resp (max +20 bpm)
        noise_std: 1.5,
        smoothing: 0.88
    },

    // Activity (0–1): Act = circadian × (base − d₁·I − d₂·F − d₃·S) + noise
    ACTIVITY: {
        d1: 0.50,      // Infection suppresses activity (max -0.50)
        d2: 0.25,      // Fatigue suppresses activity (max -0.25)
        d3: 0.15,      // Stress suppresses activity (max -0.15)
        noise_std: 0.04,
        smoothing: 0.85
    },

    // Rumination: Rum = base − e₁·I − e₂·S − e₃·F + noise
    RUMINATION: {
        e1: 25,        // Infection reduces rumination (max -25 min)
        e2: 10,        // Stress reduces rumination (max -10 min)
        e3: 8,         // Fatigue reduces rumination (max -8 min)
        noise_std: 2.5,
        smoothing: 0.88
    },

    // Lying Time: Lie = base + f₁·I + f₂·F + noise
    LYING: {
        f1: 20,        // Sick cows lie more (max +20 min)
        f2: 10,        // Fatigued cows lie more (max +10 min)
        noise_std: 2,
        smoothing: 0.90
    },

    // GPS Movement: M = base × (1 − g₁·I) + noise
    GPS: {
        g1: 0.8,       // Infection reduces movement radius (max 80% reduction)
        noise_std: 5,
        smoothing: 0.93
    }
};

// ── Gaussian Noise Generator ────────────────────────────────────────────────

function gaussNoise(std) {
    const u1 = Math.max(1e-10, Math.random());
    const u2 = Math.random();
    const result = std * Math.sqrt(-2 * Math.log(u1)) * Math.cos(2 * Math.PI * u2);
    return isFinite(result) ? result : 0;
}

// ═════════════════════════════════════════════════════════════════════════════

class SensorGenerator {
    /**
     * @param {Object} individualParams - From CowPhysiologyEngine
     */
    constructor(individualParams) {
        this.params = individualParams;

        // ── Smoothed state (for lag-1 autocorrelation ≈ 0.97) ────────
        this.smoothed = {
            temperature: individualParams.tempBaseline,
            heartRate: individualParams.hrBaseline,
            respiration: individualParams.respBaseline,
            activity: individualParams.activityBaseline,
            rumination: individualParams.ruminationBaseline,
            lying: individualParams.lyingBaseline,
            gpsRadius: individualParams.movementRadiusBaseline
        };

        // Track previous temperature for dT/dt
        this.prevTemp = individualParams.tempBaseline || 38.5;
    }

    // ─────────────────────────────────────────────────────────────────────────
    // GENERATE — Produce all 7 sensor readings from hidden states
    // ─────────────────────────────────────────────────────────────────────────

    /**
     * @param {Object} hiddenState    - From CowPhysiologyEngine.getSnapshot()
     * @param {Object} env            - From EnvironmentModel.getEnvironment()
     * @param {number} circadianTemp  - From EnvironmentModel.getCircadianTempOffset()
     * @param {number} circadianAct   - From EnvironmentModel.getCircadianActivityMultiplier()
     * @returns {Object} All sensor readings
     */
    generate(hiddenState, env, circadianTemp, circadianAct) {
        const p = this.params;
        const I = hiddenState.infectionLoad;
        const S = hiddenState.stressLoad;
        const C = hiddenState.compensation;
        const F = hiddenState.fatigue;

        // ── Compensation Collapse Variance ────────────────────────────
        // When C < 0.4 AND I > 0.5: regulatory breakdown
        // Noise variance increases → erratic signals, autocorr drops
        const collapseMultiplier = (hiddenState.compensationCollapse) ? 2.5 : 1.0;

        // ── Temperature ──────────────────────────────────────────────
        const tempRaw = p.tempBaseline
            + circadianTemp
            + COEFF.TEMP.a1 * I * (1 - C)  // Fever, masked by compensation
            + COEFF.TEMP.a2 * S * (1 - C)  // Heat stress, masked by compensation
            + gaussNoise(COEFF.TEMP.noise_std * collapseMultiplier);

        const temperature = this._smooth('temperature', tempRaw, COEFF.TEMP.smoothing);

        // dT/dt for heart rate coupling (guard against Inf-Inf=NaN)
        const rawDTdt = temperature - this.prevTemp;
        const dTdt = isFinite(rawDTdt) ? rawDTdt : 0;
        this.prevTemp = temperature;

        // ── Heart Rate ───────────────────────────────────────────────
        const hrRaw = p.hrBaseline
            + COEFF.HR.b1 * I
            + COEFF.HR.b2 * S
            + COEFF.HR.b3 * F
            + COEFF.HR.b4 * Math.abs(dTdt)
            + gaussNoise(COEFF.HR.noise_std * collapseMultiplier);

        const heartRate = this._smooth('heartRate', hrRaw, COEFF.HR.smoothing);

        // ── Respiration ──────────────────────────────────────────────
        const respRaw = p.respBaseline
            + COEFF.RESP.c1 * S
            + COEFF.RESP.c2 * I
            + gaussNoise(COEFF.RESP.noise_std * collapseMultiplier);

        const respiration = this._smooth('respiration', respRaw, COEFF.RESP.smoothing);

        // ── Activity ─────────────────────────────────────────────────
        const actRaw = circadianAct * (
            p.activityBaseline
            - COEFF.ACTIVITY.d1 * I
            - COEFF.ACTIVITY.d2 * F
            - COEFF.ACTIVITY.d3 * S
        ) + gaussNoise(COEFF.ACTIVITY.noise_std * collapseMultiplier);

        const activity = this._smooth('activity', Math.max(0, actRaw), COEFF.ACTIVITY.smoothing);

        // ── Rumination ───────────────────────────────────────────────
        const rumRaw = p.ruminationBaseline
            - COEFF.RUMINATION.e1 * I
            - COEFF.RUMINATION.e2 * S
            - COEFF.RUMINATION.e3 * F
            + gaussNoise(COEFF.RUMINATION.noise_std * collapseMultiplier);

        const rumination = this._smooth('rumination', Math.max(0, rumRaw), COEFF.RUMINATION.smoothing);

        // ── Lying Time ───────────────────────────────────────────────
        const lieRaw = p.lyingBaseline
            + COEFF.LYING.f1 * I
            + COEFF.LYING.f2 * F
            + gaussNoise(COEFF.LYING.noise_std);

        const lying = this._smooth('lying', Math.max(0, Math.min(60, lieRaw)), COEFF.LYING.smoothing);

        // ── GPS Movement Radius ──────────────────────────────────────
        const gpsRaw = p.movementRadiusBaseline * (1 - COEFF.GPS.g1 * I)
            + gaussNoise(COEFF.GPS.noise_std);

        const gpsRadius = this._smooth('gpsRadius', Math.max(5, gpsRaw), COEFF.GPS.smoothing);

        // NaN-safe output with baseline fallbacks
        const safe = (v, fallback) => isFinite(v) ? v : fallback;

        // ── Observable Heat Stress Index (different from latent S) ────
        // HSI = f(THI, cow_temp_deviation, respiration_elevation)
        // This is what a vet can OBSERVE, not the hidden stress state
        const thiNorm = env.thi ? Math.max(0, (env.thi - 68) / 20) : 0;
        const tempDeviation = Math.max(0, safe(temperature, p.tempBaseline) - p.tempBaseline) / 3;
        const respElevation = Math.max(0, safe(respiration, p.respBaseline) - p.respBaseline) / 30;
        const observableHSI = Math.max(0, Math.min(1,
            0.5 * thiNorm + 0.3 * tempDeviation + 0.2 * respElevation
        ));

        return {
            temperature: parseFloat(Math.max(36.5, Math.min(42.5, safe(temperature, p.tempBaseline))).toFixed(2)),
            heartRate: Math.round(Math.max(40, Math.min(180, safe(heartRate, p.hrBaseline)))),
            respiration: Math.round(Math.max(10, Math.min(80, safe(respiration, p.respBaseline)))),
            activity: parseFloat(Math.max(0, Math.min(1, safe(activity, p.activityBaseline))).toFixed(3)),
            rumination: parseFloat(Math.max(0, Math.min(60, safe(rumination, p.ruminationBaseline))).toFixed(1)),
            lying: parseFloat(Math.max(0, Math.min(60, safe(lying, p.lyingBaseline))).toFixed(1)),
            gpsRadius: parseFloat(Math.max(5, safe(gpsRadius, p.movementRadiusBaseline)).toFixed(1)),
            // Observable heat stress index (HSI ≠ latent stressLoad)
            heatStressIndex: parseFloat(observableHSI.toFixed(3))
        };
    }

    // ─────────────────────────────────────────────────────────────────────────
    // EXPONENTIAL SMOOTHING — Ensures high autocorrelation
    // ─────────────────────────────────────────────────────────────────────────

    _smooth(key, rawValue, alpha) {
        const prev = isFinite(this.smoothed[key]) ? this.smoothed[key] : rawValue;
        const val = alpha * prev + (1 - alpha) * rawValue;
        this.smoothed[key] = isFinite(val) ? val : rawValue;
        return this.smoothed[key];
    }
}

module.exports = SensorGenerator;
