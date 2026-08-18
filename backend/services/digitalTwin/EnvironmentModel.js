/**
 * EnvironmentModel — Zone-Level Environmental Simulation
 * 
 * GoMata Digital Twin Simulator v2
 * 
 * Generates biologically realistic environmental conditions:
 *   - Ambient temperature with circadian + seasonal patterns
 *   - Humidity (anti-correlated with temperature)
 *   - THI (Temperature-Humidity Index) — primary heat stress driver
 *   - Ammonia (NH₃) — zone-level with daily fluctuation
 *   - Airflow rate — static per zone with minor variation
 *   - Stocking density — animals per m²
 * 
 * No external API required — all patterns are physics-based.
 */

'use strict';

// ── Constants ───────────────────────────────────────────────────────────────

const TWO_PI = 2 * Math.PI;

// Indian tropical climate defaults (Uttar Pradesh baseline)
const DEFAULTS = {
    BASE_TEMP_C: 28.0,          // Annual mean ambient temperature
    TEMP_SEASONAL_AMP: 8.0,     // Summer 36°C, Winter 20°C
    TEMP_DIURNAL_AMP: 6.0,      // Daily swing ±6°C
    BASE_HUMIDITY: 55,           // Annual mean RH%
    HUMIDITY_DIURNAL_AMP: 20,    // RH swings ±20% daily
    AMMONIA_BASE_PPM: 8,        // Baseline barn ammonia
    AMMONIA_RANGE_PPM: 6,       // Fluctuates ±6 ppm
    AIRFLOW_BASE: 1.5,          // m/s baseline airflow
    AIRFLOW_NOISE: 0.3,         // m/s variation
    DEFAULT_ZONE_AREA_M2: 500,  // Default zone area
};

// ═════════════════════════════════════════════════════════════════════════════

class EnvironmentModel {
    /**
     * @param {Object} opts
     * @param {number} [opts.baseTempC]       - Annual mean ambient temp
     * @param {number} [opts.seasonalAmp]     - Seasonal temperature amplitude
     * @param {number} [opts.diurnalAmp]      - Daily temperature amplitude
     * @param {number} [opts.baseHumidity]    - Annual mean humidity %
     * @param {number} [opts.ammoniaPPM]      - Ammonia baseline
     * @param {number} [opts.airflowRate]     - Airflow m/s
     * @param {number} [opts.zoneAreaM2]      - Zone area in m²
     * @param {Date}   [opts.startDate]       - Simulation start date
     */
    constructor(opts = {}) {
        this.baseTempC = opts.baseTempC ?? DEFAULTS.BASE_TEMP_C;
        this.seasonalAmp = opts.seasonalAmp ?? DEFAULTS.TEMP_SEASONAL_AMP;
        this.diurnalAmp = opts.diurnalAmp ?? DEFAULTS.TEMP_DIURNAL_AMP;
        this.baseHumidity = opts.baseHumidity ?? DEFAULTS.BASE_HUMIDITY;
        this.ammoniaPPM = opts.ammoniaPPM ?? DEFAULTS.AMMONIA_BASE_PPM;
        this.airflowRate = opts.airflowRate ?? DEFAULTS.AIRFLOW_BASE;
        this.zoneAreaM2 = opts.zoneAreaM2 ?? DEFAULTS.DEFAULT_ZONE_AREA_M2;
        this.startDate = opts.startDate || new Date();
    }

    // ─────────────────────────────────────────────────────────────────────────
    // CORE: Compute full environment snapshot at time t
    // ─────────────────────────────────────────────────────────────────────────

    /**
     * @param {number} tickIndex      - Tick number since simulation start
     * @param {number} tickMinutes    - Minutes per tick (default 5)
     * @param {number} animalsInZone  - Current animal count in zone
     * @returns {Object} Environment snapshot
     */
    getEnvironment(tickIndex, tickMinutes = 5, animalsInZone = 50) {
        const minutesSinceStart = tickIndex * tickMinutes;
        const hourOfDay = ((this.startDate.getHours() + minutesSinceStart / 60) % 24);
        const dayOfYear = this._dayOfYear(this.startDate) + (minutesSinceStart / 1440);

        // ── Ambient Temperature ──────────────────────────────────────
        const ambientTemp = this._computeAmbientTemp(hourOfDay, dayOfYear);

        // ── Humidity (anti-correlated with temp) ─────────────────────
        const humidity = this._computeHumidity(hourOfDay, dayOfYear);

        // ── THI (Temperature-Humidity Index) ─────────────────────────
        const thi = this._computeTHI(ambientTemp, humidity);

        // ── Ammonia ──────────────────────────────────────────────────
        const ammonia = this._computeAmmonia(hourOfDay);

        // ── Airflow ──────────────────────────────────────────────────
        const airflow = this.airflowRate + (Math.random() - 0.5) * DEFAULTS.AIRFLOW_NOISE * 2;

        // ── Stocking Density ─────────────────────────────────────────
        // Raw: cows per m² (physical unit, expected 0.01–0.5)
        const densityRaw = animalsInZone / this.zoneAreaM2;
        // Normalized: 0–1 scale (3.0 cows/m² = overcrowded = 1.0)
        const densityNorm = Math.min(1.0, densityRaw / 3.0);

        return {
            ambientTemp: parseFloat(ambientTemp.toFixed(2)),
            humidity: parseFloat(Math.max(20, Math.min(95, humidity)).toFixed(1)),
            thi: parseFloat(thi.toFixed(2)),
            ammonia: parseFloat(Math.max(0, ammonia).toFixed(1)),
            airflow: parseFloat(Math.max(0.1, airflow).toFixed(2)),
            stockingDensity_raw: parseFloat(densityRaw.toFixed(4)),
            stockingDensity_normalized: parseFloat(densityNorm.toFixed(4)),
            hourOfDay: parseFloat(hourOfDay.toFixed(2)),
            dayOfYear: parseFloat(dayOfYear.toFixed(1))
        };
    }

    // ─────────────────────────────────────────────────────────────────────────
    // STRESS COMPUTATION — Used by CowPhysiologyEngine
    // ─────────────────────────────────────────────────────────────────────────

    /**
     * Compute environmental stress load S(t) using sigmoid-based HSI.
     * 
     * HSI_heat = sigmoid((THI - 72) / 8) × seasonal_multiplier
     * composite_stress = w1*HSI + w2*ammoniaNorm + w3*densityNorm + w4*(1-airflowNorm)
     * 
     * Seasonal multiplier concentrates heat stress in summer (day 150-250).
     * Winter base is ~40% of summer peak.
     * 
     * Weights: Heat 60%, Ammonia 10%, Density 10%, Airflow deficit 20%
     * 
     * @param {Object} env - Environment snapshot from getEnvironment()
     * @returns {number} Stress load [0, 1]
     */
    computeStressLoad(env) {
        // ── Sigmoid HSI (S-curve, 50% at THI=72, ~95% at THI=96) ──
        const hsiHeat = 1 / (1 + Math.exp(-(env.thi - 72) / 8));

        // ── Seasonal multiplier (summer peak, winter trough) ─────────
        // Uses sigmoid-amplified sine for sharp contrast
        const dayOfYear = (env.dayOfYear !== undefined) ? env.dayOfYear % 365 : 180;
        const sinVal = Math.sin(2 * Math.PI * (dayOfYear - 180) / 365);
        const seasonalMult = 0.4 + 0.6 / (1 + Math.exp(-4 * sinVal));
        // Winter: ~0.4, Summer: ~1.0

        // ── Ammonia normalized (5 ppm baseline, 35 ppm = max stress) ──
        const ammoniaNorm = Math.max(0, Math.min(1, (env.ammonia - 5) / 30));

        // ── Density (pre-normalized 0–1 from getEnvironment) ─────────
        const densityNorm = env.stockingDensity_normalized || 0;

        // ── Airflow relief (higher airflow = less stress, 3 m/s = full) ──
        const airflowNorm = Math.min(1, env.airflow / 3.0);

        // ── Weighted composite stress (w₁+w₂+w₃+w₄ = 1.0) ──────────
        const W_HEAT = 0.60;
        const W_AMMONIA = 0.10;
        const W_DENSITY = 0.10;
        const W_AIRFLOW = 0.20; // inverse: (1 - airflow_norm)

        const totalStress = W_HEAT * hsiHeat * seasonalMult
            + W_AMMONIA * ammoniaNorm
            + W_DENSITY * densityNorm
            + W_AIRFLOW * (1 - airflowNorm);

        return Math.max(0, Math.min(1.0, totalStress)); // Strict [0, 1]
    }

    // ─────────────────────────────────────────────────────────────────────────
    // CIRCADIAN PATTERNS
    // ─────────────────────────────────────────────────────────────────────────

    /**
     * Get circadian body temperature offset.
     * Cattle body temp peaks around 6 PM (+0.5°C), troughs at 5 AM (-0.3°C).
     * 
     * @param {number} hourOfDay
     * @returns {number} Offset in °C
     */
    getCircadianTempOffset(hourOfDay) {
        // Peak at 18:00 (6 PM), trough at 5:00 (5 AM)
        // Phase shift: peak at hour 18 → cos argument peaks at 0
        const phase = TWO_PI * ((hourOfDay - 18) / 24);
        return 0.4 * Math.cos(phase); // Range: -0.4 to +0.4°C
    }

    /**
     * Get circadian activity multiplier.
     * Cattle are most active morning (6-9 AM) and evening (4-7 PM).
     * Least active at night (10 PM - 4 AM) and midday (12-2 PM).
     * 
     * @param {number} hourOfDay
     * @returns {number} Multiplier [0.3, 1.2]
     */
    getCircadianActivityMultiplier(hourOfDay) {
        // Bimodal: peaks at 7 AM and 5 PM, troughs at 1 AM and 1 PM
        const morning = 0.3 * Math.cos(TWO_PI * ((hourOfDay - 7) / 24));
        const evening = 0.3 * Math.cos(TWO_PI * ((hourOfDay - 17) / 24));
        const nightDip = -0.2 * Math.cos(TWO_PI * ((hourOfDay - 1) / 24));
        return Math.max(0.3, Math.min(1.2, 0.7 + morning + evening + nightDip));
    }

    // ─────────────────────────────────────────────────────────────────────────
    // PRIVATE: Physics-based computations
    // ─────────────────────────────────────────────────────────────────────────

    _computeAmbientTemp(hourOfDay, dayOfYear) {
        // Seasonal: peaks at day 135 (mid-May India), trough at day 350 (mid-Dec)
        const seasonal = this.seasonalAmp * Math.sin(TWO_PI * (dayOfYear - 80) / 365);

        // Diurnal: peaks at 14:00 (2 PM), trough at 05:00 (5 AM)
        const diurnal = this.diurnalAmp * Math.sin(TWO_PI * (hourOfDay - 8) / 24);

        // Small random weather perturbation ±1.5°C
        const noise = (Math.random() - 0.5) * 3.0;

        return this.baseTempC + seasonal + diurnal + noise;
    }

    _computeHumidity(hourOfDay, dayOfYear) {
        // Anti-correlated with temperature
        // High humidity at night/morning, low in afternoon
        const diurnal = this.baseHumidity + DEFAULTS.HUMIDITY_DIURNAL_AMP * Math.cos(TWO_PI * (hourOfDay - 5) / 24);

        // Monsoon effect: humidity peaks June-September (day 150-270)
        const monsoonCenter = 210; // Late July
        const monsoonAmp = 15;
        const monsoon = monsoonAmp * Math.exp(-Math.pow((dayOfYear - monsoonCenter) / 60, 2));

        const noise = (Math.random() - 0.5) * 8;
        return diurnal + monsoon + noise;
    }

    _computeTHI(tempC, rh) {
        // Standard THI formula
        return (1.8 * tempC + 32) - (0.55 - 0.0055 * rh) * (1.8 * tempC - 26);
    }

    _computeAmmonia(hourOfDay) {
        // Higher at night (poor ventilation), lower during day
        const diurnal = DEFAULTS.AMMONIA_RANGE_PPM * 0.5 * Math.cos(TWO_PI * (hourOfDay - 3) / 24);
        const noise = (Math.random() - 0.5) * 2;
        return this.ammoniaPPM + diurnal + noise;
    }

    _dayOfYear(date) {
        const start = new Date(date.getFullYear(), 0, 0);
        const diff = date - start;
        return Math.floor(diff / 86400000);
    }
}

module.exports = EnvironmentModel;
