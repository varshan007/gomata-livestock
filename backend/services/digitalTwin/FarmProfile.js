/**
 * FarmProfile — Epidemiological Configuration Profiles
 * 
 * Dairy vs Beef parameter sets calibrated from peer-reviewed
 * livestock epidemiology data.
 * 
 * Usage:
 *   const profile = FarmProfile.get('dairy');
 *   const scheduler = new EpisodeScheduler({ ...config, farmProfile: profile });
 */

'use strict';

const PROFILES = {

    dairy: {
        farmType: 'dairy',
        // ── Annual incidence rates (per cow per year) ─────────────
        lambda_infection: 0.5,   // 0.3–0.6 clinical infections/cow/yr
        lambda_stress: 3.0,     // 2–4 heat stress waves/cow/yr

        // ── Severity distribution ────────────────────────────────
        severityWeights: [0.55, 0.30, 0.15], // mild, moderate, severe

        // ── Immune baseline ──────────────────────────────────────
        immuneBase: 0.8,

        // ── Heat tolerance distribution ──────────────────────────
        heatTolerance: { low: 0.30, medium: 0.50, high: 0.20 },

        // ── Stocking density (cows/m²) ───────────────────────────
        stockingDensityMean: 0.10,  // ~1 cow per 10m² in dairy barn

        // ── Episode duration (days) ──────────────────────────────
        infectionDuration: { min: 5, max: 14 },
        stressDuration: { min: 2, max: 7 },

        // ── Minimum gap (days) ───────────────────────────────────
        infectionGapDays: 60,
        stressGapDays: 30,

        // ── Disease type distribution ────────────────────────────
        diseaseTypes: ['brd', 'mastitis', 'laminitis', 'generic'],
        diseaseWeights: [0.30, 0.30, 0.15, 0.25],

        // ── Stress intensity distribution ────────────────────────
        stressIntensityWeights: [0.40, 0.40, 0.20] // mild, moderate, severe
    },

    beef: {
        farmType: 'beef',
        lambda_infection: 0.2,
        lambda_stress: 1.5,
        severityWeights: [0.60, 0.30, 0.10],
        immuneBase: 0.9,
        heatTolerance: { low: 0.20, medium: 0.50, high: 0.30 },
        stockingDensityMean: 0.04,  // ~1 cow per 25m² (open range)
        infectionDuration: { min: 5, max: 14 },
        stressDuration: { min: 2, max: 5 },
        infectionGapDays: 60,
        stressGapDays: 30,
        diseaseTypes: ['brd', 'laminitis', 'generic'],
        diseaseWeights: [0.45, 0.20, 0.35],
        stressIntensityWeights: [0.50, 0.35, 0.15]
    }
};

class FarmProfile {
    /**
     * @param {'dairy'|'beef'} farmType
     * @returns {Object} Profile configuration
     */
    static get(farmType = 'dairy') {
        const profile = PROFILES[farmType];
        if (!profile) {
            throw new Error(`Unknown farm type: ${farmType}. Use 'dairy' or 'beef'.`);
        }
        return { ...profile }; // Return copy
    }

    static list() {
        return Object.keys(PROFILES);
    }
}

module.exports = FarmProfile;
