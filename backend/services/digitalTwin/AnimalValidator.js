/**
 * AnimalValidator — 3-Level Validation for Animal Profiles
 * 
 * GoMata Digital Twin v4
 * 
 * Level 1: Structural — field types, ranges, enums
 * Level 2: Biological — breed-plausible values
 * Level 3: Cross-consistency — field combinations
 */

'use strict';

// ── Breed reference ranges ──────────────────────────────────────────────────

const BREED_RANGES = {
    'Holstein': { milkMin: 20, milkMax: 55, weightMin: 550, weightMax: 800 },
    'Jersey': { milkMin: 15, milkMax: 35, weightMin: 350, weightMax: 550 },
    'Gir': { milkMin: 5, milkMax: 20, weightMin: 300, weightMax: 500 },
    'Sahiwal': { milkMin: 8, milkMax: 25, weightMin: 350, weightMax: 550 },
    'Red Sindhi': { milkMin: 5, milkMax: 18, weightMin: 250, weightMax: 450 },
    'Tharparkar': { milkMin: 5, milkMax: 18, weightMin: 280, weightMax: 480 },
    'default': { milkMin: 3, milkMax: 60, weightMin: 200, weightMax: 900 }
};

const VALID_LACTATION_STAGES = ['early', 'mid', 'late', 'dry'];
const VALID_HEAT_TOLERANCES = ['low', 'medium', 'high'];

// ═════════════════════════════════════════════════════════════════════════════

class AnimalValidator {
    /**
     * Validate an animal profile at all 3 levels.
     * 
     * @param {Object} profile - Animal profile to validate
     * @returns {Object} { valid, errors, warnings, level1, level2, level3 }
     */
    static validate(profile) {
        const errors = [];
        const warnings = [];

        const l1 = AnimalValidator._level1Structural(profile, errors);
        const l2 = AnimalValidator._level2Biological(profile, errors, warnings);
        const l3 = AnimalValidator._level3CrossConsistency(profile, errors, warnings);

        return {
            valid: errors.length === 0,
            errors,
            warnings,
            level1: l1,
            level2: l2,
            level3: l3
        };
    }

    // ─────────────────────────────────────────────────────────────────────────
    // LEVEL 1 — Structural Validation
    // ─────────────────────────────────────────────────────────────────────────

    static _level1Structural(p, errors) {
        let passed = 0, total = 0;

        // BCS ∈ [1, 5]
        total++;
        if (p.bodyConditionScore !== undefined) {
            if (p.bodyConditionScore >= 1 && p.bodyConditionScore <= 5) {
                passed++;
            } else {
                errors.push(`BCS ${p.bodyConditionScore} out of range [1, 5]`);
            }
        } else { passed++; } // Optional

        // Parity ≥ 0
        total++;
        if (p.parity !== undefined) {
            if (Number.isInteger(p.parity) && p.parity >= 0) {
                passed++;
            } else {
                errors.push(`Parity ${p.parity} must be non-negative integer`);
            }
        } else { passed++; }

        // Lactation stage valid enum
        total++;
        if (p.lactationStage) {
            if (VALID_LACTATION_STAGES.includes(p.lactationStage)) {
                passed++;
            } else {
                errors.push(`Invalid lactationStage: ${p.lactationStage}`);
            }
        } else { passed++; }

        // Heat tolerance valid enum
        total++;
        if (p.geneticHeatTolerance) {
            if (VALID_HEAT_TOLERANCES.includes(p.geneticHeatTolerance)) {
                passed++;
            } else {
                errors.push(`Invalid geneticHeatTolerance: ${p.geneticHeatTolerance}`);
            }
        } else { passed++; }

        // Calving date ≤ today
        total++;
        if (p.calvingDate) {
            const d = new Date(p.calvingDate);
            if (!isNaN(d) && d <= new Date()) {
                passed++;
            } else {
                errors.push(`CalvingDate ${p.calvingDate} is in the future or invalid`);
            }
        } else { passed++; }

        // Previous disease count ≥ 0
        total++;
        if (p.previousDiseaseCount !== undefined) {
            if (Number.isInteger(p.previousDiseaseCount) && p.previousDiseaseCount >= 0) {
                passed++;
            } else {
                errors.push(`previousDiseaseCount must be non-negative integer`);
            }
        } else { passed++; }

        return { passed, total };
    }

    // ─────────────────────────────────────────────────────────────────────────
    // LEVEL 2 — Biological Plausibility
    // ─────────────────────────────────────────────────────────────────────────

    static _level2Biological(p, errors, warnings) {
        let passed = 0, total = 0;
        const breed = p.breed || 'default';
        const ranges = BREED_RANGES[breed] || BREED_RANGES['default'];

        // Milk yield matches breed
        total++;
        if (p.baselineMilkYield !== undefined) {
            if (p.baselineMilkYield >= ranges.milkMin && p.baselineMilkYield <= ranges.milkMax) {
                passed++;
            } else {
                warnings.push(`Milk yield ${p.baselineMilkYield} unusual for ${breed} [${ranges.milkMin}-${ranges.milkMax}]`);
                passed++; // Warning only, not error
            }
        } else { passed++; }

        // Weight matches breed
        total++;
        if (p.baselineWeight !== undefined) {
            if (p.baselineWeight >= ranges.weightMin && p.baselineWeight <= ranges.weightMax) {
                passed++;
            } else {
                warnings.push(`Weight ${p.baselineWeight} unusual for ${breed} [${ranges.weightMin}-${ranges.weightMax}]`);
                passed++;
            }
        } else { passed++; }

        // Age-parity consistency (parity should be < age - 1.5)
        total++;
        if (p.parity !== undefined && p.age !== undefined) {
            if (p.parity <= Math.floor(p.age - 1.5)) {
                passed++;
            } else {
                errors.push(`Parity ${p.parity} impossible for age ${p.age}`);
            }
        } else { passed++; }

        return { passed, total };
    }

    // ─────────────────────────────────────────────────────────────────────────
    // LEVEL 3 — Cross-Consistency
    // ─────────────────────────────────────────────────────────────────────────

    static _level3CrossConsistency(p, errors, warnings) {
        let passed = 0, total = 0;

        // Dry → milk ≈ 0
        total++;
        if (p.lactationStage === 'dry' && p.baselineMilkYield > 0) {
            warnings.push(`Dry cow with baselineMilkYield=${p.baselineMilkYield} — will be zeroed in simulation`);
            passed++; // Non-blocking
        } else { passed++; }

        // Parity 0 → no calving date
        total++;
        if (p.parity === 0 && p.calvingDate) {
            errors.push(`Parity 0 (heifer) cannot have a calving date`);
        } else { passed++; }

        // BCS-weight correlation (very rough check)
        total++;
        if (p.bodyConditionScore !== undefined && p.baselineWeight !== undefined) {
            const expectedMinWeight = 250 + (p.bodyConditionScore - 1) * 50;
            if (p.baselineWeight >= expectedMinWeight * 0.7) {
                passed++;
            } else {
                warnings.push(`BCS ${p.bodyConditionScore} with weight ${p.baselineWeight} seems inconsistent`);
                passed++;
            }
        } else { passed++; }

        // Early lactation → should have recent calving date
        total++;
        if (p.lactationStage === 'early' && p.calvingDate) {
            const daysSinceCalving = (Date.now() - new Date(p.calvingDate)) / (1000 * 60 * 60 * 24);
            if (daysSinceCalving <= 120) {
                passed++;
            } else {
                warnings.push(`Lactation 'early' but calving was ${Math.floor(daysSinceCalving)} days ago`);
                passed++;
            }
        } else { passed++; }

        return { passed, total };
    }
}

module.exports = AnimalValidator;
