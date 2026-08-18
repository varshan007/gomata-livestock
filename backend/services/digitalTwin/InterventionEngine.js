/**
 * InterventionEngine — Disease Intervention Strategy Simulator
 * 
 * GoMata Digital Twin Simulator v2
 * 
 * Simulates 5 intervention strategies for each outbreak scenario
 * and records outcomes for economic decision intelligence training.
 * 
 * Strategies:
 *   1. No intervention (baseline)
 *   2. Vaccination (reduces susceptibility 60-90%)
 *   3. Isolation (removes infected from contact graph)
 *   4. Pen lockdown (blocks cross-zone transmission)
 *   5. Hybrid (vaccination + isolation + lockdown)
 * 
 * Generates ~5000 outcome records for training a decision intelligence model.
 */

'use strict';

const HerdTransmissionModel = require('./HerdTransmissionModel');
const EpisodeScheduler = require('./EpisodeScheduler');
const logger = require('../../utils/logger');

const LOG_SERVICE = 'intervention_engine';

// ── Strategy Definitions ────────────────────────────────────────────────────

const STRATEGIES = {
    none: {
        name: 'No Intervention',
        vaccinationCoverage: 0,
        isolationStrategy: 'none',
        penLockdown: false,
        costPerCow: 0
    },
    vaccination: {
        name: 'Vaccination Only',
        vaccinationCoverage: 0.8,
        isolationStrategy: 'none',
        penLockdown: false,
        costPerCow: 200  // ₹ per vaccinated cow
    },
    isolation: {
        name: 'Isolation Only',
        vaccinationCoverage: 0,
        isolationStrategy: 'immediate',
        penLockdown: false,
        costPerCow: 500  // ₹/day per isolated cow
    },
    lockdown: {
        name: 'Pen Lockdown',
        vaccinationCoverage: 0,
        isolationStrategy: 'none',
        penLockdown: true,
        costPerCow: 150  // ₹/day productivity impact per cow
    },
    hybrid: {
        name: 'Hybrid (Vacc + Iso + Lockdown)',
        vaccinationCoverage: 0.6,
        isolationStrategy: 'immediate',
        penLockdown: true,
        costPerCow: 350
    }
};

// ═════════════════════════════════════════════════════════════════════════════

class InterventionEngine {
    /**
     * @param {Object} opts
     * @param {number} opts.numScenarios  - Base outbreak scenarios to generate
     */
    constructor(opts = {}) {
        this.numScenarios = opts.numScenarios || 1000;
        this.results = [];
    }

    // ─────────────────────────────────────────────────────────────────────────
    // BATCH RUN — Generate intervention outcomes
    // ─────────────────────────────────────────────────────────────────────────

    /**
     * Run all intervention strategies across all outbreak scenarios.
     * Generates numScenarios × 5 strategies = ~5000 outcome records.
     * 
     * @param {Object} opts - Scenario generation options
     * @returns {Array<Object>} Intervention outcomes
     */
    async runAll(opts = {}) {
        logger.info({
            service: LOG_SERVICE,
            action: 'BATCH_START',
            scenarios: this.numScenarios,
            strategies: Object.keys(STRATEGIES).length
        }, `[InterventionEngine] Starting batch: ${this.numScenarios} scenarios × ${Object.keys(STRATEGIES).length} strategies`);

        const scheduler = new EpisodeScheduler({
            totalTicks: 288 * 90, // 90 days
            tickMinutes: 5,
            numCows: 100
        });

        const scenarios = scheduler.generateOutbreakScenarios(this.numScenarios, opts);
        const allResults = [];
        const startTime = Date.now();

        for (let si = 0; si < scenarios.length; si++) {
            const scenario = scenarios[si];

            for (const [strategyKey, strategy] of Object.entries(STRATEGIES)) {
                // Override scenario with strategy params
                const modifiedScenario = {
                    ...scenario,
                    vaccinationCoverage: strategy.vaccinationCoverage,
                    isolationStrategy: strategy.isolationStrategy,
                    scenarioId: `${scenario.scenarioId}_${strategyKey}`
                };

                const transmitter = new HerdTransmissionModel({
                    r0: scenario.r0,
                    penDensity: scenario.penDensity
                });

                const outcome = transmitter.runScenario(modifiedScenario);

                // Add strategy metadata
                outcome.strategyKey = strategyKey;
                outcome.strategyName = strategy.name;
                outcome.penLockdown = strategy.penLockdown;

                // Recalculate intervention cost based on strategy
                outcome.interventionCost = this._computeStrategyCost(
                    strategy, scenario.herdSize, outcome.peakInfected,
                    outcome.outbreakDurationDays
                );

                outcome.riskAdjustedCost = outcome.economicLoss + outcome.interventionCost;

                // Relative metrics (compared to no-intervention baseline)
                outcome.baselineScenarioId = scenario.scenarioId;

                allResults.push(outcome);
            }

            // Log progress every 100 scenarios
            if ((si + 1) % 100 === 0) {
                const elapsed = ((Date.now() - startTime) / 1000).toFixed(1);
                logger.info({
                    service: LOG_SERVICE,
                    action: 'BATCH_PROGRESS',
                    completed: si + 1,
                    total: scenarios.length,
                    elapsedSec: elapsed
                });
            }
        }

        // Post-process: add relative improvement metrics
        this._addRelativeMetrics(allResults);

        this.results = allResults;

        const totalTime = ((Date.now() - startTime) / 1000).toFixed(1);
        logger.info({
            service: LOG_SERVICE,
            action: 'BATCH_COMPLETE',
            totalResults: allResults.length,
            timeSec: totalTime
        }, `[InterventionEngine] Complete: ${allResults.length} outcomes in ${totalTime}s`);

        return allResults;
    }

    // ─────────────────────────────────────────────────────────────────────────
    // COST COMPUTATION
    // ─────────────────────────────────────────────────────────────────────────

    _computeStrategyCost(strategy, herdSize, peakInfected, durationDays) {
        let cost = 0;

        // Vaccination cost
        if (strategy.vaccinationCoverage > 0) {
            cost += herdSize * strategy.vaccinationCoverage * 200;
        }

        // Isolation cost (per isolated cow per day)
        if (strategy.isolationStrategy !== 'none') {
            const avgIsolated = Math.ceil(peakInfected * 0.7);
            const isolationDays = Math.min(durationDays, 14);
            cost += avgIsolated * isolationDays * 500;
        }

        // Lockdown cost (productivity impact)
        if (strategy.penLockdown) {
            const lockdownDays = Math.min(durationDays, 21);
            cost += herdSize * lockdownDays * 50; // Reduced productivity cost
        }

        return Math.round(cost);
    }

    // ─────────────────────────────────────────────────────────────────────────
    // RELATIVE METRICS
    // ─────────────────────────────────────────────────────────────────────────

    _addRelativeMetrics(allResults) {
        // Group by base scenario
        const byScenario = new Map();
        for (const r of allResults) {
            const baseId = r.baselineScenarioId;
            if (!byScenario.has(baseId)) byScenario.set(baseId, []);
            byScenario.get(baseId).push(r);
        }

        // For each scenario group, compute improvements relative to 'none'
        for (const [, group] of byScenario) {
            const baseline = group.find(r => r.strategyKey === 'none');
            if (!baseline) continue;

            for (const r of group) {
                r.peakReduction = baseline.peakInfected > 0
                    ? parseFloat(((baseline.peakInfected - r.peakInfected) / baseline.peakInfected).toFixed(4))
                    : 0;
                r.durationReduction = baseline.outbreakDurationDays > 0
                    ? parseFloat(((baseline.outbreakDurationDays - r.outbreakDurationDays) / baseline.outbreakDurationDays).toFixed(4))
                    : 0;
                r.economicSaving = baseline.economicLoss - r.economicLoss;
                r.netBenefit = r.economicSaving - r.interventionCost;
                r.costEffectivenessRatio = r.interventionCost > 0
                    ? parseFloat((r.economicSaving / r.interventionCost).toFixed(2))
                    : 0;
                r.isOptimal = false;
            }

            // Mark the strategy with best net benefit as optimal
            const best = group.reduce((a, b) =>
                (a.netBenefit || 0) > (b.netBenefit || 0) ? a : b
            );
            best.isOptimal = true;
        }
    }

    // ─────────────────────────────────────────────────────────────────────────
    // SUMMARY STATS
    // ─────────────────────────────────────────────────────────────────────────

    getSummary() {
        if (this.results.length === 0) return null;

        const byStrategy = {};
        for (const r of this.results) {
            if (!byStrategy[r.strategyKey]) {
                byStrategy[r.strategyKey] = {
                    name: r.strategyName,
                    count: 0,
                    avgPeakReduction: 0,
                    avgDurationReduction: 0,
                    avgNetBenefit: 0,
                    optimalCount: 0
                };
            }
            const s = byStrategy[r.strategyKey];
            s.count++;
            s.avgPeakReduction += r.peakReduction || 0;
            s.avgDurationReduction += r.durationReduction || 0;
            s.avgNetBenefit += r.netBenefit || 0;
            if (r.isOptimal) s.optimalCount++;
        }

        for (const s of Object.values(byStrategy)) {
            s.avgPeakReduction = parseFloat((s.avgPeakReduction / s.count).toFixed(4));
            s.avgDurationReduction = parseFloat((s.avgDurationReduction / s.count).toFixed(4));
            s.avgNetBenefit = Math.round(s.avgNetBenefit / s.count);
        }

        return byStrategy;
    }

    getResults() { return this.results; }
}

module.exports = InterventionEngine;
