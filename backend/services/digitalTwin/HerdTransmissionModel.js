/**
 * HerdTransmissionModel — Contact-Based Disease Spread Simulation
 * 
 * GoMata Digital Twin Simulator v2
 * 
 * Implements an SIR-style (Susceptible → Infected → Recovered) 
 * contact-based transmission model within a herd.
 * 
 * Transmission Rule:
 *   If distance(i,j) < threshold:
 *     P_infect = θ · contact_intensity · I_source
 * 
 * Supports:
 *   - Zone-based contact graphs
 *   - R₀-calibrated transmission rates
 *   - Pen density effects
 *   - Vaccination immunity
 *   - Isolation (removing from contact graph)
 *   - Batch outbreak scenario generation
 */

'use strict';

const logger = require('../../utils/logger');
const LOG_SERVICE = 'herd_transmission';

// ── Constants ───────────────────────────────────────────────────────────────

const DEFAULTS = {
    BASE_THETA: 0.02,              // Base transmission probability per contact
    CONTACT_DISTANCE_M: 3.0,       // Distance threshold for close contact
    CHECK_INTERVAL_TICKS: 6,       // Check transmission every 6 ticks (30 min)
    VACCINATION_EFFICACY: 0.80,    // 80% reduction in susceptibility
    ISOLATION_DELAY_TICKS: 12,     // 1 hour delay before isolation takes effect
};

// ── Density Multipliers ─────────────────────────────────────────────────────

const DENSITY_MULTIPLIERS = {
    low: 0.5,      // < 1.0 animals/m² — sparse pasture
    medium: 1.0,   // 1.0-2.0 animals/m² — standard barn
    high: 2.0,     // > 2.0 animals/m² — intensive housing
};

// ═════════════════════════════════════════════════════════════════════════════

class HerdTransmissionModel {
    /**
     * @param {Object} opts
     * @param {number} [opts.r0]              - Reproductive number (1.5-4.0)
     * @param {string} [opts.penDensity]      - 'low' | 'medium' | 'high'
     * @param {number} [opts.contactThreshold] - Distance threshold in meters
     * @param {number} [opts.theta]           - Base transmission probability
     */
    constructor(opts = {}) {
        this.r0 = opts.r0 ?? 2.5;
        this.penDensity = opts.penDensity ?? 'medium';
        this.theta = opts.theta ?? DEFAULTS.BASE_THETA;
        this.contactThreshold = opts.contactThreshold ?? DEFAULTS.CONTACT_DISTANCE_M;

        // Scale theta to approximate target R₀
        // R₀ ≈ θ × contacts_per_tick × infectious_duration_ticks
        // Average infectious duration ≈ 1440 ticks (5 days), contacts ≈ 5-10/tick
        this.effectiveTheta = this.theta * (this.r0 / 2.5);

        // Tracking
        this.transmissionLog = [];
        this.totalTransmissions = 0;
    }

    // ─────────────────────────────────────────────────────────────────────────
    // CORE: Check and apply herd transmission for one tick
    // ─────────────────────────────────────────────────────────────────────────

    /**
     * Evaluate transmission events across all cow pairs.
     * 
     * @param {Map<string, Object>} cowEngines     - cowId → CowPhysiologyEngine
     * @param {Map<string, Object>} cowMetadata    - cowId → { animal, lngOffset, latOffset }
     * @param {number} currentTick
     * @param {Set<string>}  isolatedCows         - Set of cowIds that are isolated
     * @param {Set<string>}  vaccinatedCows       - Set of cowIds that are vaccinated
     * @returns {Array<Object>} List of new transmission events
     */
    checkTransmission(cowEngines, cowMetadata, currentTick, isolatedCows = new Set(), vaccinatedCows = new Set()) {
        // Only check periodically to save computation
        if (currentTick % DEFAULTS.CHECK_INTERVAL_TICKS !== 0) return [];

        const events = [];
        const densityMult = DENSITY_MULTIPLIERS[this.penDensity] || 1.0;

        // Build list of infectious and susceptible cows
        const infectious = [];
        const susceptible = [];

        for (const [cowId, engine] of cowEngines) {
            if (isolatedCows.has(cowId)) continue; // Isolated — no contact

            if (engine.isInfectious()) {
                infectious.push({ cowId, engine, meta: cowMetadata.get(cowId) });
            } else if (engine.isSusceptible()) {
                susceptible.push({ cowId, engine, meta: cowMetadata.get(cowId) });
            }
        }

        if (infectious.length === 0 || susceptible.length === 0) return events;

        // Evaluate all infectious → susceptible pairs
        for (const source of infectious) {
            for (const target of susceptible) {
                // Zone co-location check (same zone = high contact)
                const sameZone = source.meta.animal.zone_id === target.meta.animal.zone_id;
                if (!sameZone) continue; // Only intra-zone transmission

                // Distance check (using GPS offsets)
                const dLng = source.meta.lngOffset - target.meta.lngOffset;
                const dLat = source.meta.latOffset - target.meta.latOffset;
                const distDeg = Math.sqrt(dLng * dLng + dLat * dLat);
                const distM = distDeg * 111139;

                if (distM > this.contactThreshold * densityMult * 5) continue;

                // Contact intensity: inversely proportional to distance
                const contactIntensity = Math.max(0, 1 - (distM / (this.contactThreshold * densityMult * 5)));

                // Transmission probability
                let pInfect = this.effectiveTheta * contactIntensity * source.engine.state.I;

                // Vaccination reduces susceptibility
                if (vaccinatedCows.has(target.cowId)) {
                    pInfect *= (1 - DEFAULTS.VACCINATION_EFFICACY);
                }

                // High density increases transmission
                pInfect *= densityMult;

                // Roll the dice
                if (Math.random() < pInfect) {
                    target.engine.seedFromTransmission(
                        source.engine.state.I,
                        pInfect
                    );

                    const event = {
                        tick: currentTick,
                        sourceCowId: source.cowId,
                        targetCowId: target.cowId,
                        sourceInfectionLoad: source.engine.state.I,
                        transmissionProb: pInfect,
                        distance: parseFloat(distM.toFixed(1))
                    };

                    events.push(event);
                    this.transmissionLog.push(event);
                    this.totalTransmissions++;

                    logger.debug({
                        service: LOG_SERVICE,
                        action: 'TRANSMISSION',
                        ...event
                    });
                }
            }
        }

        if (events.length > 0) {
            logger.info({
                service: LOG_SERVICE,
                action: 'TRANSMISSION_TICK',
                tick: currentTick,
                newTransmissions: events.length,
                totalInfectious: infectious.length,
                totalSusceptible: susceptible.length
            });
        }

        return events;
    }

    // ─────────────────────────────────────────────────────────────────────────
    // HERD METRICS — For forecasting dataset
    // ─────────────────────────────────────────────────────────────────────────

    /**
     * Compute herd-level epidemic metrics for a given tick.
     * 
     * @param {Map<string, Object>} cowEngines
     * @param {number} currentTick
     * @returns {Object} Herd-level metrics
     */
    computeHerdMetrics(cowEngines, currentTick) {
        let healthyCount = 0;
        let infectedCount = 0;
        let recoveredCount = 0;
        let totalInfectionBurden = 0;
        let maxInfectionLoad = 0;

        for (const [, engine] of cowEngines) {
            const snap = engine.getSnapshot();
            if (snap.infectionLoad >= 0.05) {
                infectedCount++;
                totalInfectionBurden += snap.infectionLoad;
                maxInfectionLoad = Math.max(maxInfectionLoad, snap.infectionLoad);
            } else if (snap.immuneResponse > 0.1 && snap.infectionLoad < 0.01) {
                recoveredCount++;
            } else {
                healthyCount++;
            }
        }

        const herdSize = cowEngines.size;

        // Spread velocity: rate of change in infected fraction
        const infectedFraction = infectedCount / herdSize;

        // Herd stability index: 1 = fully healthy, 0 = everyone infected
        const herdStability = 1 - (infectedCount / herdSize);

        // Estimate 7-day peak forecast (simple exponential projection)
        const ticksPer7Days = 7 * 288; // 7 days × 288 ticks/day
        const growthRate = infectedFraction > 0.01
            ? Math.log(1 + infectedFraction) / Math.max(1, currentTick)
            : 0;
        const forecastPeak7Day = Math.min(
            herdSize,
            infectedCount * Math.exp(growthRate * ticksPer7Days)
        );

        return {
            timestamp: new Date(),
            tick: currentTick,
            herdSize,
            healthyCount,
            infectedCount,
            recoveredCount,
            infectedFraction: parseFloat(infectedFraction.toFixed(4)),
            infectionBurden: parseFloat(totalInfectionBurden.toFixed(2)),
            maxInfectionLoad: parseFloat(maxInfectionLoad.toFixed(4)),
            spreadVelocity: parseFloat(growthRate.toFixed(6)),
            herdStabilityIndex: parseFloat(herdStability.toFixed(4)),
            forecastPeak7Day: Math.round(forecastPeak7Day),
            totalTransmissions: this.totalTransmissions
        };
    }

    // ─────────────────────────────────────────────────────────────────────────
    // SCENARIO RUNNER — Batch outbreak simulation
    // ─────────────────────────────────────────────────────────────────────────

    /**
     * Run a single outbreak scenario with given parameters.
     * Creates a miniature herd population and simulates spread.
     * 
     * @param {Object} scenario - From EpisodeScheduler.generateOutbreakScenarios()
     * @returns {Object} Scenario outcome metrics
     */
    runScenario(scenario) {
        const CowPhysiologyEngine = require('./CowPhysiologyEngine');
        const miniHerd = new Map();
        const vaccinatedCows = new Set();
        const isolatedCows = new Set();

        // Create mini herd
        for (let i = 0; i < scenario.herdSize; i++) {
            const cowId = `scenario_${scenario.scenarioId}_cow_${i}`;
            const engine = new CowPhysiologyEngine(cowId);
            miniHerd.set(cowId, engine);

            // Apply vaccination
            if (Math.random() < scenario.vaccinationCoverage) {
                vaccinatedCows.add(cowId);
            }
        }

        // Seed initial infections
        const cowIds = Array.from(miniHerd.keys());
        for (let i = 0; i < scenario.initialInfected; i++) {
            const targetId = cowIds[Math.floor(Math.random() * cowIds.length)];
            const engine = miniHerd.get(targetId);
            engine.seedInfection({
                diseaseType: 'generic',
                targetSeverity: 'moderate'
            });
        }

        // Create GPS metadata for distance calculations
        const miniMeta = new Map();
        for (const cowId of cowIds) {
            const dist = Math.random() * 50; // 0-50m radius pen
            const angle = Math.random() * 2 * Math.PI;
            miniMeta.set(cowId, {
                animal: { zone_id: 'scenario_zone' },
                lngOffset: dist * Math.cos(angle) / 111139,
                latOffset: dist * Math.sin(angle) / 111139
            });
        }

        // Create transmission model with scenario R₀
        const transmitter = new HerdTransmissionModel({
            r0: scenario.r0,
            penDensity: scenario.penDensity,
            theta: this.theta
        });

        // Run simulation
        const totalTicks = scenario.durationDays * 288;
        let peakInfected = 0;
        let peakTick = 0;
        let outbreakDuration = 0;

        for (let tick = 0; tick < totalTicks; tick++) {
            // Evolve all cows (minimal stress for scenario)
            for (const [, engine] of miniHerd) {
                engine.evolve(0.1);
            }

            // Check transmission
            transmitter.checkTransmission(
                miniHerd, miniMeta, tick, isolatedCows, vaccinatedCows
            );

            // Apply isolation strategy
            if (scenario.isolationStrategy !== 'none') {
                const delay = scenario.isolationStrategy === 'immediate' ? 0
                    : scenario.isolationStrategy === 'delayed_24h' ? 288
                        : 576; // 48h

                for (const [cowId, engine] of miniHerd) {
                    if (engine.isInfectious() && engine.ticksSinceEpisodeStart > delay) {
                        isolatedCows.add(cowId);
                    }
                }
            }

            // Track metrics
            let currentInfected = 0;
            for (const [, engine] of miniHerd) {
                if (engine.isInfectious()) currentInfected++;
            }

            if (currentInfected > peakInfected) {
                peakInfected = currentInfected;
                peakTick = tick;
            }

            if (currentInfected > 0) {
                outbreakDuration = tick;
            }

            // Early termination: outbreak resolved
            if (currentInfected === 0 && tick > 288) break;
        }

        // Economic computation
        const daysOutbreak = outbreakDuration / 288;
        const economicLossPerCow = 500; // ₹/day productivity loss
        const totalEconomicLoss = peakInfected * daysOutbreak * economicLossPerCow;
        const interventionCost = this._computeInterventionCost(scenario);
        const riskAdjustedCost = totalEconomicLoss + interventionCost;

        return {
            scenarioId: scenario.scenarioId,
            herdSize: scenario.herdSize,
            vaccinationCoverage: scenario.vaccinationCoverage,
            isolationStrategy: scenario.isolationStrategy,
            r0: scenario.r0,
            penDensity: scenario.penDensity,
            initialInfected: scenario.initialInfected,
            // Outcomes
            peakInfected,
            peakInfectedFraction: parseFloat((peakInfected / scenario.herdSize).toFixed(4)),
            outbreakDurationDays: parseFloat(daysOutbreak.toFixed(1)),
            totalTransmissions: transmitter.totalTransmissions,
            // Economics
            economicLoss: Math.round(totalEconomicLoss),
            interventionCost: Math.round(interventionCost),
            riskAdjustedCost: Math.round(riskAdjustedCost)
        };
    }

    _computeInterventionCost(scenario) {
        let cost = 0;
        // Vaccination: ₹200 per vaccinated cow
        cost += scenario.herdSize * scenario.vaccinationCoverage * 200;
        // Isolation: ₹500/day per isolated cow (estimated avg 5 days)
        if (scenario.isolationStrategy !== 'none') {
            cost += scenario.initialInfected * 3 * 500 * 5;
        }
        return cost;
    }

    // ─────────────────────────────────────────────────────────────────────────
    // GETTERS
    // ─────────────────────────────────────────────────────────────────────────

    getTransmissionLog() { return this.transmissionLog; }
    getTotalTransmissions() { return this.totalTransmissions; }
}

module.exports = HerdTransmissionModel;
