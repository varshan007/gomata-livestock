/**
 * ManagementEventSimulator — Generates realistic management event timelines
 * 
 * GoMata Digital Twin v4 — Contextual Causal Twin
 * 
 * Simulates time-based management interventions that modulate latent states.
 * Events are scheduled per-cow using Poisson sampling + seasonal weighting.
 * 
 * Each event returns modifiers consumed by CowPhysiologyEngine.evolve():
 *   - susceptibilityMod:  multiplier on infection susceptibility
 *   - recoveryMod:        multiplier on γ (recovery rate)
 *   - stressSpike:        additive stress boost
 *   - transmissionMod:    multiplier on R₀
 */

'use strict';

// ── Helpers ─────────────────────────────────────────────────────────────────

function getRandomInt(min, max) {
    return Math.floor(Math.random() * (max - min + 1)) + min;
}

function poissonSample(lambda) {
    if (lambda <= 0) return 0;
    const L = Math.exp(-lambda);
    let k = 0, p = 1;
    do { k++; p *= Math.random(); } while (p > L);
    return k - 1;
}

// ═════════════════════════════════════════════════════════════════════════════

/**
 * Management event types and their simulation parameters.
 */
const EVENT_TEMPLATES = {
    vaccination: {
        annualRate: 1.5,         // 1-2 per year
        durationDays: 1,         // Injection is instant
        effectDurationDays: 120, // Immunity wanes over 120 days
        effectFn: (daysSinceEvent) => ({
            susceptibilityMod: 1 - 0.6 * Math.exp(-daysSinceEvent / 120),
            recoveryMod: 1.0,
            stressSpike: daysSinceEvent < 2 ? 0.05 : 0, // Mild stress from injection
            transmissionMod: 1.0
        })
    },

    antibiotic: {
        annualRate: 0, // Triggered by infections, not scheduled
        durationDays: 7,
        effectDurationDays: 10,
        effectFn: (daysSinceEvent, durationDays) => {
            const active = daysSinceEvent <= durationDays;
            return {
                susceptibilityMod: 1.0,
                recoveryMod: active ? 1.5 : 1.0, // 50% boost while active
                stressSpike: 0,
                transmissionMod: 1.0
            };
        }
    },

    feedChange: {
        annualRate: 3,
        durationDays: 5,     // 3-7 days
        effectDurationDays: 7,
        effectFn: (daysSinceEvent, durationDays) => {
            const active = daysSinceEvent <= durationDays;
            return {
                susceptibilityMod: 1.0,
                recoveryMod: 1.0,
                stressSpike: active ? 0.15 * Math.exp(-daysSinceEvent / 3) : 0,
                transmissionMod: 1.0
            };
        }
    },

    transport: {
        annualRate: 0.5,
        durationDays: 2,     // 1-3 days
        effectDurationDays: 5,
        effectFn: (daysSinceEvent, durationDays) => {
            const active = daysSinceEvent <= durationDays;
            return {
                susceptibilityMod: active ? 1.3 : 1.0, // Transport stress → more susceptible
                recoveryMod: 1.0,
                stressSpike: active ? 0.30 * Math.exp(-daysSinceEvent / 1.5) : 0,
                transmissionMod: 1.0
            };
        }
    },

    groupMixing: {
        annualRate: 2,
        durationDays: 10,    // 7-14 days
        effectDurationDays: 14,
        effectFn: (daysSinceEvent, durationDays) => {
            const active = daysSinceEvent <= durationDays;
            return {
                susceptibilityMod: active ? 1.15 : 1.0,
                recoveryMod: 1.0,
                stressSpike: active ? 0.08 : 0,
                transmissionMod: active ? 1.4 : 1.0 // 40% higher R₀ during mixing
            };
        }
    },

    deworming: {
        annualRate: 1.5,
        durationDays: 1,
        effectDurationDays: 30,
        effectFn: (daysSinceEvent) => ({
            susceptibilityMod: daysSinceEvent <= 30 ? 0.85 : 1.0, // 15% less susceptible
            recoveryMod: 1.0,
            stressSpike: daysSinceEvent < 1 ? 0.03 : 0,
            transmissionMod: 1.0
        })
    },

    ventilationBoost: {
        annualRate: 1,           // Seasonal activation
        durationDays: 90,       // Active for summer
        effectDurationDays: 90,
        seasonal: true,          // Concentrated in summer
        effectFn: (daysSinceEvent, durationDays) => {
            const active = daysSinceEvent <= durationDays;
            return {
                susceptibilityMod: 1.0,
                recoveryMod: 1.0,
                stressSpike: active ? -0.10 : 0, // REDUCES stress
                transmissionMod: 1.0
            };
        }
    }
};

// ═════════════════════════════════════════════════════════════════════════════

class ManagementEventSimulator {
    /**
     * @param {number} totalDays - Total simulation days
     * @param {number} ticksPerDay - Ticks per day (288 for 5-min)
     */
    constructor(totalDays, ticksPerDay = 288) {
        this.totalDays = totalDays;
        this.ticksPerDay = ticksPerDay;
    }

    /**
     * Generate complete management timeline for one cow.
     * @returns {Array} Sorted list of management events
     */
    generateTimeline() {
        const events = [];

        for (const [eventType, template] of Object.entries(EVENT_TEMPLATES)) {
            if (template.annualRate <= 0) continue;

            const yearScale = this.totalDays / 365;
            const count = poissonSample(template.annualRate * yearScale);

            for (let i = 0; i < count; i++) {
                let startDay;
                if (template.seasonal) {
                    // Summer-concentrated: day 120-270
                    startDay = 120 + Math.random() * 150;
                    if (startDay >= this.totalDays) continue;
                } else {
                    startDay = Math.random() * this.totalDays * 0.95;
                }

                const durationDays = template.durationDays +
                    getRandomInt(0, Math.floor(template.durationDays * 0.5));

                events.push({
                    type: eventType,
                    startDay,
                    durationDays,
                    effectDurationDays: template.effectDurationDays,
                    startTick: Math.floor(startDay * this.ticksPerDay)
                });
            }
        }

        events.sort((a, b) => a.startDay - b.startDay);
        return events;
    }

    /**
     * Trigger antibiotic treatment when infection is detected.
     * Called by CowPhysiologyEngine when infection is seeded.
     * 
     * IMPORTANT: Treatment starts 1-3 days after infection detection
     * (realistic clinical delay). This allows infection to grow, then
     * antibiotics bring it down, creating the maxSev=3→minSev=0 pattern.
     * 
     * @param {Array} timeline - Cow's management timeline
     * @param {number} currentDay - Current simulation day (day of infection)
     * @param {number} probability - Probability of treatment (0.7 = 70%)
     */
    triggerAntibiotic(timeline, currentDay, probability = 0.7) {
        if (Math.random() > probability) return;

        // Clinical detection delay: 1-3 days after infection onset
        const detectionDelay = 1 + Math.random() * 2;
        const treatmentStartDay = currentDay + detectionDelay;

        const durationDays = getRandomInt(5, 10);
        timeline.push({
            type: 'antibiotic',
            startDay: treatmentStartDay,
            durationDays,
            effectDurationDays: durationDays + 3,
            startTick: Math.floor(treatmentStartDay * this.ticksPerDay)
        });
    }

    /**
     * Get combined management modifiers for a cow at a specific tick.
     * All active events are composed multiplicatively.
     * 
     * @param {Array} timeline - Cow's management timeline
     * @param {number} currentTick - Current simulation tick
     * @returns {Object} Combined modifiers + active flags
     */
    getActiveModifiers(timeline, currentTick) {
        const currentDay = currentTick / this.ticksPerDay;

        let susceptibilityMod = 1.0;
        let recoveryMod = 1.0;
        let stressSpike = 0;
        let transmissionMod = 1.0;

        // Active flags for schema output
        const activeFlags = {
            vaccinationActive: false,
            antibioticActive: false,
            transportActive: false,
            feedChangeActive: false,
            groupMixingActive: false,
            ventilationBoost: false
        };

        const flagMap = {
            'vaccination': 'vaccinationActive',
            'antibiotic': 'antibioticActive',
            'transport': 'transportActive',
            'feedChange': 'feedChangeActive',
            'groupMixing': 'groupMixingActive',
            'ventilationBoost': 'ventilationBoost'
        };

        for (const event of timeline) {
            const daysSince = currentDay - event.startDay;
            if (daysSince < 0) continue; // Not started yet
            if (daysSince > event.effectDurationDays) continue; // Effect expired

            const template = EVENT_TEMPLATES[event.type];
            if (!template) continue;

            const mods = template.effectFn(daysSince, event.durationDays);

            // Compose multiplicatively (except stress which is additive)
            susceptibilityMod *= mods.susceptibilityMod;
            recoveryMod *= mods.recoveryMod;
            stressSpike += mods.stressSpike;
            transmissionMod *= mods.transmissionMod;

            // Set active flag
            if (flagMap[event.type] && daysSince <= event.durationDays) {
                activeFlags[flagMap[event.type]] = true;
            }
        }

        return {
            susceptibilityMod: Math.max(0.1, susceptibilityMod), // Never fully immune
            recoveryMod: Math.max(0.5, recoveryMod),
            stressSpike: Math.max(-0.2, Math.min(0.5, stressSpike)),
            transmissionMod: Math.max(0.5, transmissionMod),
            ...activeFlags
        };
    }
}

module.exports = ManagementEventSimulator;
