/**
 * TrainingEventV4 — Contextual Causal Twin Schema
 * 
 * GoMata Digital Twin v4
 * 
 * Separate collection: trainingevents_v4
 * Adds: production signals, management events, expanded animalProfile
 * 
 * 12 Sections:
 *   1. Multi-Tenant Context
 *   2. Temporal Context + Versioning
 *   3. Static Animal Profile (expanded with calving, milk, weight)
 *   4. Raw Sensor Signals
 *   5. Production Signals (milk, conductivity, weight, feed, water)
 *   6. Environmental Signals
 *   7. Management Events (active flags)
 *   8. Window-Based Engineered Features
 *   9. Latent Biological States (NEVER model input)
 *  10. Training Labels (multi-task + forecast)
 *  11. Intervention Context
 *  12. Data Governance
 */

'use strict';

const mongoose = require('mongoose');
const { Schema } = mongoose;

// ═════════════════════════════════════════════════════════════════════════════

const trainingEventV4Schema = new Schema({

    // ─────────────────────────────────────────────────────────────────────
    // 1️⃣  MULTI-TENANT CONTEXT
    // ─────────────────────────────────────────────────────────────────────
    tenantId: { type: Schema.Types.ObjectId, required: true, ref: 'User' },
    farmId: { type: Schema.Types.ObjectId, default: null, ref: 'Farm' },
    zoneId: { type: Schema.Types.ObjectId, default: null },
    animalId: { type: Schema.Types.ObjectId, required: true, ref: 'Livestock' },

    // ─────────────────────────────────────────────────────────────────────
    // 2️⃣  TEMPORAL CONTEXT + VERSIONING
    // ─────────────────────────────────────────────────────────────────────
    timestamp: { type: Date, required: true, default: Date.now },
    simulationVersion: { type: String, default: 'digital_twin_v4' },
    featureVersion: { type: String, default: 'v4_contextual' },
    episodeId: { type: String, default: null },
    episodeDayIndex: { type: Number, default: 0 },
    timeSinceEpisodeStart: { type: Number, default: 0 },

    // ─────────────────────────────────────────────────────────────────────
    // 3️⃣  EXPANDED ANIMAL PROFILE
    // ─────────────────────────────────────────────────────────────────────
    animalProfile: {
        parity: { type: Number, default: 0 },
        lactationStage: { type: String, enum: ['dry', 'early', 'mid', 'late'], default: 'mid' },
        calvingDate: { type: Date, default: null },
        bodyConditionScore: { type: Number, default: 3.0 },
        previousDiseaseCount: { type: Number, default: 0 },
        geneticHeatTolerance: { type: String, enum: ['low', 'medium', 'high'], default: 'medium' },
        baselineMilkYield: { type: Number, default: 18 },     // L/day
        baselineWeight: { type: Number, default: 450 },        // kg
        breed: { type: String, default: 'default' },
        age: { type: Number, default: 4 }                      // years
    },

    // ─────────────────────────────────────────────────────────────────────
    // 4️⃣  RAW SENSOR SIGNALS
    // ─────────────────────────────────────────────────────────────────────
    signals: {
        temperature_C: { type: Number },
        heartRate_bpm: { type: Number },
        respiration_bpm: { type: Number },
        activity_index: { type: Number },
        rumination_min: { type: Number },
        lying_min: { type: Number },
        gps: {
            lat: { type: Number },
            lon: { type: Number }
        }
    },

    // ─────────────────────────────────────────────────────────────────────
    // 5️⃣  PRODUCTION SIGNALS
    // ─────────────────────────────────────────────────────────────────────
    production: {
        milkYield: { type: Number, default: 0 },             // L/day
        milkConductivity: { type: Number, default: 5.0 },    // mS/cm
        bodyWeight: { type: Number, default: 450 },           // kg
        feedIntake: { type: Number, default: 0 },             // kg DM/day rate
        waterIntake: { type: Number, default: 0 }             // L/day rate
    },

    // ─────────────────────────────────────────────────────────────────────
    // 6️⃣  ENVIRONMENTAL SIGNALS
    // ─────────────────────────────────────────────────────────────────────
    environment: {
        ambientTemp_C: { type: Number },
        humidity_pct: { type: Number },
        thi: { type: Number },
        ammonia_ppm: { type: Number },
        airflow_rate: { type: Number },
        stockingDensity_raw: { type: Number },
        stockingDensity_normalized: { type: Number },
        dayOfYear: { type: Number }
    },

    // ─────────────────────────────────────────────────────────────────────
    // 7️⃣  MANAGEMENT EVENTS (active flags at this tick)
    // ─────────────────────────────────────────────────────────────────────
    managementEvents: {
        vaccinationActive: { type: Boolean, default: false },
        antibioticActive: { type: Boolean, default: false },
        transportActive: { type: Boolean, default: false },
        feedChangeActive: { type: Boolean, default: false },
        groupMixingActive: { type: Boolean, default: false },
        ventilationBoost: { type: Boolean, default: false }
    },

    // ─────────────────────────────────────────────────────────────────────
    // 8️⃣  WINDOW-BASED ENGINEERED FEATURES
    // ─────────────────────────────────────────────────────────────────────
    features: {
        temp_current: { type: Number },
        temp_6h_avg: { type: Number },
        temp_6h_std: { type: Number },
        temp_6h_slope: { type: Number },
        temp_zscore: { type: Number },
        hr_current: { type: Number },
        hr_6h_avg: { type: Number },
        hr_6h_std: { type: Number },
        hr_6h_slope: { type: Number },
        hr_zscore: { type: Number },
        activity_current: { type: Number },
        activity_6h_avg: { type: Number },
        activity_6h_std: { type: Number },
        activity_6h_slope: { type: Number },
        activity_ratio: { type: Number },
        rumination_drop: { type: Number },
        autocorrelation_temp: { type: Number },
        coefficient_variation_temp: { type: Number },
        heat_stress_index: { type: Number },
        composite_stress_index: { type: Number },
        heat_component: { type: Number },
        air_quality_component: { type: Number },
        crowding_component: { type: Number },
        ventilation_component: { type: Number },
        // v4 production features
        milk_deviation: { type: Number },         // current / baseline
        feed_deviation: { type: Number },
        weight_deviation: { type: Number },
        conductivity_deviation: { type: Number }
    },

    // ─────────────────────────────────────────────────────────────────────
    // 9️⃣  LATENT BIOLOGICAL STATES (NEVER model input)
    // ─────────────────────────────────────────────────────────────────────
    hiddenState: {
        infectionLoad: { type: Number, default: 0 },
        stressLoad: { type: Number, default: 0 },
        immuneResponse: { type: Number, default: 0 },
        compensationCapacity: { type: Number, default: 1 },
        fatigue: { type: Number, default: 0 },
        compensationCollapse: { type: Boolean, default: false }
    },

    // ─────────────────────────────────────────────────────────────────────
    // 🔟  TRAINING LABELS
    // ─────────────────────────────────────────────────────────────────────
    labels: {
        diseaseBinary: { type: Number, enum: [0, 1], default: 0 },
        infectionBinary: { type: Number, enum: [0, 1], default: 0 },
        stressBinary: { type: Number, enum: [0, 1], default: 0 },
        mixedStateBinary: { type: Number, enum: [0, 1], default: 0 },
        severityLevel: { type: Number, enum: [0, 1, 2, 3], default: 0 },
        episodePhase: {
            type: String, enum: [
                'healthy',
                'incubation', 'onset', 'peak', 'plateau', 'recovery', 'resolved',
                'stress_onset', 'stress_peak', 'stress_recovery',
                'mixed_onset', 'mixed_peak', 'mixed_recovery'
            ], default: 'healthy'
        },
        diseaseType: {
            type: String, enum: [
                'none', 'brd', 'mastitis', 'laminitis', 'generic',
                'transmitted', 'heat_stress', 'mixed_infection_stress'
            ], default: 'none'
        },
        infection_in_24h: { type: Number, enum: [0, 1], default: 0 },
        stress_in_24h: { type: Number, enum: [0, 1], default: 0 }
    },

    // ─────────────────────────────────────────────────────────────────────
    // 1️⃣1️⃣  INTERVENTION CONTEXT
    // ─────────────────────────────────────────────────────────────────────
    interventionContext: {
        vaccinationActive: { type: Boolean, default: false },
        isolationActive: { type: Boolean, default: false },
        ventilationBoost: { type: Boolean, default: false },
        antibioticActive: { type: Boolean, default: false }
    },

    // ─────────────────────────────────────────────────────────────────────
    // 1️⃣2️⃣  DATA GOVERNANCE
    // ─────────────────────────────────────────────────────────────────────
    source: { type: String, required: true, default: 'digital_twin_v4' }

}, {
    timestamps: true,
    collection: 'trainingevents_v4',
    strict: false
});


// ═════════════════════════════════════════════════════════════════════════════
// INDEXES
// ═════════════════════════════════════════════════════════════════════════════

trainingEventV4Schema.index({ tenantId: 1, timestamp: -1 });
trainingEventV4Schema.index({ animalId: 1, timestamp: -1 });
trainingEventV4Schema.index({ simulationVersion: 1 });
trainingEventV4Schema.index({ episodeId: 1 });
trainingEventV4Schema.index({ source: 1, timestamp: -1 });
trainingEventV4Schema.index({ 'labels.episodePhase': 1, source: 1 });
trainingEventV4Schema.index({ 'labels.diseaseType': 1, source: 1 });
trainingEventV4Schema.index({ tenantId: 1, animalId: 1, timestamp: -1 });


// ═════════════════════════════════════════════════════════════════════════════
// STATICS — ML Export
// ═════════════════════════════════════════════════════════════════════════════

trainingEventV4Schema.statics.ML_EXPORT_PROJECTION = {
    features: 1,
    labels: 1,
    animalProfile: 1,
    production: 1,
    environment: 1,
    signals: 1,
    managementEvents: 1,
    interventionContext: 1,
    episodeId: 1,
    episodeDayIndex: 1,
    timestamp: 1,
    // Explicitly EXCLUDE hiddenState
    hiddenState: 0
};

trainingEventV4Schema.statics.exportForTraining = function (tenantId, opts = {}) {
    const filter = {
        tenantId,
        source: opts.source || 'digital_twin_v4',
    };
    if (opts.since) filter.timestamp = { $gte: opts.since };
    if (opts.until) filter.timestamp = { ...filter.timestamp, $lte: opts.until };

    return this.find(filter)
        .select(this.ML_EXPORT_PROJECTION)
        .sort({ timestamp: 1 })
        .lean()
        .cursor();
};


module.exports = mongoose.model('TrainingEventV4', trainingEventV4Schema);
