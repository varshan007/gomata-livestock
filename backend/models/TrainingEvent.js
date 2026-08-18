/**
 * TrainingEvent — Production-Grade Biological State Event Store
 * 
 * GoMata Digital Twin v3 Schema
 * 
 * Each document = 1 training snapshot per cow per timestamp.
 * Designed for 50M+ rows, multi-task ML training, zero data leakage.
 * 
 * 10 Sections:
 *   1. Multi-Tenant Context
 *   2. Temporal Context + Versioning
 *   3. Static Animal Profile (snapshot — no joins at export)
 *   4. Raw Sensor Signals
 *   5. Environmental Signals
 *   6. Window-Based Engineered Features (ML input)
 *   7. Latent Biological States (NEVER model input)
 *   8. Training Labels (multi-task targets)
 *   9. Intervention Context (decision intelligence)
 *  10. Data Governance
 */

'use strict';

const mongoose = require('mongoose');
const { Schema } = mongoose;

// ═════════════════════════════════════════════════════════════════════════════

const trainingEventSchema = new Schema({

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
    simulationVersion: { type: String, default: 'digital_twin_v3' },
    featureVersion: { type: String, default: 'v4_windowed' },
    episodeId: { type: String, default: null },       // cowId_ep<N> or null
    episodeDayIndex: { type: Number, default: 0 },          // day within episode
    timeSinceEpisodeStart: { type: Number, default: 0 },       // minutes

    // ─────────────────────────────────────────────────────────────────────
    // 3️⃣  STATIC ANIMAL PROFILE (snapshot for ML export — no joins)
    // ─────────────────────────────────────────────────────────────────────
    animalProfile: {
        parity: { type: Number, default: 0 },
        lactationStage: { type: String, enum: ['dry', 'early', 'mid', 'late'], default: 'dry' },
        bodyConditionScore: { type: Number, default: 3.0 },   // 1.0–5.0
        geneticHeatTolerance: { type: String, enum: ['low', 'medium', 'high'], default: 'medium' },
        previousDiseaseCount: { type: Number, default: 0 }
    },

    // ─────────────────────────────────────────────────────────────────────
    // 4️⃣  RAW SENSOR SIGNALS
    // ─────────────────────────────────────────────────────────────────────
    signals: {
        temperature_C: { type: Number },
        heartRate_bpm: { type: Number },
        respiration_bpm: { type: Number },
        activity_index: { type: Number },    // 0–1 normalized
        rumination_min: { type: Number },
        lying_min: { type: Number },
        gps: {
            lat: { type: Number },
            lon: { type: Number }
        }
    },

    // ─────────────────────────────────────────────────────────────────────
    // 5️⃣  ENVIRONMENTAL SIGNALS
    // ─────────────────────────────────────────────────────────────────────
    environment: {
        ambientTemp_C: { type: Number },
        humidity_pct: { type: Number },
        thi: { type: Number },
        ammonia_ppm: { type: Number },
        airflow_rate: { type: Number },
        stocking_density_raw: { type: Number },        // cows per m² (physical)
        stocking_density_normalized: { type: Number }   // 0–1 (3.0 cows/m² = 1.0)
    },

    // ─────────────────────────────────────────────────────────────────────
    // 6️⃣  WINDOW-BASED ENGINEERED FEATURES (ML input)
    // ─────────────────────────────────────────────────────────────────────
    features: {
        // Temperature window
        temp_current: { type: Number },
        temp_6h_avg: { type: Number },
        temp_6h_std: { type: Number },
        temp_6h_slope: { type: Number },
        temp_zscore: { type: Number },

        // Heart rate window
        hr_current: { type: Number },
        hr_6h_avg: { type: Number },
        hr_6h_std: { type: Number },
        hr_6h_slope: { type: Number },
        hr_zscore: { type: Number },

        // Activity window
        activity_current: { type: Number },   // 0–1
        activity_6h_avg: { type: Number },
        activity_6h_std: { type: Number },
        activity_6h_slope: { type: Number },
        activity_ratio: { type: Number },

        // Rumination
        rumination_drop: { type: Number },

        // Stability metrics
        autocorrelation_temp: { type: Number },   // 6h-window lag-1
        coefficient_variation_temp: { type: Number },

        // Stress indicators (higher = worse, 0 = none, 1 = extreme)
        heat_stress_index: { type: Number },   // Observable, f(THI, temp_dev, resp_elev)
        composite_stress_index: { type: Number },   // 0.4*HSI + 0.3*ruminDrop + 0.3*(1-activity)
        // Raw stress components (let ML learn weighting)
        heat_component: { type: Number },          // sigmoid((THI-72)/8)
        air_quality_component: { type: Number },   // normalized ammonia
        crowding_component: { type: Number },      // density_normalized
        ventilation_component: { type: Number }    // airflow_normalized
    },

    // ─────────────────────────────────────────────────────────────────────
    // 7️⃣  LATENT BIOLOGICAL STATES (NEVER used as model input)
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
    // 8️⃣  TRAINING LABELS (multi-task targets)
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
        diseaseType: { type: String, enum: ['none', 'brd', 'mastitis', 'laminitis', 'generic', 'transmitted', 'heat_stress', 'mixed_infection_stress'], default: 'none' },
        // Forecast labels (look-ahead from schedule, NOT from hidden state)
        infection_in_24h: { type: Number, enum: [0, 1], default: 0 },
        stress_in_24h: { type: Number, enum: [0, 1], default: 0 }
    },

    // ─────────────────────────────────────────────────────────────────────
    // 9️⃣  INTERVENTION CONTEXT (decision intelligence training)
    // ─────────────────────────────────────────────────────────────────────
    interventionContext: {
        vaccinationActive: { type: Boolean, default: false },
        isolationActive: { type: Boolean, default: false },
        ventilationBoost: { type: Boolean, default: false },
        antibioticActive: { type: Boolean, default: false }
    },

    // ─────────────────────────────────────────────────────────────────────
    // 🔟  DATA GOVERNANCE
    // ─────────────────────────────────────────────────────────────────────
    source: { type: String, required: true, default: 'digital_twin_v3' },

    // ── Legacy compatibility fields (v1 writes these) ────────────────
    eventType: { type: String },
    label: { type: Number },
    startedAt: { type: Date },
    endedAt: { type: Date },
    metadata: { type: Schema.Types.Mixed }

}, {
    timestamps: true,       // createdAt, updatedAt
    collection: 'trainingevents',  // Keep same collection name for backward compat
    strict: false           // Allow v1 docs with different shape to coexist
});


// ═════════════════════════════════════════════════════════════════════════════
// INDEXES — Critical for 50M+ row scale
// ═════════════════════════════════════════════════════════════════════════════

// Primary query path: tenant + time range
trainingEventSchema.index({ tenantId: 1, timestamp: -1 });

// Per-animal time series
trainingEventSchema.index({ animalId: 1, timestamp: -1 });

// Version filtering (for retraining on specific simulation runs)
trainingEventSchema.index({ simulationVersion: 1 });

// Episode queries (fetch all events for a disease episode)
trainingEventSchema.index({ episodeId: 1 });

// Source filtering (v1 vs v3 separation)
trainingEventSchema.index({ source: 1, timestamp: -1 });

// Legacy v1 indexes (kept for backward compat)
trainingEventSchema.index({ animalId: 1, startedAt: -1 });
trainingEventSchema.index({ tenantId: 1, startedAt: -1 });


// ═════════════════════════════════════════════════════════════════════════════
// STATICS — ML Export helpers
// ═════════════════════════════════════════════════════════════════════════════

/**
 * Export projection for ML training — excludes hidden state to prevent leakage.
 */
trainingEventSchema.statics.ML_EXPORT_PROJECTION = {
    features: 1,
    labels: 1,
    animalProfile: 1,
    environment: 1,
    signals: 1,
    interventionContext: 1,
    episodeId: 1,
    episodeDayIndex: 1,
    timestamp: 1,
    // Explicitly EXCLUDE hiddenState
    hiddenState: 0
};

/**
 * Get training data for a tenant, filtered by simulation version.
 * Returns cursor for streaming large datasets.
 */
trainingEventSchema.statics.exportForTraining = function (tenantId, opts = {}) {
    const filter = {
        tenantId,
        source: opts.source || 'digital_twin_v3',
    };
    if (opts.since) filter.timestamp = { $gte: opts.since };
    if (opts.until) filter.timestamp = { ...filter.timestamp, $lte: opts.until };

    return this.find(filter)
        .select(this.ML_EXPORT_PROJECTION)
        .sort({ timestamp: 1 })
        .lean()
        .cursor();
};


module.exports = mongoose.model('TrainingEvent', trainingEventSchema);
