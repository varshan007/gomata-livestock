/**
 * HealthAgent — Production-Grade ML Health Prediction Agent
 * 
 * Reads precomputed features from Redis, calls the ML service for disease
 * prediction, caches risk scores, creates deduplicated alerts, and publishes
 * events to the event bus.
 * 
 * Safety mechanisms:
 *   1. Concurrency limiter  — max N parallel ML calls (prevents overload)
 *   2. Retry + backoff      — 3 attempts with exponential backoff
 *   3. Circuit breaker      — skips ML after 10 consecutive failures
 *   4. Consecutive detection — requires 2+ predictions above threshold
 *   5. Alert deduplication  — checks MongoDB for recent unresolved alerts
 *   6. JSON safety          — guards against parse errors, NaN, missing fields
 *   7. Structured metrics   — exposes getMetrics() for monitoring
 * 
 * Designed for:
 *   - 10,000+ animals
 *   - Multi-tenant isolation
 *   - Horizontal scaling (multiple Node instances)
 *   - Orchestrator-managed scheduling
 */

const axios = require('axios');
const LivestockMaster = require('../../../models/LivestockMaster');
const Alert = require('../../../models/Alert');
const logger = require('../../../utils/logger');
const path = require('path');
const fs = require('fs');

// ── Constants ───────────────────────────────────────────────────────────────

const LOG_SERVICE = 'health_agent';

const DEFAULTS = {
    ML_SERVICE_URL: process.env.ML_SERVICE_URL || 'http://localhost:8001',
    CONCURRENCY_LIMIT: 15,           // max parallel ML calls
    MAX_RETRIES: 3,
    BASE_BACKOFF_MS: 500,          // 500ms → 1s → 2s
    RISK_SCORE_TTL: 300,          // 5 minutes
    HEALTH_STATE_TTL: 600,          // 10 minutes — consecutive detection state
    CIRCUIT_BREAKER_THRESHOLD: 10,        // consecutive failures to trip
    CIRCUIT_BREAKER_COOLDOWN: 60_000,    // 60s cooldown
    ALERT_DEDUP_HOURS: 2,            // don't create duplicate alerts within 2h
    ML_TIMEOUT_MS: 10_000,       // 10s per ML call
    OPERATING_MODE: 'default',    // default | sensitive
};

// ── Concurrency Limiter ─────────────────────────────────────────────────────
// Lightweight p-limit replacement — no external dependency

function createLimiter(concurrency) {
    let active = 0;
    const queue = [];

    function next() {
        if (queue.length === 0 || active >= concurrency) return;
        active++;
        const { fn, resolve, reject } = queue.shift();
        fn().then(resolve, reject).finally(() => {
            active--;
            next();
        });
    }

    return function limit(fn) {
        return new Promise((resolve, reject) => {
            queue.push({ fn, resolve, reject });
            next();
        });
    };
}

// ── Sleep Helper ────────────────────────────────────────────────────────────

function sleep(ms) {
    return new Promise(resolve => setTimeout(resolve, ms));
}

// ── JSON Safety Helper ──────────────────────────────────────────────────────

function safeParseJSON(str, fallback = null) {
    if (!str || typeof str !== 'string') return fallback;
    try {
        const parsed = JSON.parse(str);
        if (typeof parsed !== 'object' || parsed === null) return fallback;
        return parsed;
    } catch {
        return fallback;
    }
}

// ── Feature Validation ──────────────────────────────────────────────────────

const REQUIRED_FEATURES = [
    'temp_current', 'temp_6h_avg', 'temp_6h_std', 'temp_6h_slope',
    'temp_max_6h', 'temp_min_6h', 'temp_range_6h',
    'hr_current', 'hr_6h_avg', 'hr_6h_std', 'hr_6h_slope',
    'activity_current', 'activity_6h_avg', 'activity_6h_std', 'activity_6h_slope',
    'temp_ratio', 'hr_ratio', 'activity_ratio',
    'temp_zscore', 'hr_zscore',
    'temp_recent_vs_baseline', 'hr_recent_vs_baseline', 'activity_recent_vs_baseline'
];

function validateFeatures(features) {
    for (const key of REQUIRED_FEATURES) {
        const val = features[key];
        if (val === undefined || val === null || typeof val !== 'number' || Number.isNaN(val)) {
            return false;
        }
    }
    return true;
}

// ═════════════════════════════════════════════════════════════════════════════
// HealthAgent
// ═════════════════════════════════════════════════════════════════════════════

class HealthAgent {
    /**
     * @param {Object} options
     * @param {Object} options.redis          - ioredis connection
     * @param {Object} [options.eventBus]     - RedisEventBus for publishing events
     * @param {Object} [options.httpClient]   - axios-compatible HTTP client (DI)
     * @param {Object} [options.logger]       - pino-compatible logger (DI)
     * @param {string} [options.mlServiceUrl] - ML service base URL
     * @param {number} [options.concurrency]  - max parallel ML calls
     * @param {string} [options.operatingMode] - 'default' or 'sensitive'
     * @param {string} [options.configPath]   - path to model_config.json
     */
    constructor(options = {}) {
        // ── Dependency injection ─────────────────────────────────────
        this.redis = options.redis;
        this.eventBus = options.eventBus || null;
        this.http = options.httpClient || axios;
        this.log = options.logger || logger;
        this.mlUrl = options.mlServiceUrl || DEFAULTS.ML_SERVICE_URL;
        this.mode = options.operatingMode || DEFAULTS.OPERATING_MODE;

        // ── Concurrency limiter ──────────────────────────────────────
        const concurrency = options.concurrency || DEFAULTS.CONCURRENCY_LIMIT;
        this._limit = createLimiter(concurrency);

        // ── Circuit breaker state ────────────────────────────────────
        this._consecutiveMLFailures = 0;
        this._circuitOpenUntil = 0;    // timestamp when circuit re-closes

        // ── Metrics ──────────────────────────────────────────────────
        this._metrics = {
            totalRuns: 0,
            animalsProcessed: 0,
            animalsSkipped: 0,
            alertsCreated: 0,
            mlCalls: 0,
            mlFailures: 0,
            mlLatencySum: 0,
            featureMissing: 0,
            featureInvalid: 0,
            circuitBroken: 0,
            consecutiveBlocked: 0,
            duplicateAlerts: 0,
        };

        // ── Load model config ────────────────────────────────────────
        this._threshold = DEFAULTS.OPERATING_MODE === 'sensitive' ? 0.293 : 0.5;
        this._loadModelConfig(options.configPath);

        this.log.info({
            service: LOG_SERVICE,
            action: 'AGENT_CREATED',
            mlUrl: this.mlUrl,
            mode: this.mode,
            threshold: this._threshold,
            concurrency
        }, `[HealthAgent] Created — mode: ${this.mode}, threshold: ${this._threshold}`);
    }

    // ─────────────────────────────────────────────────────────────────────────
    // MODEL CONFIG LOADER
    // ─────────────────────────────────────────────────────────────────────────

    /**
     * Load threshold from model_config.json.
     * Falls back to hardcoded defaults if file is missing or malformed.
     */
    _loadModelConfig(configPath) {
        const defaultPath = path.resolve(__dirname, '../../../../ml_service/models/cattle/model_config.json');
        const filePath = configPath || defaultPath;

        try {
            if (fs.existsSync(filePath)) {
                const raw = fs.readFileSync(filePath, 'utf8');
                const config = JSON.parse(raw);

                if (this.mode === 'sensitive' && typeof config.threshold_sensitive === 'number') {
                    this._threshold = config.threshold_sensitive;
                } else if (typeof config.threshold_default === 'number') {
                    this._threshold = config.threshold_default;
                }

                this._modelVersion = config.model_version || 'unknown';
                this._featureCount = config.feature_count || 23;

                this.log.info({
                    service: LOG_SERVICE,
                    action: 'CONFIG_LOADED',
                    threshold: this._threshold,
                    modelVersion: this._modelVersion,
                    mode: this.mode
                }, `[HealthAgent] Config loaded — threshold: ${this._threshold}, model: ${this._modelVersion}`);
            } else {
                this.log.warn({
                    service: LOG_SERVICE,
                    action: 'CONFIG_MISSING',
                    path: filePath
                }, `[HealthAgent] model_config.json not found — using defaults`);
            }
        } catch (err) {
            this.log.error({
                service: LOG_SERVICE,
                action: 'CONFIG_ERROR',
                error: err.message
            }, `[HealthAgent] Failed to load model config — using defaults`);
        }
    }

    // ─────────────────────────────────────────────────────────────────────────
    // CORE RUN — Entry point called by AgentOrchestrator
    // ─────────────────────────────────────────────────────────────────────────

    /**
     * Run health analysis for a specific tenant or all tenants.
     * 
     * @param {string|null} tenantId  - userId to scope animals, or null for all
     */
    async run(tenantId = null) {
        this._metrics.totalRuns++;
        const startTime = Date.now();

        // 1. Query active animals
        const query = tenantId ? { userId: tenantId } : {};
        let animals;
        try {
            animals = await LivestockMaster.find(query)
                .select('_id userId device_id livestock_id name breed farm_name zone_name health_status')
                .lean();
        } catch (err) {
            this.log.error({
                service: LOG_SERVICE,
                action: 'ANIMAL_QUERY_ERROR',
                tenantId,
                error: err.message
            }, `[HealthAgent] Failed to query animals`);
            return;
        }

        if (!animals || animals.length === 0) {
            this.log.info({
                service: LOG_SERVICE,
                action: 'NO_ANIMALS',
                tenantId
            }, `[HealthAgent] No animals found for tenant ${tenantId || 'ALL'}`);
            return;
        }

        this.log.info({
            service: LOG_SERVICE,
            action: 'RUN_STARTED',
            tenantId: tenantId || 'ALL',
            animalCount: animals.length
        }, `[HealthAgent] Processing ${animals.length} animals`);

        // 2. Process all animals with concurrency limiting
        const results = await Promise.allSettled(
            animals.map(animal => this._limit(() => this.analyzeAnimal(animal)))
        );

        // 3. Tally results
        let processed = 0, skipped = 0, errors = 0;
        for (const r of results) {
            if (r.status === 'fulfilled') {
                if (r.value === 'processed') processed++;
                else skipped++;
            } else {
                errors++;
            }
        }

        const durationMs = Date.now() - startTime;

        this.log.info({
            service: LOG_SERVICE,
            action: 'RUN_COMPLETE',
            tenantId: tenantId || 'ALL',
            processed,
            skipped,
            errors,
            durationMs,
            alertsCreated: this._metrics.alertsCreated,
            mlFailures: this._metrics.mlFailures
        }, `[HealthAgent] Run complete — ${processed} processed, ${skipped} skipped, ${errors} errors in ${durationMs}ms`);
    }

    // ─────────────────────────────────────────────────────────────────────────
    // PER-ANIMAL ANALYSIS
    // ─────────────────────────────────────────────────────────────────────────

    /**
     * Analyze a single animal: read features → call ML → cache → alert.
     * Each animal is fully isolated — one failure does not affect others.
     * 
     * @param   {Object} animal  - LivestockMaster document (lean)
     * @returns {string} 'processed' or 'skipped'
     */
    async analyzeAnimal(animal) {
        const animalId = animal._id.toString();
        const tenantId = animal.userId.toString();

        try {
            // ── 1. Read features from Redis ──────────────────────────
            const featureKey = `features:v3:${tenantId}:${animalId}`;
            const rawFeatures = await this.redis.get(featureKey);

            if (!rawFeatures) {
                this._metrics.featureMissing++;
                this._metrics.animalsSkipped++;
                return 'skipped';
            }

            const features = safeParseJSON(rawFeatures);
            if (!features) {
                this._metrics.featureInvalid++;
                this._metrics.animalsSkipped++;
                this.log.warn({
                    service: LOG_SERVICE,
                    action: 'FEATURE_PARSE_ERROR',
                    animalId
                }, `[HealthAgent] JSON parse failed for features of ${animalId}`);
                return 'skipped';
            }

            // Validate all 23 features are present and numeric
            if (!validateFeatures(features)) {
                this._metrics.featureInvalid++;
                this._metrics.animalsSkipped++;
                this.log.warn({
                    service: LOG_SERVICE,
                    action: 'FEATURE_VALIDATION_ERROR',
                    animalId
                }, `[HealthAgent] Feature validation failed for ${animalId}`);
                return 'skipped';
            }

            // ── 2. Call ML service ───────────────────────────────────
            const prediction = await this.callML(animalId, features);

            if (!prediction) {
                // ML call failed after retries or circuit breaker tripped
                this._metrics.animalsSkipped++;
                return 'skipped';
            }

            // ── 3. Cache risk score in Redis ─────────────────────────
            const riskPayload = {
                disease_prob: prediction.disease_prob,
                risk_score: prediction.risk_score,
                severity: prediction.severity,
                timestamp: new Date().toISOString(),
                model_version: this._modelVersion || 'unknown'
            };

            await this.redis.set(
                `healthRisk:${animalId}`,
                JSON.stringify(riskPayload),
                'EX', DEFAULTS.RISK_SCORE_TTL
            );

            // ── 4. Publish event ─────────────────────────────────────
            if (this.eventBus) {
                this.eventBus.emit('health:risk_updated', {
                    animalId,
                    tenantId,
                    riskScore: prediction.risk_score,
                    severity: prediction.severity,
                    diseaseProb: prediction.disease_prob
                });
            }

            // ── 5. Check alert trigger ───────────────────────────────
            if (prediction.disease_prob >= this._threshold) {
                const shouldAlert = await this.shouldTriggerAlert(animalId, prediction);

                if (shouldAlert) {
                    await this.createAlert(animal, prediction);
                }
            } else {
                // Below threshold — reset consecutive state
                await this.redis.del(`healthState:${animalId}`);
            }

            this._metrics.animalsProcessed++;
            return 'processed';

        } catch (err) {
            this.log.error({
                service: LOG_SERVICE,
                action: 'ANIMAL_ANALYSIS_ERROR',
                animalId,
                error: err.message
            }, `[HealthAgent] Unhandled error analyzing ${animalId}`);
            this._metrics.animalsSkipped++;
            return 'skipped';
        }
    }

    // ─────────────────────────────────────────────────────────────────────────
    // ML SERVICE CALL — with retry, backoff, circuit breaker
    // ─────────────────────────────────────────────────────────────────────────

    /**
     * Call the ML prediction service with retry and circuit breaker.
     * 
     * @param   {string} animalId
     * @param   {Object} features  - 23 precomputed features
     * @returns {Object|null}      - prediction response or null on failure
     */
    async callML(animalId, features) {
        // ── Circuit breaker check ────────────────────────────────────
        if (this._circuitOpenUntil > Date.now()) {
            this._metrics.circuitBroken++;
            return null; // Circuit is open — skip ML call silently
        }

        const payload = { animal_id: animalId };
        for (const key of REQUIRED_FEATURES) {
            payload[key] = features[key];
        }

        // ── Retry loop with exponential backoff ──────────────────────
        let lastError = null;

        for (let attempt = 0; attempt < DEFAULTS.MAX_RETRIES; attempt++) {
            if (attempt > 0) {
                const backoffMs = DEFAULTS.BASE_BACKOFF_MS * Math.pow(2, attempt - 1);
                await sleep(backoffMs);
            }

            const callStart = Date.now();
            this._metrics.mlCalls++;

            try {
                const response = await this.http.post(
                    `${this.mlUrl}/predict/health`,
                    payload,
                    { timeout: DEFAULTS.ML_TIMEOUT_MS }
                );

                const latencyMs = Date.now() - callStart;
                this._metrics.mlLatencySum += latencyMs;

                // ── Validate response ────────────────────────────────
                const data = response.data;
                if (!data || typeof data.risk_score !== 'number') {
                    throw new Error(`Invalid ML response: missing risk_score`);
                }

                // Normalize response to consistent format
                const prediction = {
                    disease_prob: typeof data.disease_prob === 'number'
                        ? data.disease_prob
                        : data.risk_score,
                    risk_score: data.risk_score,
                    severity: data.severity || this._computeSeverity(data.risk_score),
                    explanation: data.explanation || data.prediction || ''
                };

                // ── Reset circuit breaker on success ─────────────────
                this._consecutiveMLFailures = 0;

                return prediction;

            } catch (err) {
                lastError = err;
                const latencyMs = Date.now() - callStart;

                this.log.warn({
                    service: LOG_SERVICE,
                    action: 'ML_CALL_RETRY',
                    animalId,
                    attempt: attempt + 1,
                    maxRetries: DEFAULTS.MAX_RETRIES,
                    latencyMs,
                    error: err.message
                }, `[HealthAgent] ML call attempt ${attempt + 1} failed for ${animalId}`);
            }
        }

        // ── All retries exhausted ────────────────────────────────────
        this._metrics.mlFailures++;
        this._consecutiveMLFailures++;

        // Trip circuit breaker if threshold exceeded
        if (this._consecutiveMLFailures >= DEFAULTS.CIRCUIT_BREAKER_THRESHOLD) {
            this._circuitOpenUntil = Date.now() + DEFAULTS.CIRCUIT_BREAKER_COOLDOWN;
            this.log.error({
                service: LOG_SERVICE,
                action: 'CIRCUIT_BREAKER_OPEN',
                consecutiveFailures: this._consecutiveMLFailures,
                cooldownMs: DEFAULTS.CIRCUIT_BREAKER_COOLDOWN
            }, `[HealthAgent] Circuit breaker OPEN — ${this._consecutiveMLFailures} consecutive failures. Cooldown: 60s`);
        }

        this.log.error({
            service: LOG_SERVICE,
            action: 'ML_CALL_FAILED',
            animalId,
            error: lastError ? lastError.message : 'unknown'
        }, `[HealthAgent] ML call failed after all retries for ${animalId}`);

        return null;
    }

    // ─────────────────────────────────────────────────────────────────────────
    // CONSECUTIVE DETECTION — prevents single-spike false alerts
    // ─────────────────────────────────────────────────────────────────────────

    /**
     * Determine if an alert should be triggered based on:
     *   1. High severity → immediate alert
     *   2. 2+ consecutive above-threshold predictions → alert
     *   3. First prediction → store state, wait for confirmation
     * 
     * @param   {string} animalId
     * @param   {Object} prediction
     * @returns {boolean}
     */
    async shouldTriggerAlert(animalId, prediction) {
        // High severity always triggers immediately
        if (prediction.severity === 'high' || prediction.severity === 'Critical') {
            return true;
        }

        const stateKey = `healthState:${animalId}`;
        const rawState = await this.redis.get(stateKey);
        const prevState = safeParseJSON(rawState);

        if (prevState && prevState.aboveThreshold === true) {
            // Second consecutive above-threshold — trigger alert
            await this.redis.del(stateKey);
            return true;
        }

        // First above-threshold prediction — store state, wait for confirmation
        await this.redis.set(stateKey, JSON.stringify({
            aboveThreshold: true,
            diseaseProb: prediction.disease_prob,
            timestamp: new Date().toISOString()
        }), 'EX', DEFAULTS.HEALTH_STATE_TTL);

        this._metrics.consecutiveBlocked++;

        this.log.info({
            service: LOG_SERVICE,
            action: 'CONSECUTIVE_WAITING',
            animalId,
            diseaseProb: prediction.disease_prob
        }, `[HealthAgent] ${animalId} above threshold — waiting for confirmation`);

        return false;
    }

    // ─────────────────────────────────────────────────────────────────────────
    // ALERT CREATION — with MongoDB deduplication
    // ─────────────────────────────────────────────────────────────────────────

    /**
     * Create a health risk alert, but only if no unresolved alert exists
     * for this animal within the deduplication window.
     * 
     * @param {Object} animal     - LivestockMaster document
     * @param {Object} prediction - ML prediction result
     */
    async createAlert(animal, prediction) {
        const animalId = animal._id.toString();
        const tenantId = animal.userId.toString();

        try {
            // ── Deduplication check ──────────────────────────────────
            const dedupWindow = new Date(Date.now() - DEFAULTS.ALERT_DEDUP_HOURS * 60 * 60 * 1000);

            const existingAlert = await Alert.findOne({
                livestockId: animal._id,
                alertType: { $in: ['Health', 'Temperature'] },    // Health risk alert (backward compat)
                resolved: false,
                timestamp: { $gte: dedupWindow }
            }).lean();

            if (existingAlert) {
                this._metrics.duplicateAlerts++;
                this.log.info({
                    service: LOG_SERVICE,
                    action: 'ALERT_DEDUPLICATED',
                    animalId,
                    existingAlertId: existingAlert._id.toString()
                }, `[HealthAgent] Alert deduplicated for ${animalId} — existing unresolved alert`);
                return;
            }

            // ── Map severity ─────────────────────────────────────────
            const severityMap = {
                'low': 'Medium',
                'medium': 'High',
                'high': 'Critical'
            };
            const alertSeverity = severityMap[prediction.severity] || 'High';

            // ── Build contextual message ─────────────────────────────
            const probPercent = (prediction.disease_prob * 100).toFixed(1);
            const name = animal.name || 'Unknown';
            const breed = animal.breed || 'Unknown breed';
            const farm = animal.farm_name || 'Unknown farm';
            const zone = animal.zone_name || 'Unknown zone';
            const device = animal.device_id || '';

            // Human-readable explanation of why the alert was triggered
            let explanation = '';
            if (prediction.disease_prob >= 0.90) {
                explanation = `Very high disease probability (${probPercent}%) — immediate veterinary inspection recommended. ` +
                    `The ML model detected significant anomalies in vital signs and activity patterns.`;
            } else if (prediction.disease_prob >= 0.70) {
                explanation = `Elevated disease probability (${probPercent}%) — schedule veterinary check within 24 hours. ` +
                    `The ML model detected abnormal patterns in recent telemetry data.`;
            } else {
                explanation = `Moderate disease probability (${probPercent}%) — monitor closely over next 12 hours. ` +
                    `Minor deviations from normal patterns detected.`;
            }

            const message = `${name} (${breed}) at ${farm} / ${zone} — ${probPercent}% disease probability detected by ML model. ${explanation}`;

            // ── Create alert ─────────────────────────────────────────
            const alert = await Alert.create({
                livestockId: animal._id,
                userId: animal.userId,
                alertType: 'Health',
                severity: alertSeverity,
                message,
                animalName: name,
                farmName: farm,
                zoneName: zone,
                breed: breed,
                deviceId: device,
                diseaseProbability: prediction.disease_prob,
                alertSource: 'ml_health_agent',
                resolved: false,
                status: 'Pending'
            });

            this._metrics.alertsCreated++;

            this.log.info({
                service: LOG_SERVICE,
                action: 'ALERT_CREATED',
                animalId,
                tenantId,
                alertId: alert._id.toString(),
                severity: alertSeverity,
                diseaseProb: prediction.disease_prob
            }, `[HealthAgent] Alert created for ${animalId} — severity: ${alertSeverity}, prob: ${(prediction.disease_prob * 100).toFixed(1)}%`);

            // ── Publish alert event ──────────────────────────────────
            if (this.eventBus) {
                this.eventBus.emit('alert:saved', {
                    alertId: alert._id.toString(),
                    livestockId: animalId,
                    tenantId,
                    severity: alertSeverity,
                    message: alert.message,
                    animalName: name,
                    farmName: farm,
                    zoneName: zone,
                    breed: breed,
                    deviceId: device,
                    diseaseProbability: prediction.disease_prob
                });

                // Trigger sync to frontend
                this.eventBus.emit('db:write', {
                    type: 'alert',
                    data: alert
                });
            }

        } catch (err) {
            this.log.error({
                service: LOG_SERVICE,
                action: 'ALERT_CREATION_ERROR',
                animalId,
                error: err.message
            }, `[HealthAgent] Failed to create alert for ${animalId}`);
        }
    }

    // ─────────────────────────────────────────────────────────────────────────
    // HELPERS
    // ─────────────────────────────────────────────────────────────────────────

    /**
     * Compute severity from risk score when ML service doesn't provide one.
     */
    _computeSeverity(riskScore) {
        if (riskScore >= 0.8) return 'high';
        if (riskScore >= 0.5) return 'medium';
        return 'low';
    }

    // ─────────────────────────────────────────────────────────────────────────
    // MONITORING API
    // ─────────────────────────────────────────────────────────────────────────

    /**
     * Return structured metrics for monitoring dashboards.
     */
    getMetrics() {
        const m = this._metrics;
        return {
            totalRuns: m.totalRuns,
            animalsProcessed: m.animalsProcessed,
            animalsSkipped: m.animalsSkipped,
            alertsCreated: m.alertsCreated,
            duplicateAlerts: m.duplicateAlerts,
            consecutiveBlocked: m.consecutiveBlocked,
            mlCalls: m.mlCalls,
            mlFailures: m.mlFailures,
            circuitBroken: m.circuitBroken,
            featureMissing: m.featureMissing,
            featureInvalid: m.featureInvalid,
            avgPredictionLatencyMs: m.mlCalls > 0
                ? Math.round(m.mlLatencySum / (m.mlCalls - m.mlFailures || 1))
                : 0,
            circuitBreakerOpen: this._circuitOpenUntil > Date.now(),
            threshold: this._threshold,
            operatingMode: this.mode
        };
    }

    /**
     * Reset metrics — useful between test runs.
     */
    resetMetrics() {
        for (const key of Object.keys(this._metrics)) {
            this._metrics[key] = 0;
        }
    }
}

module.exports = HealthAgent;
