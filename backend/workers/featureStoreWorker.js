/**
 * Feature Store Worker
 * 
 * Continuously computes and caches the 23 XGBoost ML features per animal
 * in Redis so the Health Agent / ML service can predict without touching MongoDB.
 * 
 * Runs on a 60-second interval. Features expire after 300s (5 min) in Redis.
 * Feature names match model_config.json exactly.
 */

const path = require('path');
const envFile = process.env.NODE_ENV === 'production'
    ? '.env.production'
    : process.env.NODE_ENV === 'staging'
        ? '.env.staging'
        : '.env.development';

require('dotenv').config({ path: path.join(__dirname, '../', envFile) });

const mongoose = require('mongoose');
const redisConnection = require('../config/redis');
const DeviceTelemetry = require('../models/DeviceTelemetry');
const LivestockMaster = require('../models/LivestockMaster');
const { avg, std, slope } = require('../utils/stats');
const logger = require('../utils/logger');

// ── Constants ───────────────────────────────────────────────────────────────
const FEATURE_WINDOW = 12;      // same as training
const REFRESH_MS = 60000;   // compute every 60 seconds
const REDIS_TTL = 300;     // 5-min expiry, refreshed every 60s
const MIN_RECORDS = 3;       // need at least 3 telemetry points

// ── Feature Computation ─────────────────────────────────────────────────────

/**
 * Compute the 23 XGBoost features for a single animal from its recent telemetry.
 * Returns null if insufficient data.
 */
async function computeFeatures(animalId) {
    const telemetry = await DeviceTelemetry
        .find({ animalId })
        .sort({ timestamp: -1 })
        .limit(FEATURE_WINDOW)
        .lean();

    if (telemetry.length < MIN_RECORDS) return null;

    // Reverse to chronological order (oldest → newest)
    const temps = telemetry.map(t => t.temperature || 39.0).reverse();
    const hrs = telemetry.map(t => t.heartRate || 60).reverse();
    const activities = telemetry.map(t => typeof t.activity === 'number' ? t.activity : 75).reverse();

    const t_avg = avg(temps), t_std = std(temps);
    const h_avg = avg(hrs), h_std = std(hrs);
    const a_avg = avg(activities), a_std = std(activities);

    return {
        // Temperature (7 features)
        temp_current: temps[temps.length - 1],
        temp_6h_avg: t_avg,
        temp_6h_std: t_std,
        temp_6h_slope: slope(temps),
        temp_max_6h: Math.max(...temps),
        temp_min_6h: Math.min(...temps),
        temp_range_6h: Math.max(...temps) - Math.min(...temps),

        // Heart Rate (4 features)
        hr_current: hrs[hrs.length - 1],
        hr_6h_avg: h_avg,
        hr_6h_std: h_std,
        hr_6h_slope: slope(hrs),

        // Activity (4 features)
        activity_current: activities[activities.length - 1],
        activity_6h_avg: a_avg,
        activity_6h_std: a_std,
        activity_6h_slope: slope(activities),

        // Ratios (3 features)
        temp_ratio: t_avg > 0 ? temps[temps.length - 1] / t_avg : 1,
        hr_ratio: h_avg > 0 ? hrs[hrs.length - 1] / h_avg : 1,
        activity_ratio: a_avg > 0.01 ? activities[activities.length - 1] / a_avg : 1,

        // Z-scores (2 features)
        temp_zscore: t_std > 0 ? (temps[temps.length - 1] - t_avg) / t_std : 0,
        hr_zscore: h_std > 0 ? (hrs[hrs.length - 1] - h_avg) / h_std : 0,

        // Recent vs Baseline (3 features)
        temp_recent_vs_baseline: avg(temps.slice(-3)) - t_avg,
        hr_recent_vs_baseline: avg(hrs.slice(-3)) - h_avg,
        activity_recent_vs_baseline: avg(activities.slice(-3)) - a_avg,

        // Metadata
        updated_at: new Date().toISOString(),
        window_used: telemetry.length,
        feature_version: 'v3'
    };
}

// ── Redis Key ───────────────────────────────────────────────────────────────

function getFeatureKey(tenantId, animalId) {
    return `features:v3:${tenantId}:${animalId}`;
}

// ── Single Animal Update ────────────────────────────────────────────────────

async function updateAnimalFeatures(animal) {
    try {
        const features = await computeFeatures(animal._id);
        if (!features) return false;

        const key = getFeatureKey(animal.userId, animal._id);
        await redisConnection.setex(key, REDIS_TTL, JSON.stringify(features));
        return true;
    } catch (err) {
        logger.error({
            service: 'feature_store',
            action: 'COMPUTE_ERROR',
            animalId: animal._id,
            error: err.message
        }, `Failed to compute features for ${animal.livestock_id}`);
        return false;
    }
}

// ── Refresh All Animals ─────────────────────────────────────────────────────

async function refreshAllFeatures() {
    const startTime = Date.now();

    try {
        const animals = await LivestockMaster.find({
            livestock_id: { $exists: true },
            device_id: { $exists: true },
            userId: { $exists: true }
        }).select('_id livestock_id device_id userId').lean();

        const results = await Promise.allSettled(
            animals.map(a => updateAnimalFeatures(a))
        );

        const succeeded = results.filter(r => r.status === 'fulfilled' && r.value === true).length;
        const skipped = results.filter(r => r.status === 'fulfilled' && r.value === false).length;
        const failed = results.filter(r => r.status === 'rejected').length;
        const duration = Date.now() - startTime;

        logger.info({
            service: 'feature_store',
            action: 'REFRESH_COMPLETE',
            total: animals.length,
            cached: succeeded,
            skipped,
            failed,
            duration_ms: duration
        }, `Feature store refreshed: ${succeeded}/${animals.length} animals cached in ${duration}ms`);

    } catch (err) {
        logger.error({
            service: 'feature_store',
            action: 'REFRESH_ERROR',
            error: err.message
        }, 'Feature store refresh cycle failed');
    }
}

// ── Public API ──────────────────────────────────────────────────────────────

/**
 * Read cached features for a single animal from Redis.
 * Returns null if not cached or expired.
 */
async function getCachedFeatures(tenantId, animalId) {
    const key = getFeatureKey(tenantId, animalId);
    const data = await redisConnection.get(key);
    if (!data) return null;

    try {
        return JSON.parse(data);
    } catch {
        return null;
    }
}

// ── Lifecycle ───────────────────────────────────────────────────────────────

let refreshInterval = null;

function start() {
    logger.info({
        service: 'feature_store',
        action: 'WORKER_STARTED',
        refreshMs: REFRESH_MS,
        featureWindow: FEATURE_WINDOW,
        redisTtl: REDIS_TTL
    }, `[FeatureStoreWorker] Started — refreshing every ${REFRESH_MS / 1000}s`);

    // Run immediately on start, then every REFRESH_MS
    refreshAllFeatures();
    refreshInterval = setInterval(refreshAllFeatures, REFRESH_MS);
}

function stop() {
    if (refreshInterval) {
        clearInterval(refreshInterval);
        refreshInterval = null;
    }
    logger.info({ service: 'feature_store', action: 'WORKER_STOPPED' }, '[FeatureStoreWorker] Stopped');
}

module.exports = {
    start,
    stop,
    refreshAllFeatures,
    getCachedFeatures,
    computeFeatures,
    getFeatureKey,
    FEATURE_WINDOW
};
