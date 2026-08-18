/**
 * Sequence Store Worker (Phase 12)
 *
 * Maintains a sliding window List in Redis containing exactly the last 288 tick vectors 
 * (48 hours at 10-minute intervals) for every animal. This feeds the BiLSTM Trajectory Engine.
 * 
 * Runs on a 10-minute cron interval, pushing exactly 1 vector per animal into Redis Lists,
 * preventing the ML service from making expensive historical MongoDB aggregations on every prediction.
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
const logger = require('../utils/logger');
const { avg } = require('../utils/stats');

// ── Constants ───────────────────────────────────────────────────────────────
const SEQUENCE_WINDOW = 288;      // 48 hours * 6 ticks/hr
const TICK_INTERVAL_MS = 600000;  // 10 minutes
const REDIS_TTL = 86400;          // 24-hour expiry if stop ticking

// ── Feature Generation ──────────────────────────────────────────────────────

async function compute10MinTick(animal) {
    // 1. Get telemetry in the last 10 minutes
    const tenMinsAgo = new Date(Date.now() - TICK_INTERVAL_MS);
    const telemetry = await DeviceTelemetry
        .find({ animalId: animal._id, timestamp: { $gte: tenMinsAgo } })
        .lean();

    // Calculate avgs for this 10-min block, or fallback to the master profile bounds
    // The Phase 11 BiLSTM expects 18 features (some simulated/unavailable natively in current IoT DB schema)
    const temps = telemetry.map(t => t.temperature || 39.0);
    const hrs = telemetry.map(t => t.heartRate || 65.0);
    const acts = telemetry.map(t => typeof t.activity === 'number' ? t.activity : 0.65);

    const temp = temps.length ? avg(temps) : (animal.temperature || 38.5);
    const hr = hrs.length ? avg(hrs) : (animal.heart_rate || 65.0);
    const activity = acts.length ? avg(acts) : 0.65;

    // We stub environmental and production data that would come from 3rd party farm APIs.
    // In a real Phase 12 implementation, these would be pulled from external integrations (DeLaval, etc.)
    const resp = 25.0;            // stub physiological
    const rumination = 38.0;      // stub physiological
    const lying = 25.0;           // stub physiological
    const thi = 65.0;             // stub environmental
    const ambient_temp = 22.0;    // stub environmental
    const humidity = 55.0;        // stub environmental

    // Static / sparse production data from LivestockMaster
    const milk_yield = 25.0;
    const feed_intake = 20.0;
    const conductivity = 5.0;
    const body_weight = animal.weight || 550.0;
    const parity = 2.0;           // stub management
    const bcs = 3.2;              // stub management
    const age = animal.age || 4.0;

    // Hours since sparse events
    const hrs_since_vax = 999.0;
    const hrs_since_abx = 999.0;

    // Exact array mapping for PyTorch Standard Scaler input
    return [
        temp, hr, resp, activity, rumination, lying,
        thi, ambient_temp, humidity, milk_yield, feed_intake,
        conductivity, body_weight, parity, bcs, age,
        hrs_since_vax, hrs_since_abx
    ];
}

// ── Redis List Operations ───────────────────────────────────────────────────

function getSequenceKey(tenantId, animalId) {
    return `sequence:v10:${tenantId}:${animalId}`;
}

async function updateAnimalSequence(animal) {
    try {
        const tickVector = await compute10MinTick(animal);
        const key = getSequenceKey(animal.userId, animal._id);

        // Push this vector to the right of the Redis List
        await redisConnection.rpush(key, JSON.stringify(tickVector));

        // Trim the list to perfectly maintain the 288 boundary
        // LTRIM keeps indices from (-SEQUENCE_WINDOW) to (-1)
        await redisConnection.ltrim(key, -SEQUENCE_WINDOW, -1);

        // Refresh TTL so it drops if the collar dies
        await redisConnection.expire(key, REDIS_TTL);

        return true;
    } catch (err) {
        logger.error({
            service: 'sequence_store',
            action: 'COMPUTE_ERROR',
            animalId: animal._id,
            error: err.message
        }, `Failed to update sequence for ${animal.livestock_id}`);
        return false;
    }
}

// ── Refresh Loop ────────────────────────────────────────────────────────────

async function pushTickAllAnimals() {
    const startTime = Date.now();

    try {
        const animals = await LivestockMaster.find({
            livestock_id: { $exists: true },
            device_id: { $exists: true },
            userId: { $exists: true }
        }).select('_id livestock_id device_id userId temperature heart_rate weight age').lean();

        const results = await Promise.allSettled(
            animals.map(a => updateAnimalSequence(a))
        );

        const succeeded = results.filter(r => r.status === 'fulfilled' && r.value === true).length;
        const failed = results.filter(r => r.status === 'rejected').length;
        const duration = Date.now() - startTime;

        logger.info({
            service: 'sequence_store',
            action: 'TICK_COMPLETE',
            total: animals.length,
            pushed: succeeded,
            failed,
            duration_ms: duration
        }, `Sequence tick pushed: ${succeeded}/${animals.length} BiLSTM rolling windows updated in ${duration}ms`);

    } catch (err) {
        logger.error({
            service: 'sequence_store',
            action: 'TICK_ERROR',
            error: err.message
        }, 'Sequence list push cycle failed');
    }
}

// ── Lifecycle ───────────────────────────────────────────────────────────────

let tickInterval = null;

function start() {
    logger.info({
        service: 'sequence_store',
        action: 'WORKER_STARTED',
        intervalMs: TICK_INTERVAL_MS,
        windowSize: SEQUENCE_WINDOW
    }, `[SequenceStoreWorker] Started — pushing BiLSTM ticks every ${TICK_INTERVAL_MS / 60000} minutes`);

    // Run first tick immediately, then lock to interval
    pushTickAllAnimals();
    tickInterval = setInterval(pushTickAllAnimals, TICK_INTERVAL_MS);
}

function stop() {
    if (tickInterval) {
        clearInterval(tickInterval);
        tickInterval = null;
    }
    logger.info({ service: 'sequence_store', action: 'WORKER_STOPPED' }, '[SequenceStoreWorker] Stopped');
}

/**
 * Public API to fetch the full (padded if necessary) 288-tensor for ML.
 */
async function getSequenceForInference(tenantId, animalId) {
    const key = getSequenceKey(tenantId, animalId);
    const data = await redisConnection.lrange(key, 0, -1);

    if (!data || data.length === 0) return null;

    const sequence = data.map(d => JSON.parse(d));

    // Pad left if it hasn't reached 288 yet
    if (sequence.length < SEQUENCE_WINDOW) {
        const paddingRequired = SEQUENCE_WINDOW - sequence.length;
        const firstVector = sequence[0]; // naive forward-fill using the first known vector
        const padding = Array(paddingRequired).fill(firstVector);
        return [...padding, ...sequence];
    }

    return sequence;
}

module.exports = {
    start,
    stop,
    pushTickAllAnimals,
    getSequenceForInference,
    getSequenceKey,
    SEQUENCE_WINDOW
};
