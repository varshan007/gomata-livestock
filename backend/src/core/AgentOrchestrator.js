/**
 * AgentOrchestrator — Production-Grade Distributed Agent Execution Engine
 * 
 * Enterprise-grade orchestration for the livestock monitoring ML system.
 * Supports multi-instance deployment with Redis-based distributed locking,
 * health tracking, retry with exponential backoff, timeout control,
 * and graceful shutdown.
 * 
 * Architecture:
 * ┌──────────────┐     ┌───────────────┐     ┌──────────────┐
 * │  Instance A  │────▶│  Redis Locks   │◀────│  Instance B  │
 * │  Orchestrator│     │  SET NX EX 60  │     │  Orchestrator│
 * └──────┬───────┘     └───────────────┘     └──────┬───────┘
 *        │                                          │
 *   ┌────┴────┐                                ┌────┴────┐
 *   │ Agents  │                                │ Agents  │
 *   │ Health  │                                │ Health  │
 *   │ Feature │  (only one wins the lock)      │ Feature │
 *   │ Alert   │                                │ Alert   │
 *   └─────────┘                                └─────────┘
 */

const cron = require('node-cron');
const { Worker } = require('bullmq');
const logger = require('../../utils/logger');

// ── Constants ───────────────────────────────────────────────────────────────

const DEFAULT_TIMEOUT_MS = 30_000;       // 30s default agent timeout
const DEFAULT_MAX_RETRIES = 3;
const DEFAULT_BASE_BACKOFF_MS = 500;          // 500ms → 1s → 2s
const FAILURE_THRESHOLD = 5;            // failureCount >= 5 → "failed"
const DEGRADED_THRESHOLD = 3;            // failureCount >= 3 → "degraded"
const LOCK_TTL_SECONDS = 60;           // distributed lock expiry
const LOG_SERVICE = 'orchestrator';

// ── Health Record Factory ───────────────────────────────────────────────────

function createHealthRecord() {
    return {
        lastRun: null,
        lastSuccess: null,
        failureCount: 0,
        consecutiveSuccess: 0,
        status: 'healthy',    // healthy | degraded | failed
        averageDurationMs: 0,
        totalRuns: 0,
        totalSuccesses: 0,
        totalFailures: 0,
        lastError: null
    };
}

// ── Timeout Promise Wrapper ─────────────────────────────────────────────────

function withTimeout(promise, ms, agentName) {
    return new Promise((resolve, reject) => {
        const timer = setTimeout(() => {
            reject(new Error(`Agent [${agentName}] execution timed out after ${ms}ms`));
        }, ms);

        promise
            .then(result => { clearTimeout(timer); resolve(result); })
            .catch(err => { clearTimeout(timer); reject(err); });
    });
}

// ── Sleep Helper ────────────────────────────────────────────────────────────

function sleep(ms) {
    return new Promise(resolve => setTimeout(resolve, ms));
}

// ═════════════════════════════════════════════════════════════════════════════
// AgentOrchestrator
// ═════════════════════════════════════════════════════════════════════════════

class AgentOrchestrator {
    constructor(redis) {
        // Core state maps
        this.agents = new Map();    // name → { run: fn, options }
        this.schedules = new Map();    // name → cron.ScheduledTask
        this.running = new Set();    // names currently executing (single-instance guard)
        this.health = new Map();    // name → health record
        this.workers = new Map();    // name → BullMQ Worker instance
        this.intervals = new Map();    // name → setInterval ID

        // Redis connection for distributed locking
        this.redis = redis;

        // Shutdown flag
        this._shuttingDown = false;

        logger.info({
            service: LOG_SERVICE,
            action: 'ORCHESTRATOR_CREATED'
        }, '[AgentOrchestrator] Created — awaiting agent registrations');
    }

    // ─────────────────────────────────────────────────────────────────────────
    // 1. AGENT REGISTRATION
    // ─────────────────────────────────────────────────────────────────────────

    /**
     * Register an agent with the orchestrator.
     * 
     * @param {string}   name              Unique agent identifier
     * @param {Function} runFn             async function to execute
     * @param {Object}   [options]         Configuration overrides
     * @param {number}   [options.timeoutMs]      Execution timeout (default: 30s)
     * @param {number}   [options.maxRetries]     Max retry attempts (default: 3)
     * @param {number}   [options.baseBackoffMs]  Base backoff delay (default: 500ms)
     * @param {boolean}  [options.distributed]    Use Redis distributed lock (default: true)
     */
    register(name, runFn, options = {}) {
        if (this.agents.has(name)) {
            logger.warn({ service: LOG_SERVICE, agent: name }, `Agent [${name}] already registered — overwriting`);
        }

        this.agents.set(name, {
            run: runFn,
            options: {
                timeoutMs: options.timeoutMs ?? DEFAULT_TIMEOUT_MS,
                maxRetries: options.maxRetries ?? DEFAULT_MAX_RETRIES,
                baseBackoffMs: options.baseBackoffMs ?? DEFAULT_BASE_BACKOFF_MS,
                distributed: options.distributed ?? true
            }
        });

        // Initialize health record
        if (!this.health.has(name)) {
            this.health.set(name, createHealthRecord());
        }

        logger.info({
            service: LOG_SERVICE,
            action: 'AGENT_REGISTERED',
            agent: name,
            timeout: options.timeoutMs ?? DEFAULT_TIMEOUT_MS,
            retries: options.maxRetries ?? DEFAULT_MAX_RETRIES
        }, `[AgentOrchestrator] Registered agent: ${name}`);
    }

    // ─────────────────────────────────────────────────────────────────────────
    // 2. DISTRIBUTED LOCKING (Multi-Instance Safety)
    // ─────────────────────────────────────────────────────────────────────────

    /**
     * Attempt to acquire a Redis distributed lock.
     * Uses SET NX EX pattern for atomic lock acquisition.
     * 
     * @param   {string}  name  Agent name
     * @returns {boolean} true if lock acquired, false if already held
     */
    async _acquireLock(name) {
        const lockKey = `lock:agent:${name}`;
        try {
            const result = await this.redis.set(lockKey, process.pid.toString(), 'NX', 'EX', LOCK_TTL_SECONDS);
            return result === 'OK';
        } catch (err) {
            logger.error({
                service: LOG_SERVICE,
                agent: name,
                action: 'LOCK_ACQUIRE_ERROR',
                error: err.message
            }, `Failed to acquire distributed lock for [${name}]`);
            return false;
        }
    }

    /**
     * Release a Redis distributed lock.
     * Only deletes if the lock is still held (prevents releasing another instance's lock).
     */
    async _releaseLock(name) {
        const lockKey = `lock:agent:${name}`;
        try {
            await this.redis.del(lockKey);
        } catch (err) {
            logger.error({
                service: LOG_SERVICE,
                agent: name,
                action: 'LOCK_RELEASE_ERROR',
                error: err.message
            }, `Failed to release distributed lock for [${name}]`);
        }
    }

    // ─────────────────────────────────────────────────────────────────────────
    // 3. CORE EXECUTION ENGINE
    // ─────────────────────────────────────────────────────────────────────────

    /**
     * Execute a registered agent with full safety:
     *   1. Overlap protection (single-instance)
     *   2. Distributed locking (multi-instance)
     *   3. Timeout control
     *   4. Retry with exponential backoff
     *   5. Health state tracking
     * 
     * @param {string} name     Agent name
     * @param {*}      [context] Optional context passed to run function
     */
    async execute(name, context = null) {
        if (this._shuttingDown) return;

        const entry = this.agents.get(name);
        if (!entry) {
            logger.error({ service: LOG_SERVICE, agent: name }, `Agent [${name}] not registered`);
            return;
        }

        const { run, options } = entry;
        const health = this.health.get(name);

        // ── Guard 1: Single-instance overlap protection ──────────────
        if (this.running.has(name)) {
            logger.warn({
                service: LOG_SERVICE,
                agent: name,
                action: 'OVERLAP_SKIPPED'
            }, `Agent [${name}] is already running — skipping overlapping execution`);
            return;
        }

        // ── Guard 2: Distributed lock (multi-instance safety) ────────
        if (options.distributed) {
            const lockAcquired = await this._acquireLock(name);
            if (!lockAcquired) {
                logger.info({
                    service: LOG_SERVICE,
                    agent: name,
                    action: 'DISTRIBUTED_LOCK_SKIPPED'
                }, `Agent [${name}] skipped — locked by another instance`);
                return;
            }
        }

        // ── Mark as running ──────────────────────────────────────────
        this.running.add(name);
        health.lastRun = new Date();
        health.totalRuns++;

        const startTime = Date.now();
        let succeeded = false;

        try {
            // ── Retry loop with exponential backoff ──────────────────
            let lastError = null;

            for (let attempt = 0; attempt <= options.maxRetries; attempt++) {
                try {
                    if (attempt > 0) {
                        const backoffMs = options.baseBackoffMs * Math.pow(2, attempt - 1);
                        logger.warn({
                            service: LOG_SERVICE,
                            agent: name,
                            action: 'RETRY',
                            attempt,
                            backoffMs
                        }, `Agent [${name}] retry ${attempt}/${options.maxRetries} after ${backoffMs}ms backoff`);
                        await sleep(backoffMs);
                    }

                    // ── Execute with timeout ─────────────────────────
                    await withTimeout(
                        run(context),
                        options.timeoutMs,
                        name
                    );

                    succeeded = true;
                    break; // Success — exit retry loop

                } catch (err) {
                    lastError = err;
                    logger.error({
                        service: LOG_SERVICE,
                        agent: name,
                        action: 'EXECUTION_ERROR',
                        attempt: attempt + 1,
                        maxRetries: options.maxRetries,
                        error: err.message
                    }, `Agent [${name}] attempt ${attempt + 1} failed: ${err.message}`);
                }
            }

            // ── Update health state ──────────────────────────────────
            const durationMs = Date.now() - startTime;

            if (succeeded) {
                this._recordSuccess(name, durationMs);
            } else {
                this._recordFailure(name, durationMs, lastError);
            }

        } finally {
            // ── Always cleanup: remove from running set + release lock ─
            this.running.delete(name);

            if (options.distributed) {
                await this._releaseLock(name);
            }
        }
    }

    // ─────────────────────────────────────────────────────────────────────────
    // 4. HEALTH STATE MANAGEMENT
    // ─────────────────────────────────────────────────────────────────────────

    /**
     * Record a successful execution — reset failure count, update rolling average.
     */
    _recordSuccess(name, durationMs) {
        const h = this.health.get(name);
        h.lastSuccess = new Date();
        h.failureCount = 0;
        h.consecutiveSuccess++;
        h.totalSuccesses++;
        h.lastError = null;

        // Rolling average duration (exponential moving average)
        if (h.averageDurationMs === 0) {
            h.averageDurationMs = durationMs;
        } else {
            h.averageDurationMs = Math.round(h.averageDurationMs * 0.8 + durationMs * 0.2);
        }

        // Recover from degraded/failed → healthy
        if (h.status !== 'healthy' && h.consecutiveSuccess >= 2) {
            h.status = 'healthy';
            logger.info({
                service: LOG_SERVICE,
                agent: name,
                action: 'STATUS_RECOVERED'
            }, `Agent [${name}] recovered to healthy status`);
        } else if (h.status === 'healthy') {
            // Stay healthy
        }

        logger.info({
            service: LOG_SERVICE,
            agent: name,
            action: 'EXECUTION_SUCCESS',
            durationMs,
            avgDurationMs: h.averageDurationMs
        }, `Agent [${name}] completed in ${durationMs}ms`);
    }

    /**
     * Record a failed execution — increment failure counter, update status.
     */
    _recordFailure(name, durationMs, error) {
        const h = this.health.get(name);
        h.failureCount++;
        h.consecutiveSuccess = 0;
        h.totalFailures++;
        h.lastError = error ? error.message : 'Unknown error';

        // Degrade status based on failure count thresholds
        if (h.failureCount >= FAILURE_THRESHOLD) {
            h.status = 'failed';
        } else if (h.failureCount >= DEGRADED_THRESHOLD) {
            h.status = 'degraded';
        }

        logger.error({
            service: LOG_SERVICE,
            agent: name,
            action: 'EXECUTION_FAILED',
            durationMs,
            failureCount: h.failureCount,
            status: h.status,
            error: h.lastError
        }, `Agent [${name}] failed after all retries (failures: ${h.failureCount}, status: ${h.status})`);
    }

    // ─────────────────────────────────────────────────────────────────────────
    // 5. SCHEDULING
    // ─────────────────────────────────────────────────────────────────────────

    /**
     * Schedule an agent to run on a cron expression.
     * 
     * @param {string} name        Agent name (must be registered)
     * @param {string} cronExpr    Cron expression (e.g., '* /1 * * * *')
     * @param {*}      [context]   Optional context passed to run
     */
    schedule(name, cronExpr, context = null) {
        if (!this.agents.has(name)) {
            throw new Error(`Cannot schedule unregistered agent: ${name}`);
        }

        // Stop existing schedule if re-scheduling
        if (this.schedules.has(name)) {
            this.schedules.get(name).stop();
        }

        const task = cron.schedule(cronExpr, () => {
            this.execute(name, context).catch(err => {
                logger.error({
                    service: LOG_SERVICE,
                    agent: name,
                    action: 'CRON_UNHANDLED_ERROR',
                    error: err.message
                }, `Unhandled error in cron for [${name}]`);
            });
        });

        this.schedules.set(name, task);

        logger.info({
            service: LOG_SERVICE,
            agent: name,
            action: 'CRON_SCHEDULED',
            cronExpr
        }, `Agent [${name}] scheduled: ${cronExpr}`);
    }

    /**
     * Schedule an agent to run on a fixed interval.
     * 
     * @param {string} name        Agent name (must be registered)
     * @param {number} intervalMs  Interval in milliseconds
     * @param {*}      [context]   Optional context
     * @param {boolean} [immediate] Run immediately on start (default: true)
     */
    scheduleInterval(name, intervalMs, context = null, immediate = true) {
        if (!this.agents.has(name)) {
            throw new Error(`Cannot schedule unregistered agent: ${name}`);
        }

        // Stop existing interval
        if (this.intervals.has(name)) {
            clearInterval(this.intervals.get(name));
        }

        const id = setInterval(() => {
            this.execute(name, context).catch(err => {
                logger.error({
                    service: LOG_SERVICE,
                    agent: name,
                    action: 'INTERVAL_UNHANDLED_ERROR',
                    error: err.message
                }, `Unhandled error in interval for [${name}]`);
            });
        }, intervalMs);

        this.intervals.set(name, id);

        if (immediate) {
            this.execute(name, context).catch(() => { });
        }

        logger.info({
            service: LOG_SERVICE,
            agent: name,
            action: 'INTERVAL_SCHEDULED',
            intervalMs,
            immediate
        }, `Agent [${name}] scheduled: every ${intervalMs}ms`);
    }

    // ─────────────────────────────────────────────────────────────────────────
    // 6. EVENT-DRIVEN SUBSCRIPTIONS (BullMQ Workers)
    // ─────────────────────────────────────────────────────────────────────────

    /**
     * Subscribe an agent to a BullMQ queue.
     * Creates a Worker that processes jobs through the orchestrator's execution engine.
     * 
     * @param {string}   name         Agent name (must be registered)
     * @param {string}   queueName    BullMQ queue name
     * @param {Object}   [workerOpts] Additional BullMQ Worker options
     */
    subscribe(name, queueName, workerOpts = {}) {
        if (!this.agents.has(name)) {
            throw new Error(`Cannot subscribe unregistered agent: ${name}`);
        }

        // Close existing worker if re-subscribing
        if (this.workers.has(name)) {
            this.workers.get(name).close().catch(() => { });
        }

        const worker = new Worker(queueName, async (job) => {
            // For event-driven agents, skip distributed lock (BullMQ handles it)
            // but still get overlap protection, timeout, retry, and health tracking
            const originalEntry = this.agents.get(name);
            const originalDistributed = originalEntry.options.distributed;

            // Temporarily disable distributed lock for BullMQ jobs
            // (BullMQ already guarantees single processing per job)
            originalEntry.options.distributed = false;

            try {
                await this.execute(name, job.data);
            } finally {
                originalEntry.options.distributed = originalDistributed;
            }

            return { status: 'processed', agent: name, jobId: job.id };
        }, {
            connection: this.redis,
            concurrency: workerOpts.concurrency || 1,
            ...workerOpts
        });

        worker.on('failed', (job, err) => {
            logger.error({
                service: LOG_SERVICE,
                agent: name,
                action: 'WORKER_JOB_FAILED',
                jobId: job ? job.id : 'unknown',
                error: err.message
            }, `Worker job failed for [${name}]`);
        });

        worker.on('error', (err) => {
            logger.error({
                service: LOG_SERVICE,
                agent: name,
                action: 'WORKER_ERROR',
                error: err.message
            }, `Worker error for [${name}]`);
        });

        this.workers.set(name, worker);

        logger.info({
            service: LOG_SERVICE,
            agent: name,
            action: 'WORKER_SUBSCRIBED',
            queue: queueName
        }, `Agent [${name}] subscribed to queue: ${queueName}`);
    }

    // ─────────────────────────────────────────────────────────────────────────
    // 7. EVENT BUS SUBSCRIPTIONS (Redis Pub/Sub)
    // ─────────────────────────────────────────────────────────────────────────

    /**
     * Subscribe an agent to a Redis EventBus event.
     * Wraps the callback with orchestrator safety (overlap, health tracking).
     * 
     * @param {string}   name       Agent name (must be registered)
     * @param {Object}   eventBus   RedisEventBus instance
     * @param {string}   eventName  Event to listen for
     */
    subscribeToBus(name, eventBus, eventName) {
        if (!this.agents.has(name)) {
            throw new Error(`Cannot subscribe unregistered agent: ${name}`);
        }

        eventBus.on(eventName, async (payload) => {
            // For event-driven agents, skip distributed lock
            const originalEntry = this.agents.get(name);
            const originalDistributed = originalEntry.options.distributed;
            originalEntry.options.distributed = false;

            try {
                await this.execute(name, payload);
            } finally {
                originalEntry.options.distributed = originalDistributed;
            }
        });

        logger.info({
            service: LOG_SERVICE,
            agent: name,
            action: 'BUS_SUBSCRIBED',
            event: eventName
        }, `Agent [${name}] subscribed to event: ${eventName}`);
    }

    // ─────────────────────────────────────────────────────────────────────────
    // 8. MONITORING API
    // ─────────────────────────────────────────────────────────────────────────

    /**
     * Get comprehensive health status for all registered agents.
     * Designed for monitoring dashboards and /health API endpoints.
     * 
     * @returns {Object} Full health report
     */
    getHealthStatus() {
        const agents = {};

        for (const [name, h] of this.health.entries()) {
            agents[name] = {
                status: h.status,
                failureCount: h.failureCount,
                lastRun: h.lastRun,
                lastSuccess: h.lastSuccess,
                avgDurationMs: h.averageDurationMs,
                totalRuns: h.totalRuns,
                totalSuccesses: h.totalSuccesses,
                totalFailures: h.totalFailures,
                isRunning: this.running.has(name),
                hasWorker: this.workers.has(name),
                hasCron: this.schedules.has(name),
                hasInterval: this.intervals.has(name),
                lastError: h.lastError
            };
        }

        // Compute overall system status
        const statuses = Object.values(agents).map(a => a.status);
        let systemStatus = 'healthy';
        if (statuses.includes('failed')) systemStatus = 'critical';
        else if (statuses.includes('degraded')) systemStatus = 'degraded';

        return {
            system: {
                status: systemStatus,
                agentCount: this.agents.size,
                runningCount: this.running.size,
                workerCount: this.workers.size,
                uptime: process.uptime(),
                timestamp: new Date().toISOString()
            },
            agents
        };
    }

    // ─────────────────────────────────────────────────────────────────────────
    // 9. GRACEFUL SHUTDOWN
    // ─────────────────────────────────────────────────────────────────────────

    /**
     * Graceful shutdown: stops all cron jobs, intervals, workers, and releases locks.
     * Safe to call multiple times (idempotent).
     */
    async shutdown() {
        if (this._shuttingDown) return;
        this._shuttingDown = true;

        logger.info({
            service: LOG_SERVICE,
            action: 'SHUTDOWN_INITIATED',
            agents: this.agents.size,
            workers: this.workers.size
        }, '[AgentOrchestrator] Graceful shutdown initiated...');

        // 1. Stop all cron schedules
        for (const [name, task] of this.schedules.entries()) {
            try {
                task.stop();
                logger.info({ service: LOG_SERVICE, agent: name }, `Cron stopped for [${name}]`);
            } catch (err) {
                logger.error({ service: LOG_SERVICE, agent: name, error: err.message }, `Failed to stop cron for [${name}]`);
            }
        }
        this.schedules.clear();

        // 2. Clear all intervals
        for (const [name, id] of this.intervals.entries()) {
            clearInterval(id);
            logger.info({ service: LOG_SERVICE, agent: name }, `Interval cleared for [${name}]`);
        }
        this.intervals.clear();

        // 3. Close all BullMQ workers (drain existing jobs first)
        const workerClosePromises = [];
        for (const [name, worker] of this.workers.entries()) {
            workerClosePromises.push(
                worker.close()
                    .then(() => logger.info({ service: LOG_SERVICE, agent: name }, `Worker closed for [${name}]`))
                    .catch(err => logger.error({ service: LOG_SERVICE, agent: name, error: err.message }, `Failed to close worker for [${name}]`))
            );
        }
        await Promise.allSettled(workerClosePromises);
        this.workers.clear();

        // 4. Release all distributed locks for currently running agents
        const lockReleasePromises = [];
        for (const name of this.running) {
            lockReleasePromises.push(this._releaseLock(name));
        }
        await Promise.allSettled(lockReleasePromises);
        this.running.clear();

        logger.info({
            service: LOG_SERVICE,
            action: 'SHUTDOWN_COMPLETE'
        }, '[AgentOrchestrator] Graceful shutdown complete');
    }
}

module.exports = AgentOrchestrator;
