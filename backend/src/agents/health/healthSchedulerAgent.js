/**
 * HealthSchedulerAgent — Multi-Tenant Health Scan Orchestration
 * 
 * Wraps HealthAgent.run(tenantId) across all active tenants with:
 *   1. Tenant discovery via LivestockMaster.distinct('userId')
 *   2. Batched concurrent execution (5 tenants at a time)
 *   3. Per-tenant failure isolation (one tenant failing doesn't stop others)
 *   4. Aggregate metrics and structured logging
 *   5. Ready for event-driven triggers (direct .run() call)
 * 
 * Scheduling is handled by AgentOrchestrator — this class is stateless
 * and safe for distributed execution behind a Redis lock.
 */

const LivestockMaster = require('../../../models/LivestockMaster');
const logger = require('../../../utils/logger');

const LOG_SERVICE = 'health_scheduler';
const DEFAULT_TENANT_CONCURRENCY = 5;

// ═════════════════════════════════════════════════════════════════════════════
// HealthSchedulerAgent
// ═════════════════════════════════════════════════════════════════════════════

class HealthSchedulerAgent {
    /**
     * @param {Object} options
     * @param {Object} options.healthAgent   - HealthAgent instance (already configured)
     * @param {number} [options.tenantBatch] - Max tenants processed in parallel (default: 5)
     */
    constructor(options = {}) {
        if (!options.healthAgent) {
            throw new Error('[HealthSchedulerAgent] healthAgent is required');
        }

        this.healthAgent = options.healthAgent;
        this.tenantBatch = options.tenantBatch || DEFAULT_TENANT_CONCURRENCY;

        // ── Aggregate metrics across all runs ────────────────────────
        this._metrics = {
            totalRuns: 0,
            totalTenants: 0,
            tenantsSucceeded: 0,
            tenantsFailed: 0,
            totalDurationMs: 0,
            lastRun: null,
            lastDurationMs: 0,
        };

        logger.info({
            service: LOG_SERVICE,
            action: 'SCHEDULER_CREATED',
            tenantBatch: this.tenantBatch
        }, `[HealthSchedulerAgent] Created — tenant batch size: ${this.tenantBatch}`);
    }

    // ─────────────────────────────────────────────────────────────────────────
    // CORE RUN — Process all tenants in batches
    // ─────────────────────────────────────────────────────────────────────────

    /**
     * Discover all active tenants and run HealthAgent.run(tenantId) for each.
     * Batches tenants with concurrency limit to avoid overwhelming ML/Redis.
     * 
     * This method is the orchestrator's run function — it handles:
     *   - Tenant discovery
     *   - Batched execution
     *   - Per-tenant error isolation
     *   - Aggregate logging
     * 
     * Can also be called directly for event-driven triggers.
     */
    async run() {
        this._metrics.totalRuns++;
        this._metrics.lastRun = new Date();
        const startTime = Date.now();

        // ── 1. Discover active tenants ───────────────────────────────
        let tenantIds;
        try {
            tenantIds = await LivestockMaster.distinct('userId');
        } catch (err) {
            logger.error({
                service: LOG_SERVICE,
                action: 'TENANT_DISCOVERY_ERROR',
                error: err.message
            }, `[HealthSchedulerAgent] Failed to discover tenants`);
            return;
        }

        if (!tenantIds || tenantIds.length === 0) {
            logger.info({
                service: LOG_SERVICE,
                action: 'NO_TENANTS'
            }, `[HealthSchedulerAgent] No active tenants found — skipping`);
            return;
        }

        logger.info({
            service: LOG_SERVICE,
            action: 'RUN_STARTED',
            tenantCount: tenantIds.length,
            batchSize: this.tenantBatch
        }, `[HealthSchedulerAgent] Starting health scan for ${tenantIds.length} tenants (batch: ${this.tenantBatch})`);

        // ── 2. Process tenants in batches ────────────────────────────
        let succeeded = 0;
        let failed = 0;

        for (let i = 0; i < tenantIds.length; i += this.tenantBatch) {
            const batch = tenantIds.slice(i, i + this.tenantBatch);
            const batchNum = Math.floor(i / this.tenantBatch) + 1;
            const totalBatches = Math.ceil(tenantIds.length / this.tenantBatch);

            logger.info({
                service: LOG_SERVICE,
                action: 'BATCH_STARTED',
                batch: batchNum,
                total: totalBatches,
                tenants: batch.length
            }, `[HealthSchedulerAgent] Batch ${batchNum}/${totalBatches} — ${batch.length} tenants`);

            // Execute batch with Promise.allSettled (never rejects)
            const results = await Promise.allSettled(
                batch.map(tenantId => this._runTenant(tenantId.toString()))
            );

            // Tally results
            for (const result of results) {
                if (result.status === 'fulfilled') {
                    succeeded++;
                } else {
                    failed++;
                }
            }
        }

        // ── 3. Log aggregate results ─────────────────────────────────
        const durationMs = Date.now() - startTime;
        this._metrics.totalTenants += tenantIds.length;
        this._metrics.tenantsSucceeded += succeeded;
        this._metrics.tenantsFailed += failed;
        this._metrics.totalDurationMs += durationMs;
        this._metrics.lastDurationMs = durationMs;

        // Get HealthAgent's per-run metrics
        const healthMetrics = this.healthAgent.getMetrics();

        logger.info({
            service: LOG_SERVICE,
            action: 'RUN_COMPLETE',
            tenantCount: tenantIds.length,
            succeeded,
            failed,
            durationMs,
            animalsProcessed: healthMetrics.animalsProcessed,
            alertsCreated: healthMetrics.alertsCreated,
            mlFailures: healthMetrics.mlFailures
        }, `[HealthSchedulerAgent] Complete — ${succeeded}/${tenantIds.length} tenants OK, ` +
        `${healthMetrics.animalsProcessed} animals, ${healthMetrics.alertsCreated} alerts, ${durationMs}ms`);
    }

    // ─────────────────────────────────────────────────────────────────────────
    // PER-TENANT EXECUTION — Isolated error handling
    // ─────────────────────────────────────────────────────────────────────────

    /**
     * Execute HealthAgent.run for a single tenant.
     * Wrapped in try/catch so one tenant's failure doesn't affect others.
     */
    async _runTenant(tenantId) {
        const startTime = Date.now();

        try {
            await this.healthAgent.run(tenantId);

            logger.info({
                service: LOG_SERVICE,
                action: 'TENANT_COMPLETE',
                tenantId,
                durationMs: Date.now() - startTime
            }, `[HealthSchedulerAgent] Tenant ${tenantId} complete in ${Date.now() - startTime}ms`);

        } catch (err) {
            logger.error({
                service: LOG_SERVICE,
                action: 'TENANT_ERROR',
                tenantId,
                durationMs: Date.now() - startTime,
                error: err.message
            }, `[HealthSchedulerAgent] Tenant ${tenantId} failed: ${err.message}`);

            throw err; // Re-throw so Promise.allSettled records it as rejected
        }
    }

    // ─────────────────────────────────────────────────────────────────────────
    // MONITORING API
    // ─────────────────────────────────────────────────────────────────────────

    /**
     * Aggregate metrics for monitoring dashboards.
     */
    getMetrics() {
        const m = this._metrics;
        return {
            totalRuns: m.totalRuns,
            totalTenants: m.totalTenants,
            tenantsSucceeded: m.tenantsSucceeded,
            tenantsFailed: m.tenantsFailed,
            lastRun: m.lastRun,
            lastDurationMs: m.lastDurationMs,
            avgDurationMs: m.totalRuns > 0
                ? Math.round(m.totalDurationMs / m.totalRuns)
                : 0,
            tenantBatch: this.tenantBatch,
            healthAgentMetrics: this.healthAgent.getMetrics()
        };
    }
}

module.exports = HealthSchedulerAgent;
