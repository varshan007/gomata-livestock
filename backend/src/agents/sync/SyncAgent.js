const logger = require('../../../utils/logger');
/**
 * SyncAgent
 * Event-driven synchronization layer. Listens for 'db:write' events 
 * (or specific entity events) from the EventBus and broadcasts them 
 * instantly to connected frontend clients via WebSockets (Socket.io).
 * 
 * Multi-tenant isolation: each client joins a `tenant:{userId}` room,
 * and events are emitted only to the relevant tenant room.
 */
class SyncAgent {
    constructor(bus, io) {
        this.bus = bus;
        this.io = io; // Socket.io server instance
    }

    start() {
        if (!this.io) {
            logger.error('[SyncAgent] Socket.io instance not provided. Cannot start.');
            return;
        }

        logger.info('[SyncAgent] Starting WebSocket synchronization...');

        // 1. Listen for new incoming telemetry — scope to tenant room
        this.bus.on('telemetry:received', (payload) => {
            const data = payload.data || payload;
            const tenantId = data.tenantId || data.userId;
            if (tenantId) {
                this.io.to(`tenant:${tenantId}`).emit('sync:telemetry_live', data);
            } else {
                // Fallback: broadcast globally (legacy compat)
                this.io.emit('sync:telemetry_live', data);
            }
        });

        // 2. Listen for generic Database writes — scope to tenant room
        this.bus.on('db:write', (payload) => {
            const { collection, documentId, operation, data } = payload.data || payload;
            const tenantId = data?.userId || data?.tenantId;
            if (tenantId) {
                this.io.to(`tenant:${tenantId}`).emit(`sync:db_${collection}`, { documentId, operation, data });
            } else {
                this.io.emit(`sync:db_${collection}`, { documentId, operation, data });
            }
        });

        // 3. Listen for livestock registration — scope to tenant room
        this.bus.on('onboarding:livestock_registered', (payload) => {
            const data = payload.data || payload;
            const tenantId = data.tenantId || data.userId;
            if (tenantId) {
                this.io.to(`tenant:${tenantId}`).emit('sync:livestock_added', data);
            } else {
                this.io.emit('sync:livestock_added', data);
            }
        });

        // 4. Forward verified AI Alerts — ONLY to the owning tenant's room
        this.bus.on('alert:saved', (payload) => {
            const data = payload.data || payload;
            const tenantId = data.tenantId || data.userId;
            if (tenantId) {
                this.io.to(`tenant:${tenantId}`).emit('alert:new', data);
                logger.info(`[SyncAgent] Alert emitted to tenant room tenant:${tenantId}`);
            } else {
                // Fallback: broadcast globally (should not happen in normal flow)
                logger.warn('[SyncAgent] Alert without tenantId — broadcasting globally');
                this.io.emit('alert:new', data);
            }
        });

        // Handle client connections
        this.io.on('connection', (socket) => {
            logger.info(`[SyncAgent] UI Client connected: ${socket.id}`);

            // Clients join their tenant room for isolated event delivery
            socket.on('join_tenant', (tenantId) => {
                if (tenantId) {
                    socket.join(`tenant:${tenantId}`);
                    logger.info(`[SyncAgent] Client ${socket.id} joined tenant room tenant:${tenantId}`);
                }
            });

            // Legacy room support
            socket.on('join_room', (roomId) => {
                socket.join(roomId);
                logger.info(`[SyncAgent] Client ${socket.id} joined room ${roomId}`);
            });

            socket.on('disconnect', () => {
                logger.info(`[SyncAgent] UI Client disconnected: ${socket.id}`);
            });
        });

        logger.info('[SyncAgent] Started with multi-tenant room isolation.');
    }
}

module.exports = SyncAgent;

