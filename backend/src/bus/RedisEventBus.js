const logger = require('../../utils/logger');
const Redis = require('ioredis');
const crypto = require('crypto');

/**
 * RedisEventBus
 * A generic, robust, multi-agent communication bus built on Redis Pub/Sub.
 * Injects tracing headers into every payload for cross-agent observability.
 */
class RedisEventBus {
    constructor(redisUrl = 'redis://127.0.0.1:6379') {
        this.publisher = new Redis(redisUrl);
        // Disable readyCheck on subscriber to prevent ioredis from running INFO after subscribing
        this.subscriber = new Redis(redisUrl, { enableReadyCheck: false });
        this.handlers = new Map();

        this.subscriber.on('message', (channel, message) => {
            const handlers = this.handlers.get(channel) || [];
            if (handlers.length > 0) {
                try {
                    const parsed = JSON.parse(message);
                    handlers.forEach(handler => handler(parsed));
                } catch (error) {
                    logger.error(`[EventBus] Error parsing message on channel ${channel}:`, error);
                }
            }
        });

        logger.info(`[EventBus] Connected to Redis at ${redisUrl}`);
    }

    /**
     * Emit an event to the bus.
     * Automatically wraps the payload in an envelope with traceId and timestamp.
     * @param {string} eventName The channel name (e.g. 'telemetry:received')
     * @param {object} payload The data to send
     * @param {string} providedTraceId Optional traceId from upstream to persist tracing
     */
    emit(eventName, payload, providedTraceId = null) {
        const envelope = {
            eventName,
            traceId: providedTraceId || crypto.randomUUID(),
            timestamp: new Date().toISOString(),
            data: payload
        };

        this.publisher.publish(eventName, JSON.stringify(envelope));
    }

    /**
     * Subscribe to an event topic.
     * @param {string} eventName The channel name
     * @param {function} handler Callback function receiving the parsed envelope
     */
    on(eventName, handler) {
        if (!this.handlers.has(eventName)) {
            this.subscriber.subscribe(eventName, (err) => {
                if (err) {
                    logger.error(`[EventBus] Failed to subscribe to ${eventName}:`, err);
                }
            });
            this.handlers.set(eventName, []);
        }
        this.handlers.get(eventName).push(handler);
    }
}

// Export a singleton instance.
const eventBus = new RedisEventBus(process.env.REDIS_URL || 'redis://127.0.0.1:6379');
module.exports = eventBus;
