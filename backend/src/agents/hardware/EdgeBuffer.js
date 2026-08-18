const logger = require('../../../utils/logger');
/**
 * EdgeBuffer
 * A robust local buffer for MQTT telemetry.
 * Stores incoming payloads when the backend or event bus is inaccessible.
 */
class EdgeBuffer {
    constructor(models) {
        this.models = models; // Using MongoDB to store buffered messages safely across restarts.
        // Assume models.Buffer exists (or we use an in-memory fallback for now)
        this.memoryQueue = [];
    }

    /**
     * Push a message to the buffer when offline.
     */
    async push(topic, payload) {
        logger.info(`[EdgeBuffer] Connection offline. Buffering message from ${topic}`);

        const timestampedPayload = {
            topic,
            payload: payload.toString(),
            bufferedAt: new Date(),
            synced: false
        };

        if (this.models && this.models.Buffer) {
            try {
                await this.models.Buffer.create(timestampedPayload);
            } catch (err) {
                logger.error('[EdgeBuffer] Failed to write to DB buffer, falling back to memory', err);
                this.memoryQueue.push(timestampedPayload);
            }
        } else {
            this.memoryQueue.push(timestampedPayload);
        }
    }

    /**
     * Flush all buffered messages to the EventBus upon reconnection.
     */
    async flush(bus) {
        logger.info(`[EdgeBuffer] Connection restored. Flushing buffer...`);
        let flushedCount = 0;

        // Flush Memory Queue
        while (this.memoryQueue.length > 0) {
            const msg = this.memoryQueue.shift();
            this._emitToBus(bus, msg);
            flushedCount++;
        }

        // Flush DB Queue if it exists
        if (this.models && this.models.Buffer) {
            try {
                const unsyncedMsgs = await this.models.Buffer.find({ synced: false }).sort({ bufferedAt: 1 });
                for (let msg of unsyncedMsgs) {
                    this._emitToBus(bus, msg);
                    msg.synced = true;
                    await msg.save();
                    flushedCount++;
                }
            } catch (err) {
                logger.error('[EdgeBuffer] Failed to flush DB buffer', err);
            }
        }

        if (flushedCount > 0) {
            logger.info(`[EdgeBuffer] Flushed ${flushedCount} historical messages to SyncAgent.`);
        }
    }

    _emitToBus(bus, msg) {
        const hwId = msg.topic.split('/')[1];
        let data = { raw: msg.payload };

        try { data = JSON.parse(msg.payload); } catch (e) { }

        // Emit with original timestamp preserved
        bus.emit('telemetry:received', {
            hwId,
            ...data,
            ts: msg.bufferedAt
        });
    }
}

module.exports = EdgeBuffer;
