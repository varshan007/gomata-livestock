const redisConnection = require('../config/redis');

class FeatureStoreClient {
    /**
     * Rebuild the Redis key format based on Tenant and Animal IDs (v2 — legacy)
     */
    _getKey(tenantId, animalId) {
        return `features:v2:${tenantId}:${animalId}`;
    }

    /**
     * v3 key format — matches XGBoost 23-feature model
     */
    _getKeyV3(tenantId, animalId) {
        return `features:v3:${tenantId}:${animalId}`;
    }

    /**
     * Retrieve a feature vector from Redis Feature Store (v2 — legacy).
     */
    async getFeatures(tenantId, animalId) {
        const key = this._getKey(tenantId, animalId);
        const data = await redisConnection.get(key);
        if (!data) return null;

        try {
            return JSON.parse(data);
        } catch (error) {
            console.error(`[FeatureStoreClient] Error parsing feature vector for ${key}`, error);
            return null;
        }
    }

    /**
     * Retrieve v3 ML features from Redis (23-feature XGBoost vector).
     * Falls back to null if not cached or expired.
     */
    async getFeaturesV3(tenantId, animalId) {
        const key = this._getKeyV3(tenantId, animalId);
        const data = await redisConnection.get(key);
        if (!data) return null;

        try {
            return JSON.parse(data);
        } catch (error) {
            console.error(`[FeatureStoreClient] Error parsing v3 features for ${key}`, error);
            return null;
        }
    }

    /**
     * Write a complete feature vector to Redis with a TTL (v2 — legacy).
     */
    async setFeatures(tenantId, animalId, features) {
        const key = this._getKey(tenantId, animalId);
        await redisConnection.set(key, JSON.stringify(features), 'EX', 300);
    }

    /**
     * Write v3 ML features to Redis with a TTL.
     */
    async setFeaturesV3(tenantId, animalId, features) {
        const key = this._getKeyV3(tenantId, animalId);
        await redisConnection.set(key, JSON.stringify(features), 'EX', 300);
    }
}

module.exports = new FeatureStoreClient();
