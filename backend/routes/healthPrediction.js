const logger = require('../utils/logger');
const express = require('express');
const { successResponse, errorResponse } = require('../utils/apiResponse');
const router = express.Router();
const Alert = require('../models/Alert');
const LivestockMaster = require('../models/LivestockMaster');
const { protect } = require('../middleware/authMiddleware');
const Redis = require('ioredis');

const redis = new Redis(process.env.REDIS_URL || 'redis://127.0.0.1:6379');

/**
 * GET /api/health-prediction/:livestockId
 * Returns real-time health risk prediction for a specific animal.
 * Combines: Redis risk score + Redis explanation cache + MongoDB alert data.
 * Enforces multi-tenant isolation via userId.
 */
router.get('/:livestockId', protect, async (req, res) => {
    try {
        const { livestockId } = req.params;
        const tenantId = req.user.tenantId;

        // 0. Verify ownership — only show predictions for user's own livestock
        const livestock = await LivestockMaster.findOne({
            _id: livestockId,
            userId: tenantId
        }).select('name breed farm_name zone_name device_id').lean();

        if (!livestock) {
            return errorResponse(res, 'NOT_FOUND', 'Livestock not found or access denied', 404);
        }

        // 1. Read cached risk score from Redis
        let riskData = null;
        try {
            const riskRaw = await redis.get(`risk:${livestockId}`) || await redis.get(`healthRisk:${livestockId}`);
            if (riskRaw) {
                riskData = JSON.parse(riskRaw);
            }
        } catch (e) {
            logger.warn('Failed to read risk cache:', e.message);
        }

        // 2. Fetch latest unresolved alert for this animal (tenant-scoped)
        const latestAlert = await Alert.findOne({
            livestockId,
            userId: tenantId,
            resolved: false
        }).sort({ timestamp: -1 }).lean();

        // 3. Read cached LLM explanation if alert exists
        let explanation = '';
        if (latestAlert) {
            explanation = latestAlert.explanation || '';

            // Try Redis cache for fresher explanation
            if (!explanation) {
                try {
                    const expRaw = await redis.get(`explanation:${latestAlert._id}`);
                    if (expRaw) {
                        const parsed = JSON.parse(expRaw);
                        explanation = parsed.explanation || '';
                    }
                } catch (e) {
                    // Ignore cache miss
                }
            }
        }

        // 4. Read features for additional context
        let features = null;
        try {
            // Try tenant-scoped key first, then fall back
            const featRaw = await redis.get(`features:v3:${tenantId}:${livestockId}`) ||
                await redis.get(`features:${livestockId}`);
            if (featRaw) {
                features = JSON.parse(featRaw);
            }
        } catch (e) {
            // Ignore
        }

        // 5. Build response — use LivestockMaster data as primary source for metadata
        const response = {
            animalId: livestockId,
            riskScore: riskData?.risk_score || riskData?.disease_prob || (latestAlert?.diseaseProbability) || 0,
            severity: riskData?.severity || latestAlert?.severity || 'Normal',
            diseaseProbability: riskData?.disease_prob || latestAlert?.diseaseProbability || 0,
            explanation: explanation,
            alertId: latestAlert?._id || null,
            alertType: latestAlert?.alertType || null,
            // Use LivestockMaster as primary source for animal metadata
            animalName: livestock.name || latestAlert?.animalName || '',
            farmName: livestock.farm_name || latestAlert?.farmName || '',
            zoneName: livestock.zone_name || latestAlert?.zoneName || '',
            breed: livestock.breed || latestAlert?.breed || '',
            lastUpdated: riskData?.timestamp || latestAlert?.timestamp || null,
            currentVitals: features ? {
                temperature: features.temperature_mean || features.temp_current || features.temperature || null,
                heartRate: features.heart_rate_mean || features.hr_current || features.heart_rate || null,
                activityIndex: features.activity_index_mean || features.activity_current || features.activity_index || null
            } : null
        };

        return successResponse(res, response);

    } catch (error) {
        logger.error('Error fetching health prediction:', error);
        return errorResponse(res, 'HEALTH_PREDICTION_ERROR', 'Failed to fetch health prediction', 500, error.message);
    }
});

module.exports = router;

