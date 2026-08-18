const logger = require('../utils/logger');
const express = require('express');
const { successResponse, errorResponse } = require('../utils/apiResponse');
const router = express.Router();
const LivestockMaster = require('../models/LivestockMaster');
const { protect } = require('../middleware/authMiddleware');

// GET /api/dashboard/summary
router.get('/summary', protect, async (req, res) => {
    try {
        const total_animals = await LivestockMaster.countDocuments({ userId: req.user.tenantId });

        // Let's add some extra counts just in case the UI wants to show active vs alerts
        const active_devices = await LivestockMaster.countDocuments({ userId: req.user.tenantId, device_status: 'Active' });
        const alerts_count = await LivestockMaster.countDocuments({ userId: req.user.tenantId, health_status: { $ne: 'Normal' } });

        return successResponse(res, {
            total_animals,
            active_devices,
            alerts_count
        });
    } catch (error) {
        logger.error('[Dashboard API] Error generating summary: ', error);
        return errorResponse(res, 'DASHBOARD_SUMMARY_FAILED', 'Internal Server Error', 500, error.message);
    }
});

module.exports = router;
