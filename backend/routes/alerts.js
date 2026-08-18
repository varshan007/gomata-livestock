const express = require('express');
const { successResponse, errorResponse } = require('../utils/apiResponse');
const router = express.Router();
const Alert = require('../models/Alert');
const Livestock = require('../models/Livestock');
const LivestockMaster = require('../models/LivestockMaster');
const { protect } = require('../middleware/authMiddleware');

// GET all alerts
router.get('/', protect, async (req, res) => {
    try {
        const { resolved } = req.query;
        let filter = {};

        // If 'staff', they only see alerts for livestock in their assigned zones
        if (req.user.type === 'staff') {
            const allowedLivestock = await Livestock.find({ zoneId: { $in: req.user.assignedZones || [] }, userId: req.user.tenantId }).select('_id');
            const allowedMaster = await LivestockMaster.find({ userId: req.user.tenantId }).select('_id');
            const allIds = [...allowedLivestock.map(l => l._id), ...allowedMaster.map(l => l._id)];
            filter.livestockId = { $in: allIds };
        } else {
            const adminLivestock = await Livestock.find({ userId: req.user.tenantId }).select('_id');
            const adminMaster = await LivestockMaster.find({ userId: req.user.tenantId }).select('_id');
            const allIds = [...adminLivestock.map(l => l._id), ...adminMaster.map(l => l._id)];
            filter.livestockId = { $in: allIds };
        }

        filter.userId = req.user.tenantId;

        // If resolved query param is provided, filter by it
        if (resolved !== undefined) {
            filter.resolved = resolved === 'true';
        }

        const alerts = await Alert.find(filter)
            .sort({ timestamp: -1 })
            .populate('livestockId', 'name tagNumber livestock_id'); // Populate livestock details

        return successResponse(res, alerts);
    } catch (error) {
        return errorResponse(res, 'FETCH_ALERTS_FAILED', 'Failed to fetch alerts', 500, error.message);
    }
});

// GET alerts for specific livestock
router.get('/livestock/:livestockId', protect, async (req, res) => {
    try {
        const alerts = await Alert.find({ livestockId: req.params.livestockId, userId: req.user.tenantId })
            .sort({ timestamp: -1 });
        return successResponse(res, alerts);
    } catch (error) {
        return errorResponse(res, 'FETCH_LIVESTOCK_ALERTS_FAILED', 'Failed to fetch livestock alerts', 500, error.message);
    }
});

// GET count of unresolved alerts
router.get('/unresolved/count', protect, async (req, res) => {
    try {
        let filter = { resolved: false, userId: req.user.tenantId };
        if (req.user.type === 'staff') {
            const allowedLivestock = await Livestock.find({ zoneId: { $in: req.user.assignedZones || [] }, userId: req.user.tenantId }).select('_id');
            filter.livestockId = { $in: allowedLivestock.map(l => l._id) };
        }

        const count = await Alert.countDocuments(filter);
        return successResponse(res, { count });
    } catch (error) {
        return errorResponse(res, 'FETCH_ALERTS_COUNT_FAILED', 'Failed to get unresolved alerts count', 500, error.message);
    }
});

// MARK alert as resolved
router.patch('/:id/resolve', protect, async (req, res) => {
    try {
        const alert = await Alert.findOne({ _id: req.params.id, userId: req.user.tenantId });
        if (!alert) {
            return errorResponse(res, 'ALERT_NOT_FOUND', 'Alert not found', 404);
        }

        alert.resolved = true;
        alert.status = 'Resolved';
        const updatedAlert = await alert.save();
        return successResponse(res, updatedAlert);
    } catch (error) {
        return errorResponse(res, 'RESOLVE_ALERT_FAILED', 'Failed to resolve alert', 500, error.message);
    }
});

// ACKNOWLEDGE alert
router.put('/:id/acknowledge', protect, async (req, res) => {
    try {
        // req.user has _id
        const userId = req.user._id;
        const alert = await Alert.findOne({ _id: req.params.id, userId: req.user.tenantId });
        if (!alert) {
            return errorResponse(res, 'ALERT_NOT_FOUND', 'Alert not found', 404);
        }

        alert.status = 'Acknowledged';
        alert.acknowledgedAt = new Date();
        if (userId) alert.assignedTo = userId;

        const updatedAlert = await alert.save();
        return successResponse(res, updatedAlert);
    } catch (error) {
        return errorResponse(res, 'ACKNOWLEDGE_ALERT_FAILED', 'Failed to acknowledge alert', 500, error.message);
    }
});

// DELETE alert
router.delete('/:id', protect, async (req, res) => {
    try {
        if (req.user.type === 'staff' && req.user.staffRole === 'Viewer') {
            return errorResponse(res, 'UNAUTHORIZED_ACTION', 'Viewers cannot delete alerts', 403);
        }

        const alert = await Alert.findOne({ _id: req.params.id, userId: req.user.tenantId });
        if (!alert) {
            return errorResponse(res, 'ALERT_NOT_FOUND', 'Alert not found', 404);
        }

        await alert.deleteOne();
        return successResponse(res, { message: 'Alert deleted' });
    } catch (error) {
        return errorResponse(res, 'DELETE_ALERT_FAILED', 'Failed to delete alert', 500, error.message);
    }
});

module.exports = router;
