const logger = require('../utils/logger');
const express = require('express');
const { successResponse, errorResponse } = require('../utils/apiResponse');
const router = express.Router();
const LivestockMaster = require('../models/LivestockMaster');
const { protect } = require('../middleware/authMiddleware');

const Zone = require('../models/Zone');

router.get('/', protect, async (req, res) => {
    try {
        let baseQuery = {
            userId: req.user.tenantId,
            device_status: { $ne: 'Unassigned' }
        };
        if (req.user.type === 'staff') {
            const allowedZones = await Zone.find({ _id: { $in: req.user.assignedZones || [] } });
            baseQuery.zone_name = { $in: allowedZones.map(z => z.name) };
        }

        const livestockWithDevices = await LivestockMaster.find(baseQuery)
            .select('device_id name livestock_id device_type battery signal_strength last_updated device_status zone_name')
            .lean();

        const devices = livestockWithDevices.map(l => ({
            id: l.device_id,
            animal: l.name + ' (' + l.livestock_id + ')',
            type: l.device_type || 'GPS Sensor',
            battery: (l.battery !== null ? Math.round(l.battery) + '%' : 'N/A'),
            batteryRaw: l.battery,
            signal: l.signal_strength || -80,
            lastSync: l.last_updated ? new Date(l.last_updated).toLocaleString() : 'Never sync',
            status: l.device_status || 'Offline',
            location: l.zone_name || 'Unassigned'
        }));

        return successResponse(res, devices);
    } catch (error) {
        logger.error('Error fetching devices:', error);
        return errorResponse(res, 'FETCH_DEVICES_FAILED', 'Server error retrieving devices', 500, error.message);
    }
});

module.exports = router;
