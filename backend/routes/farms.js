const logger = require('../utils/logger');
const express = require('express');
const { successResponse, errorResponse } = require('../utils/apiResponse');
const router = express.Router();
const Farm = require('../models/Farm');
const Zone = require('../models/Zone');
const { protect } = require('../middleware/authMiddleware');

router.get('/', protect, async (req, res) => {
    try {
        let farmLookup = { userId: req.user.tenantId };
        if (req.user.type === 'staff') {
            farmLookup._id = { $in: req.user.assignedFarms || [] };
        }

        const farms = await Farm.find(farmLookup).lean();

        let zoneLookup = { farmId: { $in: farms.map(f => f._id) } };
        if (req.user.type === 'staff') {
            zoneLookup._id = { $in: req.user.assignedZones || [] };
        }

        const zones = await Zone.find(zoneLookup).lean();

        // Match zones to farms and calculate approximate areas
        const farmsWithZones = farms.map(farm => {
            farm.zones = zones.filter(z => z.farmId.toString() === farm._id.toString()).map(z => {
                let areaSize = 0;
                if (z.locationType === 'Circular Mapping' && z.geofence && z.geofence.radius) {
                    // Circle area mapped to acres
                    areaSize = (Math.PI * Math.pow(z.geofence.radius, 2)) / 4046.86;
                } else if (z.geofence && z.geofence.type === 'Polygon' && z.geofence.coordinates && z.geofence.coordinates.length > 0) {
                    // Simple bounding box area approximation
                    const lats = z.geofence.coordinates[0].map(c => c[1]);
                    const lngs = z.geofence.coordinates[0].map(c => c[0]);
                    const height = (Math.max(...lats) - Math.min(...lats)) * 111320; // rough meters length from lat delta
                    const width = (Math.max(...lngs) - Math.min(...lngs)) * 111320 * Math.cos(Math.min(...lats) * Math.PI / 180); // meters length from lng delta
                    areaSize = (height * width) / 4046.86; // approximate acres
                }
                return {
                    ...z,
                    areaSize: parseFloat(areaSize.toFixed(2)) // Round to 2 decimal places
                };
            });
            return farm;
        });

        return successResponse(res, farmsWithZones);
    } catch (error) {
        logger.error('Error fetching farms:', error);
        return errorResponse(res, 'FETCH_FARMS_FAILED', 'Server error retrieving farms', 500, error.message);
    }
});

module.exports = router;
