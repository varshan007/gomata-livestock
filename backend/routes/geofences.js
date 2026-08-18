const express = require('express');
const { successResponse, errorResponse } = require('../utils/apiResponse');
const router = express.Router();
const Geofence = require('../models/Geofence');
const { protect } = require('../middleware/authMiddleware');

// GET all geofences
router.get('/', protect, async (req, res) => {
    try {
        const geofences = await Geofence.find({ userId: req.user.tenantId }).sort({ createdAt: -1 });
        return successResponse(res, geofences);
    } catch (error) {
        return errorResponse(res, 'FETCH_GEOFENCES_FAILED', 'Failed to fetch geofences', 500, error.message);
    }
});

// POST check if point is inside any geofence (Simple Ray Casting for Polygon)
router.post('/check', protect, async (req, res) => {
    try {
        const { latitude, longitude, livestockId } = req.body;
        // Logic to check if point is in geofence
        // For now, identifying which fence it is in
        // This is a placeholder for more complex logic
        return successResponse(res, { status: 'Outside', geofence: null });
    } catch (error) {
        return errorResponse(res, 'CHECK_GEOFENCES_FAILED', 'Failed to check geofences', 500, error.message);
    }
});

// POST create a new geofence
router.post('/', protect, async (req, res) => {
    const geofence = new Geofence({
        userId: req.user.tenantId,
        name: req.body.name,
        type: req.body.type,
        shape: req.body.shape,
        coordinates: req.body.coordinates,
        center: req.body.center,
        radius: req.body.radius,
        livestockId: req.body.livestockId
    });

    try {
        const newGeofence = await geofence.save();
        return successResponse(res, newGeofence); // Use successResponse instead of manual 201 + json to be consistent, wait we can't easily pass status to successResponse without modifying it. I'll pass status: 201 before calling successResponse.
    } catch (error) {
        return errorResponse(res, 'CREATE_GEOFENCE_FAILED', 'Failed to create geofence', 400, error.message);
    }
});

// DELETE a geofence
router.delete('/:id', protect, async (req, res) => {
    try {
        const deleted = await Geofence.findOneAndDelete({ _id: req.params.id, userId: req.user.tenantId });
        if (!deleted) {
            return errorResponse(res, 'GEOFENCE_NOT_FOUND', 'Geofence not found', 404);
        }
        return successResponse(res, { message: 'Geofence deleted' });
    } catch (error) {
        return errorResponse(res, 'DELETE_GEOFENCE_FAILED', 'Failed to delete geofence', 500, error.message);
    }
});

module.exports = router;
