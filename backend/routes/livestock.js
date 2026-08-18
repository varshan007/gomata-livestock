const logger = require('../utils/logger');
const express = require('express');
const { successResponse, errorResponse } = require('../utils/apiResponse');
const router = express.Router();
const LivestockMaster = require('../models/LivestockMaster');
const DeviceTelemetry = require('../models/DeviceTelemetry');
const Farm = require('../models/Farm');
const Zone = require('../models/Zone');
const mongoose = require('mongoose');
const { protect } = require('../middleware/authMiddleware');

// GET all livestock for Dashboard Grid
router.get('/', protect, async (req, res) => {
  try {
    let baseQuery = { userId: req.user.tenantId };
    if (req.user.type === 'staff') {
      const allowedZones = await Zone.find({ _id: { $in: req.user.assignedZones || [] } });
      baseQuery.zone_name = { $in: allowedZones.map(z => z.name) };
    }

    const livestock = await LivestockMaster.find(baseQuery)
      .select('livestock_id name breed temperature heart_rate battery signal_strength zone_name last_updated health_status farm_id zone_id last_location')
      .sort({ last_updated: -1 });

    // Map to exactly what the existing Dashboard.js frontend is expecting to see without touching React
    const mappedList = livestock.map(l => ({
      _id: l.livestock_id,
      livestock: {
        _id: l.livestock_id,
        name: l.name,
        tagNumber: l.name, // Spoofing tagNumber as name for filter logic
        breed: l.breed,
        type: l.species || 'Cow',
        gender: 'Female', // Default mock
        farmId: l.farm_id,
        zoneId: l.zone_id,
        location: l.zone_name || 'Unassigned',
        coords: l.last_location?.coordinates || null
      },
      latestSensorData: {
        temperature: l.temperature || 38.0,
        heartRate: l.heart_rate || 60,
        battery: l.battery || 100,
        signalStrength: l.signal_strength || -50,
        timestamp: l.last_updated
      },
      unresolvedAlerts: l.health_status === 'Normal' ? 0 : 1
    }));

    return successResponse(res, mappedList);
  } catch (error) {
    logger.error('[Livestock API] Error fetching livestock list: ', error);
    return errorResponse(res, 'FETCH_LIVESTOCK_FAILED', 'Failed to fetch livestock list', 500, error.message);
  }
});

// GET single livestock for Detail Page
router.get('/:id', protect, async (req, res) => {
  try {
    const { id } = req.params;
    let livestock = null;

    let baseQuery = { userId: req.user.tenantId };
    if (req.user.type === 'staff') {
      const allowedZones = await Zone.find({ _id: { $in: req.user.assignedZones || [] } });
      baseQuery.zone_name = { $in: allowedZones.map(z => z.name) };
    }

    // Fallback: If it's a valid 24-character hex string, check the _id first
    if (mongoose.Types.ObjectId.isValid(id)) {
      livestock = await LivestockMaster.findOne({ _id: id, ...baseQuery });
    }

    // Otherwise, or if not found by _id, check the custom livestock_id string
    if (!livestock) {
      livestock = await LivestockMaster.findOne({ livestock_id: id, ...baseQuery });
    }

    if (!livestock) {
      return errorResponse(res, 'LIVESTOCK_NOT_FOUND', `Livestock not found for ID: ${id}`, 404);
    }
    const l = livestock;

    // Fetch all related zones for the parent Farm to render on the Map
    let allZoneGeofences = [];
    try {
      const farm = await Farm.findOne({ userId: req.user._id, name: l.farm_name });
      if (farm) {
        const zones = await Zone.find({ farmId: farm._id });
        allZoneGeofences = zones.map(z => ({
          id: z._id.toString(),
          name: z.name,
          geofence: z.geofence
        }));
      }
    } catch (zoneErr) {
      logger.warn("[Livestock API] Non-fatal error resolving sibling zones:", zoneErr);
    }

    // Return exactly what LivestockDetail.js expects
    return successResponse(res, {
      _id: l.livestock_id,
      livestock: {
        _id: l.livestock_id,
        name: l.name,
        tagNumber: l.name,
        breed: l.breed,
        type: l.species || 'Cow',
        age: l.age,
        weight: l.weight,
        gender: 'Female',
        farmId: l.farm_id,
        farmName: l.farm_name,
        zoneId: l.zone_id,
        zoneName: l.zone_name,
        deviceId: l.device_id,
        deviceType: l.device_type,
        farmGeofence: l.farm_geofence,
        zoneGeofence: l.zone_geofence,
        allZoneGeofences: allZoneGeofences,
        lastLocation: l.last_location,
        vaccinationNotes: l.vaccination_notes,
        breedingNotes: l.breeding_notes,
        additionalNotes: l.additional_notes
      },
      latestSensorData: {
        temperature: l.temperature || 38.0,
        heartRate: l.heart_rate || 60,
        battery: l.battery || 100,
        signalStrength: l.signal_strength || -50,
        timestamp: l.last_updated
      },
      unresolvedAlerts: l.health_status === 'Normal' ? 0 : 1
    });
  } catch (error) {
    logger.error(`[Livestock API] Error fetching livestock details for ${req.params.id}: `, error);
    return errorResponse(res, 'FETCH_LIVESTOCK_DETAIL_FAILED', 'Failed to fetch livestock details', 500, error.message);
  }
});

// GET Temperature History for 48h Graph
router.get('/:id/temperature-history', protect, async (req, res) => {
  try {
    const { id } = req.params;
    let livestock = null;

    let baseQuery = { userId: req.user.tenantId };
    if (req.user.type === 'staff') {
      const allowedZones = await Zone.find({ _id: { $in: req.user.assignedZones || [] } });
      baseQuery.zone_name = { $in: allowedZones.map(z => z.name) };
    }

    if (mongoose.Types.ObjectId.isValid(id)) {
      livestock = await LivestockMaster.findOne({ _id: id, ...baseQuery });
    }
    if (!livestock) {
      livestock = await LivestockMaster.findOne({ livestock_id: id, ...baseQuery });
    }

    if (!livestock) {
      return errorResponse(res, 'LIVESTOCK_NOT_FOUND', `Livestock not found for ID: ${id}`, 404);
    }

    // Pull the underlying telemetry series directly from DeviceTelemetry
    const history = await DeviceTelemetry.find({ deviceId: livestock.device_id })
      .sort({ timestamp: -1 })
      .limit(48); // Limiting to latest 48 points to emulate "hours"

    const mappedHistory = history.map(h => ({
      timestamp: h.timestamp,
      temperature: h.temperature
    })).reverse(); // Oldest first for line chart

    return successResponse(res, mappedHistory);
  } catch (error) {
    logger.error(`[Livestock API] Error fetching history for ${req.params.id}: `, error);
    return errorResponse(res, 'FETCH_TEMP_HISTORY_FAILED', 'Failed to fetch history', 500, error.message);
  }
});

// Phase 9: Breed Aggregation Summary
router.get('/breeds/summary', protect, async (req, res) => {
  try {
    let baseQuery = { userId: req.user.tenantId };
    if (req.user.type === 'staff') {
      const allowedZones = await Zone.find({ _id: { $in: req.user.assignedZones || [] } });
      baseQuery.zone_name = { $in: allowedZones.map(z => z.name) };
    }

    const summary = await LivestockMaster.aggregate([
      { $match: baseQuery },
      {
        $group: {
          _id: "$breed",
          species: { $first: "$species" },
          count: { $sum: 1 },
          dateAdded: { $min: "$createdAt" }
        }
      },
      {
        $project: {
          breed: { $ifNull: ["$_id", "Unknown"] },
          species: { $ifNull: ["$species", "Unknown"] },
          count: 1,
          dateAdded: 1,
          _id: 0
        }
      }
    ]);
    return successResponse(res, summary);
  } catch (error) {
    logger.error('Error grouping breeds:', error);
    return errorResponse(res, 'FETCH_BREED_SUMMARY_FAILED', 'Server error retrieving breed summary', 500, error.message);
  }
});

module.exports = router;