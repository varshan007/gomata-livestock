const express = require('express');
const { successResponse, errorResponse } = require('../utils/apiResponse');
const logger = require('../utils/logger');
const router = express.Router();
const SensorData = require('../models/SensorData');
const Alert = require('../models/Alert');
const { protect } = require('../middleware/authMiddleware');

// POST sensor data (from ESP32 or ingestion agents)
router.post('/', async (req, res) => {
  if (req.headers['x-api-key'] !== process.env.API_KEY) {
    return errorResponse(res, 'UNAUTHORIZED_HARDWARE', 'Unauthorized hardware', 401);
  }
  try {
    const { livestockId, deviceId, temperature } = req.body;
    const sensorData = new SensorData({
      livestockId: livestockId,
      deviceId: deviceId,
      temperature: temperature,
      latitude: req.body.latitude,
      longitude: req.body.longitude,
      batteryLevel: req.body.batteryLevel
    });

    const savedData = await sensorData.save();

    logger.info({ action: 'telemetry_ingestion', result: 'success', animalId: livestockId || deviceId, temp: temperature, type: 'api' }, `Telemetry ingested manually via API for ${livestockId || deviceId}`);

    // Check for temperature alerts
    await checkTemperatureAlert(savedData);

    return successResponse(res, savedData, 'Sensor data created successfully', 201);
  } catch (error) {
    return errorResponse(res, 'INGEST_SENSOR_DATA_FAILED', 'Failed to ingest sensor data', 400, error.message);
  }
});

// GET sensor data history for a livestock
router.get('/livestock/:livestockId', protect, async (req, res) => {
  try {
    const { hours = 24 } = req.query;
    const timeAgo = new Date(Date.now() - hours * 60 * 60 * 1000);

    const data = await SensorData.find({
      livestockId: req.params.livestockId,
      timestamp: { $gte: timeAgo }
    }).sort({ timestamp: -1 });

    return successResponse(res, data);
  } catch (error) {
    return errorResponse(res, 'FETCH_SENSOR_HISTORY_FAILED', 'Failed to fetch sensor data', 500, error.message);
  }
});

// GET latest sensor data for a livestock
router.get('/livestock/:livestockId/latest', protect, async (req, res) => {
  try {
    const latestData = await SensorData.findOne({
      livestockId: req.params.livestockId
    }).sort({ timestamp: -1 });

    // Return empty object if no data found, instead of null, to avoid frontend crashes if not handled
    if (!latestData) {
      return successResponse(res, null);
    }
    return successResponse(res, latestData);
  } catch (error) {
    return errorResponse(res, 'FETCH_LATEST_SENSOR_FAILED', 'Failed to fetch latest sensor data', 500, error.message);
  }
});

// GET stats for a livestock
router.get('/livestock/:livestockId/stats', protect, async (req, res) => {
  try {
    const { hours = 24 } = req.query;
    const timeAgo = new Date(Date.now() - hours * 60 * 60 * 1000);

    // Aggregate to calculate min, max, avg
    const stats = await SensorData.aggregate([
      {
        $match: {
          livestockId: new require('mongoose').Types.ObjectId(req.params.livestockId),
          timestamp: { $gte: timeAgo }
        }
      },
      {
        $group: {
          _id: null,
          average: { $avg: "$temperature" },
          max: { $max: "$temperature" },
          min: { $min: "$temperature" }
        }
      }
    ]);

    if (stats.length > 0) {
      return successResponse(res, {
        average: parseFloat(stats[0].average.toFixed(1)),
        max: stats[0].max,
        min: stats[0].min
      });
    } else {
      return successResponse(res, { average: 0, max: 0, min: 0 });
    }
  } catch (error) {
    return errorResponse(res, 'FETCH_STATS_FAILED', 'Failed to fetch stats', 500, error.message);
  }
});

const Livestock = require('../models/Livestock'); // Moved to top

// GET dashboard overview
router.get('/dashboard/overview', protect, async (req, res) => {
  try {
    // Determine filter: if user has farmId, use it. For now, fetch ALL to debug.
    // Removed { status: 'Active' } to ensure we see the animals just created.
    const allLivestock = await Livestock.find({ userId: req.user.tenantId }).sort({ createdAt: -1 });

    logger.info(`Dashboard: Found ${allLivestock.length} animals`);

    const overview = await Promise.all(allLivestock.map(async (animal) => {
      try {
        const latestSensorData = await SensorData.findOne({ livestockId: animal._id })
          .sort({ timestamp: -1 });

        const unresolvedAlerts = await Alert.countDocuments({
          livestockId: animal._id,
          resolved: false
        });

        return {
          livestock: animal,
          latestSensorData: latestSensorData || null, // Ensure null if undefined
          unresolvedAlerts: unresolvedAlerts || 0
        };
      } catch (err) {
        logger.error(`Error processing animal ${animal._id}:`, err);
        return { livestock: animal, latestSensorData: null, unresolvedAlerts: 0 };
      }
    }));

    return successResponse(res, overview);
  } catch (error) {
    logger.error("Dashboard Overview Error:", error);
    return errorResponse(res, 'DASHBOARD_OVERVIEW_FAILED', 'Failed to fetch dashboard overview', 500, error.message);
  }
});

// Helper function to check temperature and create alerts
async function checkTemperatureAlert(sensorData) {
  const temp = sensorData.temperature;
  let alertType = null;
  let severity = null;
  let message = null;

  // Cattle normal temp: 38-39°C
  if (temp > 40) {
    severity = 'Critical';
    message = `Critical: Temperature ${temp}°C - Immediate veterinary attention needed`;
  } else if (temp >= 39.5) {
    severity = 'High';
    message = `Warning: Temperature ${temp}°C - Monitor for fever`;
  } else if (temp < 37.5) {
    severity = 'Medium';
    message = `Alert: Low temperature ${temp}°C - Check for hypothermia`;
  }

  if (severity) {
    const alert = new Alert({
      livestockId: sensorData.livestockId,
      alertType: 'Temperature',
      severity: severity,
      message: message
    });
    await alert.save();
  }
}

// GET historical path (lighter payload for map)
router.get('/livestock/:livestockId/path', protect, async (req, res) => {
  try {
    const { hours = 24 } = req.query;
    const timeAgo = new Date(Date.now() - hours * 60 * 60 * 1000);

    const data = await SensorData.find({
      livestockId: req.params.livestockId,
      timestamp: { $gte: timeAgo }
    }).select('latitude longitude timestamp').sort({ timestamp: 1 }); // Sorted chronologically for path

    return successResponse(res, data);
  } catch (error) {
    return errorResponse(res, 'FETCH_LIVESTOCK_PATH_FAILED', 'Failed to fetch livestock path', 500, error.message);
  }
});

// GET movement analytics (distance, activity)
router.get('/livestock/:livestockId/analytics', protect, async (req, res) => {
  try {
    const { hours = 24 } = req.query;
    const timeAgo = new Date(Date.now() - hours * 60 * 60 * 1000);

    const data = await SensorData.find({
      livestockId: req.params.livestockId,
      timestamp: { $gte: timeAgo }
    }).sort({ timestamp: 1 });

    if (data.length < 2) {
      return successResponse(res, { totalDistance: 0, avgSpeed: 0, activityLevel: { active: 0, resting: 0 } });
    }

    let totalDistance = 0;
    let activePoints = 0;
    let restingPoints = 0;

    for (let i = 1; i < data.length; i++) {
      const p1 = data[i - 1];
      const p2 = data[i];

      // Simple Haversine calculation
      const R = 6371; // Earth radius in km
      const dLat = (p2.latitude - p1.latitude) * Math.PI / 180;
      const dLon = (p2.longitude - p1.longitude) * Math.PI / 180;
      const a =
        Math.sin(dLat / 2) * Math.sin(dLat / 2) +
        Math.cos(p1.latitude * Math.PI / 180) * Math.cos(p2.latitude * Math.PI / 180) *
        Math.sin(dLon / 2) * Math.sin(dLon / 2);
      const c = 2 * Math.atan2(Math.sqrt(a), Math.sqrt(1 - a));
      const dist = R * c;

      totalDistance += dist;

      // Simple activity classification based on distance between points
      // Assuming points are ~1 hour apart in seeded data (or 10s in real time)
      if (dist > 0.05) activePoints++; // Arbitrary threshold for now
      else restingPoints++;
    }

    return successResponse(res, {
      totalDistance: parseFloat(totalDistance.toFixed(2)),
      activityBreakdown: {
        active: activePoints,
        resting: restingPoints
      }
    });

  } catch (error) {
    return errorResponse(res, 'FETCH_ANALYTICS_FAILED', 'Failed to fetch analytics', 500, error.message);
  }
});

module.exports = router;