const logger = require('../../../utils/logger');
const { point, polygon } = require('@turf/helpers');
const booleanPointInPolygon = require('@turf/boolean-point-in-polygon').default;
const Geofence = require('../../../models/Geofence');
const Livestock = require('../../../models/Livestock');
const crypto = require('crypto');

class LocationAgent {
    constructor(bus) {
        this.bus = bus;
    }

    start() {
        logger.info('[LocationAgent] Started. Monitoring GPS telemetry for geofence breaches...');

        this.bus.on('telemetry:received', async (payload) => {
            const telemetryData = payload.data ? payload.data : payload; // Handle different payload structures
            const traceId = payload.traceId || crypto.randomUUID();
            const { hwId, lat, lng } = telemetryData;

            if (!lat || !lng) return; // Need valid GPS coordinates

            try {
                // Find livestock by hardware ID
                const queryConditions = [{ deviceId: hwId }, { tagNumber: hwId }];
                if (hwId.length === 24) {
                    queryConditions.push({ _id: hwId });
                }

                const livestock = await Livestock.findOne({ $or: queryConditions });

                if (!livestock) return;

                // Find global or livestock-specific Safe polygons
                const geofences = await Geofence.find({
                    $or: [
                        { livestockId: livestock._id },
                        { livestockId: null }
                    ],
                    type: 'Safe',
                    shape: 'Polygon'
                });

                if (geofences.length === 0) return;

                // Turf uses [longitude, latitude]
                const cowPoint = point([lng, lat]);

                for (const fence of geofences) {
                    if (fence.coordinates && fence.coordinates.length >= 3) {
                        const coords = fence.coordinates.map(c => [c.longitude, c.latitude]);

                        // Polygon must be closed (first and last coordinate match)
                        const first = coords[0];
                        const last = coords[coords.length - 1];
                        if (first[0] !== last[0] || first[1] !== last[1]) {
                            coords.push([...first]);
                        }

                        const poly = polygon([coords]);
                        const isInside = booleanPointInPolygon(cowPoint, poly);

                        if (!isInside) {
                            logger.info(`[LocationAgent] 🚨 Geofence Breach for ${hwId} escaping ${fence.name}`);

                            this.bus.emit('alert:create', {
                                hwId,
                                type: 'Geofence',
                                severity: 'High',
                                message: `${livestock.name || hwId} has escaped the safe zone (${fence.name}).`,
                                action: `Intercept and return the animal to the designated safe zone immediately.`,
                                metrics: { lat, lng }
                            }, traceId);

                            // Only emit once per telemetry reading to avoid duplicate alerts for overlapping zones
                            break;
                        }
                    }
                }
            } catch (error) {
                logger.error('[LocationAgent] Error checking geofences:', error);
            }
        });
    }
}

module.exports = LocationAgent;
