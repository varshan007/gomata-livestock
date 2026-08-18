const logger = require('../utils/logger');
const express = require('express');
const { successResponse, errorResponse } = require('../utils/apiResponse');
const router = express.Router();
const mongoose = require('mongoose');

// Models
const User = require('../models/User');
const Farm = require('../models/Farm');
const Zone = require('../models/Zone');
const Livestock = require('../models/Livestock');
const Device = require('../models/Device');
const LivestockMaster = require('../models/LivestockMaster');

// Services
const IdGeneratorService = require('../services/IdGeneratorService');

// Auth Utils
const jwt = require('jsonwebtoken');
const generateToken = (id) => jwt.sign({ id, role: 'admin' }, process.env.JWT_SECRET, { expiresIn: '30d' });

function formatGeoJSON(geofence) {
    if (!geofence || !geofence.type) return null;

    // Fast-path: Frontend already sent valid GeoJSON structure
    if (geofence.coordinates) {
        return {
            type: geofence.type,
            coordinates: geofence.coordinates,
            ...(geofence.radius ? { radius: geofence.radius } : {})
        };
    }

    // Legacy parser fallback
    if (geofence.type === 'Polygon' && Array.isArray(geofence.points)) {
        return {
            type: "Polygon",
            coordinates: [geofence.points.map(p => [(p.lng || p[1]), (p.lat || p[0])])]
        };
    } else if (geofence.type === 'Point' && geofence.center) {
        return {
            type: "Point",
            coordinates: [(geofence.center.lng || geofence.center[1]), (geofence.center.lat || geofence.center[0])],
            radius: geofence.radius || 0
        };
    }
    return null;
}

router.post('/initialize', async (req, res) => {
    try {
        const { name, email, password, phone, dob, farms, zones, livestock } = req.body;

        // Collect newly generated Sequence IDs for the frontend animation
        const assignedIds = {
            farms: [],
            zones: [],
            livestock: []
        };

        logger.info(`[SYSTEM INIT] Starting pipeline for ${email}`);

        // 1. Device Verification
        // Strict verification: Does this onboarding hardware ID actually exist in our master Device collection?
        const hardwareIds = livestock.map(l => l.deviceId).filter(Boolean);
        const verifiedDevices = await Device.find({ deviceId: { $in: hardwareIds } });
        const verifiedIds = verifiedDevices.map(d => d.deviceId);

        const unverified = hardwareIds.filter(id => !verifiedIds.includes(id));
        if (unverified.length > 0) {
            if (process.env.NODE_ENV !== "production") {
                logger.info(`[SYSTEM INIT] DEV MODE BYPASS: Ignored unverified devices: ${unverified.join(', ')}`);
            } else {
                logger.info(`[SYSTEM INIT] Unverified Devices block: ${unverified.join(', ')}`);
                return errorResponse(res, 'DEVICES_NOT_VERIFIED', `Verification Failed. The following device serial numbers are invalid or unregistered in GoMata hardware control: ${unverified.join(', ')}`, 400);
            }
        }

        // Mark these devices as assigned
        await Device.updateMany(
            { deviceId: { $in: verifiedIds } },
            { $set: { status: 'Active' } }
        );

        // 2. User Creation (Legacy logic persistence)
        const userExists = await User.findOne({ email });
        if (userExists) {
            return errorResponse(res, 'USER_EXISTS', 'User already exists', 400);
        }

        const user = await User.create([{ name, email, password, phone, dob }]);
        const createdUser = user[0];

        // Mappings dictionaries to tie the dynamic React tempIds to absolute DB refs
        const farmMap = {};
        const zoneMap = {};

        // 3. Farm Generation
        for (const f of (farms || [])) {
            const generatedFarmId = await IdGeneratorService.generateFarmId();
            assignedIds.farms.push(generatedFarmId);

            const formattedGeo = formatGeoJSON(f.geofence);

            const newFarm = new Farm({
                userId: createdUser._id,
                name: f.name,
                locationType: (f.locationType === 'Circle' || f.locationType === 'Circular Mapping') ? 'Circular Mapping' : 'Polygon Mapping',
                geofence: formattedGeo
            });
            await newFarm.save();

            farmMap[f.tempId] = {
                objectId: newFarm._id,
                seqId: generatedFarmId,
                name: newFarm.name,
                geo: formattedGeo
            };
        }

        // 4. Zone Generation
        for (const z of (zones || [])) {
            const parentFarm = farmMap[z.farmTempId];
            if (!parentFarm) continue;

            const generatedZoneId = await IdGeneratorService.generateZoneId(parentFarm.seqId, parentFarm.name);
            assignedIds.zones.push(generatedZoneId);

            const formattedGeo = formatGeoJSON(z.geofence);

            const newZone = new Zone({
                farmId: parentFarm.objectId,
                name: z.name,
                locationType: (z.locationType === 'Circle' || z.locationType === 'Circular Mapping') ? 'Circular Mapping' : 'Polygon Mapping',
                geofence: formattedGeo
            });
            await newZone.save();

            zoneMap[z.tempId] = {
                objectId: newZone._id,
                seqId: generatedZoneId,
                name: newZone.name,
                geo: formattedGeo
            };
        }

        // 5. Livestock & LivestockMaster Mapping
        for (const l of (livestock || [])) {
            const farmContext = farmMap[l.farmTempId];
            let zoneContext = zoneMap[l.zoneTempId];

            if (!zoneContext && farmContext) {
                const generatedZoneId = await IdGeneratorService.generateZoneId(farmContext.seqId, farmContext.name);
                assignedIds.zones.push(generatedZoneId);
                const newZone = new Zone({
                    farmId: farmContext.objectId,
                    name: 'Default Zone',
                    locationType: 'Polygon Mapping',
                    geofence: farmContext.geo
                });
                await newZone.save();
                zoneContext = {
                    objectId: newZone._id,
                    seqId: generatedZoneId,
                    name: newZone.name,
                    geo: farmContext.geo
                };
                zoneMap[l.zoneTempId || 'default'] = zoneContext;
            }

            const parsedAge = parseFloat(String(l.age).replace(/[^0-9.]/g, '')) || 0;
            const parsedWeight = parseFloat(String(l.weight).replace(/[^0-9.]/g, '')) || 0;

            // Legacy Table
            const newLivestock = new Livestock({
                tagNumber: l.name, // Usually tagNumber maps closely to name in demo
                name: l.name,
                breed: l.breed,
                type: l.type,
                age: parsedAge,
                weight: parsedWeight,
                deviceId: l.deviceId,
                deviceType: l.deviceType,
                farmId: farmContext.objectId,
                zoneId: zoneContext?.objectId || null,
                vaccinationNotes: l.vaccinationNotes,
                breedingNotes: l.breedingNotes,
                additionalNotes: l.additionalNotes
            });
            await newLivestock.save();

            // Atomic Automated Seq ID Generation using Abbreviations
            const generatedLivestockId = await IdGeneratorService.generateLivestockId(farmContext.seqId, zoneContext.seqId, farmContext.name, zoneContext.name);
            assignedIds.livestock.push(generatedLivestockId);

            const generatedMapId = IdGeneratorService.generateMappingId(l.deviceId, generatedLivestockId);

            // Establish the Aggregated view for the Frontend UI
            const masterRecord = new LivestockMaster({
                userId: createdUser._id,
                livestock_id: generatedLivestockId,
                name: l.name,
                species: l.type,
                breed: l.breed,
                age: parsedAge,
                weight: parsedWeight,

                farm_id: farmContext.seqId,
                farm_name: farmContext.name,
                farm_geofence: farmContext.geo,

                zone_id: zoneContext.seqId,
                zone_name: zoneContext.name,
                zone_geofence: zoneContext.geo,

                device_id: l.deviceId,
                device_type: l.deviceType,
                mapping_id: generatedMapId,

                vaccination_notes: l.vaccinationNotes,
                breeding_notes: l.breedingNotes,
                additional_notes: l.additionalNotes,

                // Setting starting vitals/status so the UI doesn't crash on initial load pre-simulator
                device_status: 'Active',
                health_status: 'Normal',
                temperature: 38.0,
                heart_rate: 60,
                battery: 100,
                signal_strength: -55
            });
            await masterRecord.save();

            // Link Device backwards
            await Device.updateOne(
                { deviceId: l.deviceId },
                { $set: { assignedTo: newLivestock._id } }
            );
        }

        // Push initial user bindings
        createdUser.farm = farmMap[Object.keys(farmMap)[0]]?.objectId || null;
        await createdUser.save();

        logger.info(`[SYSTEM INIT] Pipeline Complete. Success for ${email} with verified IDs.`);

        return successResponse(res, {
            _id: createdUser._id,
            name: createdUser.name,
            email: createdUser.email,
            token: generateToken(createdUser._id),
            init_success: true,
            devices_verified: hardwareIds.length,
            generated_ids: assignedIds
        });

    } catch (error) {
        logger.error(`[SYSTEM INIT] ERROR: ${error.message}`);
        return errorResponse(res, 'SYSTEM_INIT_FAILED', 'Initialization failed', 500, error.message);
    }
});

module.exports = router;
