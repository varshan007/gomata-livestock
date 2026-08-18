const express = require('express');
const { successResponse, errorResponse } = require('../utils/apiResponse');
const logger = require('../utils/logger');
const router = express.Router();
const jwt = require('jsonwebtoken');
const User = require('../models/User');
const Farm = require('../models/Farm');
const Zone = require('../models/Zone');
const Livestock = require('../models/Livestock');
const mongoose = require('mongoose');
const { protect } = require('../middleware/authMiddleware');
const OTP = require('../models/OTP');
const { sendEmailOTP, sendPhoneOTP } = require('../utils/otpService');
const crypto = require('crypto');

const generateToken = (id) => {
    return jwt.sign({ id, role: 'admin' }, process.env.JWT_SECRET, {
        expiresIn: '30d',
    });
};

// @desc    Register new user
// @route   POST /api/auth/register
// @access  Public
router.post('/register', async (req, res) => {
    const { name, email, password, phone, dob, farms, zones, livestock } = req.body;

    try {
        const userExists = await User.findOne({ email });

        if (userExists) {
            throw new Error('User already exists');
        }

        // 1. Create User
        const user = new User({ name, email, password, phone, dob });
        await user.save();

        // 2. Create Farms
        const farmIdMap = {};
        if (farms && farms.length > 0) {
            for (const f of farms) {
                const newFarm = new Farm({
                    userId: user._id,
                    name: f.name,
                    locationType: (f.locationType === 'Circle' || f.locationType === 'Circular Mapping') ? 'Circular Mapping' : 'Polygon Mapping',
                    geofence: f.geofence
                });
                await newFarm.save();
                farmIdMap[f.tempId] = newFarm._id;
            }
        }

        // 3. Create Zones
        const zoneIdMap = {};
        if (zones && zones.length > 0) {
            for (const z of zones) {
                const newZone = new Zone({
                    farmId: farmIdMap[z.farmTempId],
                    name: z.name,
                    locationType: (z.locationType === 'Circle' || z.locationType === 'Circular Mapping') ? 'Circular Mapping' : 'Polygon Mapping',
                    geofence: z.geofence
                });
                await newZone.save();
                zoneIdMap[z.tempId] = newZone._id;
            }
        }

        // 4. Create Livestock
        if (livestock && livestock.length > 0) {
            for (const l of livestock) {
                const newLivestock = new Livestock({
                    tagNumber: l.tagNumber || `UID-${crypto.randomBytes(4).toString('hex').toUpperCase()}`,
                    name: l.name,
                    breed: l.breed,
                    type: l.type,
                    age: l.age,
                    weight: l.weight,
                    deviceId: l.deviceId,
                    deviceType: l.deviceType,
                    farmId: farmIdMap[l.farmTempId],
                    zoneId: zoneIdMap[l.zoneTempId],
                    vaccinationNotes: l.vaccinationNotes,
                    breedingNotes: l.breedingNotes,
                    additionalNotes: l.additionalNotes
                });
                await newLivestock.save();
            }
        }

        logger.info({ action: 'auth_register', result: 'success', tenantId: user._id, userId: user._id, role: 'admin' }, `New farm owner registered: ${email}`);
        return successResponse(res, {
            _id: user._id,
            name: user.name,
            email: user.email,
            token: generateToken(user._id),
        });
    } catch (error) {
        logger.error({ action: 'auth_register', result: 'error', error: error.message }, "Registration Transaction Failed");
        return errorResponse(res, 'REGISTRATION_FAILED', 'Registration failed', 400, error.message);
    }
});

// @desc    Authenticate a user
// @route   POST /api/auth/login
// @access  Public
router.post('/login', async (req, res) => {
    const { email, password } = req.body;

    try {
        logger.info({ action: 'auth_login', email: email }, `Login Attempt for: ${email}`);
        logger.info(`Active DB: ${mongoose.connection.name}`); // Check which DB is connected

        const user = await User.findOne({ email });

        if (!user) {
            logger.warn({ action: 'auth_login', result: 'failure', reason: 'user_not_found', email: email }, "User not found in DB");
            return errorResponse(res, 'USER_NOT_FOUND', 'User not found in database', 401);
        }

        const isMatch = await user.matchPassword(password);
        logger.info(`User found: ${user.name}`);
        logger.info(`Stored Hash: ${user.password}`);
        logger.info(`Received Password: '${password}' (Length: ${password.length})`);
        logger.info(`Match Result: ${isMatch}`);

        if (isMatch) {
            logger.info({ action: 'auth_login', result: 'success', tenantId: user._id, userId: user._id, role: user.role }, `User logged in: ${email}`);
            return successResponse(res, {
                _id: user._id,
                name: user.name,
                email: user.email,
                farm: user.farm,
                token: generateToken(user._id),
            });
        } else {
            // Temporary detailed error for debugging
            return errorResponse(res, 'INVALID_CREDENTIALS', `Password mismatch. Received: '${password}' (Len: ${password.length})`, 401);
        }
    } catch (error) {
        return errorResponse(res, 'LOGIN_FAILED', 'Login failed', 500, error.message);
    }
});

// @desc    Get user profile
// @route   GET /api/auth/me
// @access  Private
router.get('/me', protect, async (req, res) => {
    if (req.user.type === 'staff') {
        // Staff user profile is already on req.user via the middleware
        return successResponse(res, {
            _id: req.user._id,
            name: req.user.name,
            email: req.user.email,
            role: req.user.staffRole,
            type: 'staff',
            assignedFarms: req.user.assignedFarms,
            assignedZones: req.user.assignedZones,
            adminUserId: req.user.adminUserId,
            createdAt: req.user.createdAt
        });
    }

    const user = await User.findById(req.user._id);

    if (user) {
        return successResponse(res, {
            _id: user._id,
            name: user.name,
            email: user.email,
            phone: user.phone,
            farm: user.farm,
            settings: user.settings,
            createdAt: user.createdAt
        });
    } else {
        return errorResponse(res, 'USER_NOT_FOUND', 'User not found', 404);
    }
});

// @desc    Update user profile
// @route   PUT /api/auth/profile
// @access  Private
router.put('/profile', protect, async (req, res) => {
    const user = await User.findById(req.user._id);

    if (user) {
        user.name = req.body.name || user.name;
        user.phone = req.body.phone || user.phone;

        if (req.body.farm) {
            user.farm = { ...user.farm, ...req.body.farm };
        }

        if (req.body.settings) {
            user.settings = { ...user.settings, ...req.body.settings };
        }

        if (req.body.password) {
            user.password = req.body.password;
        }

        const updatedUser = await user.save();

        return successResponse(res, {
            _id: updatedUser._id,
            name: updatedUser.name,
            email: updatedUser.email,
            phone: updatedUser.phone,
            farm: updatedUser.farm,
            settings: updatedUser.settings,
            token: generateToken(updatedUser._id) // Optional: issue new token
        });
    } else {
        return errorResponse(res, 'USER_NOT_FOUND', 'User not found', 404);
    }
});

// @desc    Send 6-digit OTP
// @route   POST /api/auth/send-otp
// @access  Public
router.post('/send-otp', async (req, res) => {
    const { identifier, type } = req.body;

    if (!identifier || !type) {
        return errorResponse(res, 'INVALID_INPUT', 'Identifier and type are required', 400);
    }

    try {
        const otpCode = crypto.randomInt(100000, 999999).toString();

        await OTP.deleteMany({ identifier });

        await OTP.create({
            identifier,
            code: otpCode,
            type
        });

        // 🔥 DEV MODE BYPASS (ALWAYS WORKS)
        if (process.env.NODE_ENV !== "production") {
            logger.info({ action: 'auth_send_otp', result: 'success', identifier: identifier, mode: 'dev', otp: otpCode }, `DEV MODE OTP for ${identifier}: ${otpCode}`);
            return successResponse(res, {
                message: "OTP generated (DEV MODE)",
                otp: otpCode
            });
        }

        // 🔥 PRODUCTION MODE
        let success = false;

        if (type === 'email') {
            success = await sendEmailOTP(identifier, otpCode);
        } else if (type === 'phone' || type === 'mobile') {
            success = await sendPhoneOTP(identifier, otpCode);
        }

        if (success) {
            return successResponse(res, { message: `OTP sent successfully via ${type}` });
        } else {
            return errorResponse(res, 'OTP_FAILED', `OTP dispatch failed via ${type}`, 500);
        }

    } catch (error) {
        logger.error("OTP ERROR:", error);
        return errorResponse(res, 'OTP_FAILED', 'OTP dispatch failed', 500, error.message);
    }
});

// @desc    Verify OTP
// @route   POST /api/auth/verify-otp
// @access  Public
router.post('/verify-otp', async (req, res) => {
    const { identifier, code } = req.body;

    if (!identifier || !code) {
        return errorResponse(res, 'INVALID_INPUT', 'Identifier and code are required', 400);
    }

    try {
        const validOtp = await OTP.findOne({ identifier, code });

        if (validOtp) {
            await OTP.deleteOne({ _id: validOtp._id }); // Single-use!
            return successResponse(res, { message: "OTP verified successfully" });
        } else {
            return errorResponse(res, 'INVALID_OTP', 'Invalid or expired OTP', 400);
        }
    } catch (error) {
        return errorResponse(res, 'OTP_VERIFICATION_FAILED', 'OTP verification failed', 500, error.message);
    }
});

module.exports = router;
