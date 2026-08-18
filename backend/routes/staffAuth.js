const express = require('express');
const { successResponse, errorResponse } = require('../utils/apiResponse');
const logger = require('../utils/logger');
const router = express.Router();
const jwt = require('jsonwebtoken');
const Staff = require('../models/Staff');

// @route   POST /api/staffAuth/login
// @desc    Auth Staff & get token
// @access  Public
router.post('/login', async (req, res) => {
    const { userId, password } = req.body;

    try {
        const staff = await Staff.findOne({ userId });

        if (!staff) {
            return errorResponse(res, 'INVALID_CREDENTIALS', 'Invalid User ID or password', 401);
        }

        if (staff.status === 'Inactive') {
            return errorResponse(res, 'ACCOUNT_INACTIVE', 'Account is deactivated', 403);
        }

        if (!staff.passwordSet) {
            return errorResponse(res, 'PASSWORD_NOT_SET', 'First time login requires password setup.', 403, { firstTime: true });
        }

        if (password && (await staff.matchPassword(password))) {
            const token = jwt.sign(
                { id: staff._id, role: 'staff' },
                process.env.JWT_SECRET,
                { expiresIn: '30d' }
            );

            return successResponse(res, {
                _id: staff._id,
                name: staff.name,
                email: staff.email,
                role: staff.role,
                type: 'staff',
                assignedFarms: staff.assignedFarms,
                assignedZones: staff.assignedZones,
                adminUserId: staff.adminUserId,
                token: token
            });
        } else {
            return errorResponse(res, 'INVALID_CREDENTIALS', 'Invalid User ID or password', 401);
        }
    } catch (err) {
        logger.error({ action: 'auth_login', result: 'error', role: 'staff', error: err.message }, 'Staff login error');
        return errorResponse(res, 'SERVER_ERROR', 'Server Error during staff login', 500, err.message);
    }
});

// @route   POST /api/staffAuth/setup-password
// @desc    Setup password for first-time staff login
// @access  Public
router.post('/setup-password', async (req, res) => {
    const { userId, phone, newPassword } = req.body;

    try {
        const staff = await Staff.findOne({ userId, phone });

        if (!staff) {
            return errorResponse(res, 'VERIFICATION_FAILED', 'Verification failed. Invalid User ID or Phone Number.', 400);
        }

        if (staff.passwordSet) {
            return errorResponse(res, 'PASSWORD_ALREADY_SET', 'Password has already been set for this account.', 400);
        }

        staff.password = newPassword;
        staff.passwordSet = true;
        await staff.save();

        const token = jwt.sign(
            { id: staff._id, role: 'staff' },
            process.env.JWT_SECRET,
            { expiresIn: '30d' }
        );

        logger.info({ action: 'auth_setup_password', result: 'success', tenantId: staff.adminUserId, userId: staff._id, role: 'staff', staffRole: staff.role }, `Staff user password set: ${staff.phone}`);

        return successResponse(res, {
            _id: staff._id,
            name: staff.name,
            email: staff.email,
            userId: staff.userId,
            role: staff.role,
            type: 'staff',
            assignedFarms: staff.assignedFarms,
            assignedZones: staff.assignedZones,
            adminUserId: staff.adminUserId,
            token: token
        });
    } catch (err) {
        logger.error({ action: 'auth_setup_password', result: 'error', error: err.message }, 'Staff password setup error');
        return errorResponse(res, 'SERVER_ERROR', 'Server Error during password setup', 500, err.message);
    }
});

module.exports = router;
