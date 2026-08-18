const logger = require('../utils/logger');
const express = require('express');
const { successResponse, errorResponse } = require('../utils/apiResponse');
const router = express.Router();
const Staff = require('../models/Staff');
const { protect } = require('../middleware/authMiddleware');

// @route   GET /api/staff
// @desc    Get all staff for the admin
// @access  Private (Admin only)
router.get('/', protect, async (req, res) => {
    if (req.user.type === 'staff') {
        return errorResponse(res, 'UNAUTHORIZED_ACCESS', 'Access denied: Staff cannot manage staff.', 403);
    }

    try {
        const staffList = await Staff.find({ adminUserId: req.user._id }).select('-password');
        return successResponse(res, staffList);
    } catch (err) {
        logger.error('Error fetching staff:', err.message);
        return errorResponse(res, 'FETCH_STAFF_FAILED', 'Server Error fetching staff', 500, err.message);
    }
});

// @route   POST /api/staff
// @desc    Create a new staff user
// @access  Private (Admin only)
router.post('/', protect, async (req, res) => {
    if (req.user.type === 'staff') {
        return errorResponse(res, 'UNAUTHORIZED_ACCESS', 'Access denied: Staff cannot create staff.', 403);
    }

    const { name, email, phone, position, role, assignedFarms, assignedZones, assignedDevices, primaryResponsibility, assignedShift, alertPreferences, status } = req.body;

    if (!name || !phone || !role) {
        return errorResponse(res, 'INVALID_INPUT', 'Name, Phone and Role are required.', 400);
    }

    try {
        // Auto-generate User ID e.g. fullname0001@gomata.ai.com
        const baseName = name.replace(/\s+/g, '').toLowerCase();
        const count = await Staff.countDocuments({ userId: { $regex: new RegExp(`^${baseName}[0-9]*@gomata\\.ai\\.com$`, 'i') } });
        const formattedCount = (count + 1).toString().padStart(4, '0');
        const userId = `${baseName}${formattedCount}@gomata.ai.com`;

        let existingStaff = email ? await Staff.findOne({ email }) : null;
        if (existingStaff) {
            return errorResponse(res, 'EMAIL_EXISTS', 'A staff member with this email already exists', 400);
        }

        const staff = new Staff({
            name,
            email: email || undefined,
            phone,
            userId,
            position: position || 'Worker',
            role: role || 'Viewer',
            assignedFarms: assignedFarms || [],
            assignedZones: assignedZones || [],
            assignedDevices: assignedDevices || [],
            primaryResponsibility: primaryResponsibility || 'Farm Operations',
            assignedShift: assignedShift || 'Full Day',
            alertPreferences: alertPreferences || [],
            status: status || 'Active',
            passwordSet: false,
            adminUserId: req.user._id
        });

        await staff.save();

        // Return without password
        const staffResponse = await Staff.findById(staff._id).select('-password');
        return successResponse(res, staffResponse);
    } catch (err) {
        logger.error('Error creating staff:', err.message);
        return errorResponse(res, 'CREATE_STAFF_FAILED', 'Server Error creating staff', 500, err.message);
    }
});

// @route   DELETE /api/staff/:id
// @desc    Delete a staff user
// @access  Private (Admin only)
router.delete('/:id', protect, async (req, res) => {
    if (req.user.type === 'staff') {
        return errorResponse(res, 'UNAUTHORIZED_ACCESS', 'Access denied: Staff cannot delete staff.', 403);
    }

    try {
        const staff = await Staff.findOne({ _id: req.params.id, adminUserId: req.user._id });
        if (!staff) {
            return errorResponse(res, 'STAFF_NOT_FOUND', 'Staff user not found', 404);
        }

        await staff.remove();
        return successResponse(res, { message: 'Staff user removed' });
    } catch (err) {
        logger.error('Error removing staff:', err.message);
        return errorResponse(res, 'DELETE_STAFF_FAILED', 'Server Error removing staff', 500, err.message);
    }
});

// @route   PUT /api/staff/:id
// @desc    Update a staff user
// @access  Private (Admin only)
router.put('/:id', protect, async (req, res) => {
    if (req.user.type === 'staff') {
        return errorResponse(res, 'UNAUTHORIZED_ACCESS', 'Access denied: Staff cannot update staff.', 403);
    }

    const { name, email, phone, position, role, assignedFarms, assignedZones, assignedDevices, primaryResponsibility, assignedShift, alertPreferences, status } = req.body;

    try {
        const staff = await Staff.findOne({ _id: req.params.id, adminUserId: req.user._id });
        if (!staff) {
            return errorResponse(res, 'STAFF_NOT_FOUND', 'Staff user not found', 404);
        }

        staff.name = name || staff.name;
        staff.email = email || staff.email;
        if (phone) staff.phone = phone;
        if (position) staff.position = position;
        staff.role = role || staff.role;
        staff.assignedFarms = assignedFarms || staff.assignedFarms;
        staff.assignedZones = assignedZones || staff.assignedZones;
        if (assignedDevices) staff.assignedDevices = assignedDevices;
        if (primaryResponsibility) staff.primaryResponsibility = primaryResponsibility;
        if (assignedShift) staff.assignedShift = assignedShift;
        if (alertPreferences) staff.alertPreferences = alertPreferences;
        if (status) staff.status = status;

        await staff.save();
        const updatedStaff = await Staff.findById(staff._id).select('-password');
        return successResponse(res, updatedStaff);
    } catch (err) {
        logger.error('Error updating staff:', err.message);
        return errorResponse(res, 'UPDATE_STAFF_FAILED', 'Server Error updating staff', 500, err.message);
    }
});

module.exports = router;
