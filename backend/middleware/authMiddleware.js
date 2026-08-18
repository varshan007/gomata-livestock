const logger = require('../utils/logger');
const jwt = require('jsonwebtoken');
const User = require('../models/User');

const protect = async (req, res, next) => {
    let token;

    if (
        req.headers.authorization &&
        req.headers.authorization.startsWith('Bearer')
    ) {
        try {
            token = req.headers.authorization.split(' ')[1];

            const decoded = jwt.verify(token, process.env.JWT_SECRET);

            if (decoded.role === 'staff') {
                const Staff = require('../models/Staff');
                const staffUser = await Staff.findById(decoded.id).select('-password');
                if (!staffUser) {
                    return res.status(401).json({ message: 'Not authorized, staff not found' });
                }

                req.user = {
                    ...staffUser._doc,
                    id: staffUser._id.toString(),
                    tenantId: staffUser.adminUserId,
                    role: 'staff',
                    staffRole: staffUser.role, // preserve database Staff functionality role
                    type: 'staff' // for backward compatibility
                };
            } else {
                req.user = await User.findById(decoded.id).select('-password');
                if (!req.user) {
                    return res.status(401).json({ message: 'Not authorized, user not found' });
                }

                req.user = {
                    ...req.user._doc,
                    id: req.user._id.toString(),
                    tenantId: req.user._id,
                    role: 'admin',
                    type: 'admin' // for backward compatibility
                };
            }

            next();
        } catch (error) {
            logger.error(error);
            res.status(401).json({ message: 'Not authorized, token failed' });
        }
    }

    if (!token) {
        res.status(401).json({ message: 'Not authorized, no token' });
    }
};

module.exports = { protect };
