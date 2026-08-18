const logger = require('../../../utils/logger');
const jwt = require('jsonwebtoken');

/**
 * AuthAgent
 * Handles security, role scopes, and JWT generation for the multi-agent system.
 */
class AuthAgent {
    constructor(bus, jwtSecret = process.env.JWT_SECRET || 'gomata_super_secret_key') {
        this.bus = bus;
        this.jwtSecret = jwtSecret;
    }

    start() {
        // In a full implementation, this agent would listen for 'auth:login' events,
        // verify credentials against the User model, and emit 'auth:success' with the token.
        this.bus.on('auth:login_request', async (payload) => {
            logger.info(`[AuthAgent] Received login request for ${payload.email}`);
            // Logic to verify user...
        });
        logger.info('[AuthAgent] Started.');
    }

    /**
     * Generates a scoped JWT for a user.
     * @param {object} user - The user document
     * @param {string} role - 'Farmer', 'Vet', 'Staff', or 'Admin'
     */
    generateToken(user, role = 'Staff') {
        return jwt.sign(
            { id: user._id, role: role, farmId: user.farmId },
            this.jwtSecret,
            { expiresIn: '7d' }
        );
    }

    /**
     * Middleware to verify agent-level or API-level token scopes.
     */
    verifyScope(requiredRole) {
        return (req, res, next) => {
            const token = req.headers.authorization?.split(' ')[1];
            if (!token) return res.status(401).json({ message: 'No token provided' });

            try {
                const decoded = jwt.verify(token, this.jwtSecret);
                req.user = decoded;

                // Simple hierarchical role check (Admin > Farmer > Vet > Staff)
                const roleHierarchy = { 'Staff': 1, 'Vet': 2, 'Farmer': 3, 'Admin': 4 };
                if (roleHierarchy[decoded.role] < roleHierarchy[requiredRole]) {
                    return res.status(403).json({ message: 'Insufficient agent scope' });
                }

                next();
            } catch (err) {
                return res.status(401).json({ message: 'Invalid token' });
            }
        }
    }
}

module.exports = AuthAgent;
