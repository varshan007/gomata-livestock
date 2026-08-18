const pinoHttp = require('pino-http');
const logger = require('../utils/logger');
const crypto = require('crypto');

const requestLogger = pinoHttp({
    logger,
    genReqId: function (req) {
        return req.headers['x-request-id'] || crypto.randomUUID();
    },
    customProps: function (req, res) {
        return {
            requestId: req.id,
            tenantId: req.user ? req.user.tenantId : undefined,
            userId: req.user ? req.user.id : undefined,
            action: 'http_request'
        };
    },
    customLogLevel: function (req, res, err) {
        if (res.statusCode >= 400 && res.statusCode < 500) {
            return 'warn';
        } else if (res.statusCode >= 500 || err) {
            return 'error';
        }
        return 'info';
    },
    customSuccessMessage: function (req, res) {
        return `Request completed with status ${res.statusCode}`;
    },
    customErrorMessage: function (req, res, err) {
        return `Request errored: ${err.message}`;
    }
});

module.exports = requestLogger;
