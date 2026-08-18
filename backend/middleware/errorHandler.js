const logger = require('../utils/logger');
const { errorResponse } = require('../utils/apiResponse');

const errorHandler = (err, req, res, next) => {
    logger.error(err.stack);

    // Default to 500 server error
    const statusCode = res.statusCode === 200 ? 500 : res.statusCode;

    return errorResponse(
        res,
        'INTERNAL_SERVER_ERROR',
        err.message || 'An unexpected error occurred',
        statusCode,
        process.env.NODE_ENV === 'development' ? err.stack : null
    );
};

// Catch 404 and forward to error handler
const notFoundHandler = (req, res, next) => {
    const error = new Error(`Not Found - ${req.originalUrl}`);
    res.status(404);
    next(error);
};

module.exports = {
    errorHandler,
    notFoundHandler
};
