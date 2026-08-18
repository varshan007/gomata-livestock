const successResponse = (res, data, meta = {}) => {
    return res.status(200).json({
        success: true,
        data,
        meta: {
            timestamp: new Date().toISOString(),
            version: '1.0',
            ...meta
        }
    });
};

const errorResponse = (res, code, message, status = 500, details = null) => {
    return res.status(status).json({
        success: false,
        error: {
            code,
            message,
            details
        }
    });
};

module.exports = {
    successResponse,
    errorResponse
};
