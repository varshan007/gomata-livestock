const pino = require('pino');

const isDevelopment = process.env.NODE_ENV !== 'production' && process.env.NODE_ENV !== 'staging';

const logger = pino({
    level: process.env.LOG_LEVEL || 'info',
    base: {
        service: 'livestock-backend'
    },
    transport: isDevelopment ? {
        target: 'pino-pretty',
        options: {
            colorize: true,
            translateTime: 'SYS:standard',
            ignore: 'pid,hostname',
        }
    } : undefined,
    formatters: {
        level: (label) => {
            return { level: label };
        },
    },
});

module.exports = logger;
