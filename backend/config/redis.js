const Redis = require('ioredis');

// Ensure REDIS_URL is available
if (!process.env.REDIS_URL) {
    console.error('REDIS_URL environment variable is not defined!');
    process.exit(1);
}

// Create the Redis connection
const redisOptions = {
    maxRetriesPerRequest: null,
    enableReadyCheck: false,
    retryStrategy: (times) => {
        // Reconnect after
        return Math.min(times * 50, 2000);
    }
};

const redisConnection = new Redis(process.env.REDIS_URL, redisOptions);

redisConnection.on('connect', () => {
    console.log('Connected to Redis server successfully.');
});

redisConnection.on('error', (err) => {
    console.error('Redis connection error:', err);
});

module.exports = redisConnection;
