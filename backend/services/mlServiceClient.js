const axios = require('axios');
const logger = require('../utils/logger');

class MLServiceClient {
    getClient() {
        if (!this.client) {
            const baseUrl = process.env.ML_SERVICE_URL;
            if (!baseUrl) {
                logger.warn({ action: 'ml_config_check', service: 'ml_client' }, 'ML_SERVICE_URL is not defined in environment variables. ML features will be disabled.');
            }
            this.client = axios.create({
                baseURL: baseUrl,
                timeout: 5000,
                headers: { 'Content-Type': 'application/json' }
            });
        }
        return this.client;
    }

    async getHealth() {
        if (!process.env.ML_SERVICE_URL) return { status: 'disabled' };
        try {
            const response = await this.getClient().get('/health');
            return response.data;
        } catch (error) {
            logger.error({ action: 'ml_health_check', result: 'error', service: 'ml_client', error: error.message }, 'ML Service Health Check Failed');
            throw new Error(`ML Service Unavailable: ${error.message}`);
        }
    }

    async predict(data) {
        if (!process.env.ML_SERVICE_URL) throw new Error('ML Service is not configured.');
        try {
            // Map structured queue payload to Python ML model schema
            const activityNum = data.telemetry && data.telemetry.activity === 'still' ? 0.0 : 1.0;
            const targetId = data.animalId || data.livestock_id; // backward-compatibility for old queued jobs
            const temp = data.telemetry ? data.telemetry.temperature : 38.5;
            const hr = data.telemetry ? data.telemetry.heartRate : 65;

            const mlPayload = {
                livestock_id: targetId ? targetId.toString() : "unknown",
                temperature: temp,
                heart_rate: hr,
                activity_level: activityNum
            };
            const response = await this.getClient().post('/predict', mlPayload);
            return response.data;
        } catch (error) {
            const animalId = data.animalId || data.livestock_id;
            const validationDetails = error.response && error.response.data ? JSON.stringify(error.response.data) : error.message;
            logger.error({ action: 'ml_predict', result: 'error', service: 'ml_client', animalId: animalId, error: validationDetails }, 'ML Prediction Failed');
            throw error;
        }
    }
}

module.exports = new MLServiceClient();
