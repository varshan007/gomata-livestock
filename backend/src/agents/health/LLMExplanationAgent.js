const logger = require('../../../utils/logger');
const { GoogleGenerativeAI } = require('@google/generative-ai');
const Alert = require('../../../models/Alert');
const Redis = require('ioredis');

const LOG_SERVICE = 'llm_explanation_agent';
const EXPLANATION_CACHE_TTL = 600; // 10 minutes
const MAX_RETRIES = 2;

const sleep = (ms) => new Promise(resolve => setTimeout(resolve, ms));

/**
 * LLMExplanationAgent
 * 
 * Subscribes to `alert:saved` events from HealthAgent.
 * Generates human-readable veterinary explanations using Gemini LLM.
 * Caches explanations in Redis, stores in MongoDB Alert document.
 * Publishes `alert:explained` event for real-time frontend updates.
 */
class LLMExplanationAgent {
    constructor(options = {}) {
        this.redis = options.redis || new Redis(process.env.REDIS_URL || 'redis://127.0.0.1:6379');
        this.eventBus = options.eventBus || null;
        this.log = options.logger || logger;
        this.apiKey = process.env.GOOGLE_API_KEY;
        this._processing = new Set(); // Prevent duplicate processing
    }

    /**
     * Process a single alert — generate LLM explanation
     * @param {Object} alertData - Alert data from event bus or direct call
     */
    async processAlert(alertData) {
        const alertId = alertData.alertId;
        const animalId = alertData.livestockId;

        // Prevent duplicate processing
        if (this._processing.has(alertId)) {
            this.log.info({ service: LOG_SERVICE, action: 'SKIP_DUPLICATE', alertId });
            return null;
        }
        this._processing.add(alertId);

        try {
            // Check Redis cache first
            const cached = await this.redis.get(`explanation:${alertId}`);
            if (cached) {
                this.log.info({ service: LOG_SERVICE, action: 'CACHE_HIT', alertId });
                this._processing.delete(alertId);
                return JSON.parse(cached);
            }

            // Fetch alert from DB for full context
            const alert = await Alert.findById(alertId).lean();
            if (!alert) {
                this.log.warn({ service: LOG_SERVICE, action: 'ALERT_NOT_FOUND', alertId });
                this._processing.delete(alertId);
                return null;
            }

            // Fetch cached features from Redis for telemetry context
            let telemetry = {};
            try {
                const featuresRaw = await this.redis.get(`features:${animalId}`);
                if (featuresRaw) {
                    telemetry = JSON.parse(featuresRaw);
                }
            } catch (e) {
                this.log.warn({ service: LOG_SERVICE, action: 'FEATURES_FETCH_FAILED', animalId });
            }

            // Generate LLM explanation
            const explanation = await this._generateExplanation(alert, telemetry);

            if (!explanation) {
                this._processing.delete(alertId);
                return null;
            }

            // Store explanation in Alert document
            await Alert.findByIdAndUpdate(alertId, { explanation });

            // Cache in Redis
            const cachePayload = {
                alertId,
                animalId,
                animalName: alert.animalName,
                severity: alert.severity,
                diseaseProbability: alert.diseaseProbability,
                explanation
            };
            await this.redis.setex(
                `explanation:${alertId}`,
                EXPLANATION_CACHE_TTL,
                JSON.stringify(cachePayload)
            );

            // Publish explained event
            if (this.eventBus) {
                this.eventBus.emit('alert:explained', {
                    alertId,
                    livestockId: animalId,
                    animalName: alert.animalName,
                    severity: alert.severity,
                    explanation
                });
            }

            this.log.info({
                service: LOG_SERVICE,
                action: 'EXPLANATION_GENERATED',
                alertId,
                animalName: alert.animalName,
                explanationLength: explanation.length
            }, `[LLMAgent] Explanation generated for ${alert.animalName}`);

            this._processing.delete(alertId);
            return cachePayload;

        } catch (err) {
            this._processing.delete(alertId);
            this.log.error({
                service: LOG_SERVICE,
                action: 'EXPLANATION_ERROR',
                alertId,
                error: err.message
            }, `[LLMAgent] Failed to generate explanation: ${err.message}`);
            return null;
        }
    }

    /**
     * Generate veterinary explanation using Gemini LLM
     */
    async _generateExplanation(alert, telemetry) {
        if (!this.apiKey) {
            this.log.warn({ service: LOG_SERVICE, action: 'NO_API_KEY' });
            return this._generateFallbackExplanation(alert, telemetry);
        }

        const genAI = new GoogleGenerativeAI(this.apiKey);
        let model;
        try {
            model = genAI.getGenerativeModel({ model: 'gemini-2.5-flash' });
        } catch (e) {
            model = genAI.getGenerativeModel({ model: 'gemini-2.0-flash-exp' });
        }

        const temp = telemetry.temperature_mean || telemetry.temperature || 'N/A';
        const hr = telemetry.heart_rate_mean || telemetry.heart_rate || 'N/A';
        const activity = telemetry.activity_index_mean || telemetry.activity_index || 'N/A';
        const probPercent = alert.diseaseProbability
            ? (alert.diseaseProbability * 100).toFixed(1)
            : 'N/A';

        const prompt = `You are GoMata AI, a veterinary health intelligence system for Indian dairy cattle.

ANIMAL PROFILE:
- Name: ${alert.animalName || 'Unknown'}
- Breed: ${alert.breed || 'Unknown'}
- Farm: ${alert.farmName || 'Unknown'}
- Zone: ${alert.zoneName || 'Unknown'}
- Device: ${alert.deviceId || 'N/A'}

CURRENT VITALS:
- Body Temperature: ${temp}°C (Normal: 38.0–39.0°C)
- Heart Rate: ${hr} bpm (Normal: 60–80 bpm)
- Activity Index: ${activity} (Normal: 0.4–0.8)
- Disease Probability: ${probPercent}%
- Alert Severity: ${alert.severity}

ML MODEL PREDICTION:
The XGBoost disease classifier detected a ${probPercent}% disease probability based on 23 engineered features from the last 24 hours of sensor data.

TASK:
Generate a concise veterinary explanation (150-200 words) with these sections:

**Current Health State:** Describe what's abnormal (e.g., "elevated temperature at 40.2°C indicates fever")
**Likely Condition:** Suggest 1-2 probable conditions based on the symptoms (e.g., early-stage mastitis, heat stress, respiratory infection)
**Recommended Actions:**
- 2-3 specific, actionable steps for the farm operator
- Include urgency level (immediate, within 24h, monitor)
**Educational Note:** One brief sentence about the condition for farmer education

Keep the tone professional but accessible for Indian dairy farmers. Use simple medical terms.`;

        let retryCount = 0;
        while (retryCount <= MAX_RETRIES) {
            try {
                const result = await model.generateContent(prompt);
                const response = await result.response;
                return response.text();
            } catch (error) {
                this.log.error({
                    service: LOG_SERVICE,
                    action: 'LLM_API_ERROR',
                    attempt: retryCount + 1,
                    error: error.message
                });
                if (error.message.includes('429') || error.message.includes('Quota')) {
                    if (retryCount < MAX_RETRIES) {
                        await sleep(Math.pow(2, retryCount) * 1000);
                        retryCount++;
                        continue;
                    }
                }
                // Fall back to template-based explanation
                return this._generateFallbackExplanation(alert, telemetry);
            }
        }
    }

    /**
     * Fallback: Template-based explanation when LLM is unavailable
     */
    _generateFallbackExplanation(alert, telemetry) {
        const temp = telemetry.temperature_mean || telemetry.temperature || 'N/A';
        const hr = telemetry.heart_rate_mean || telemetry.heart_rate || 'N/A';
        const prob = alert.diseaseProbability
            ? (alert.diseaseProbability * 100).toFixed(1)
            : 'N/A';
        const name = alert.animalName || 'This animal';
        const breed = alert.breed || 'Unknown breed';

        let condition = 'general health concern';
        let urgency = 'Monitor closely';
        let actions = [];

        if (parseFloat(temp) > 39.5) {
            condition = 'fever — possible infection, heat stress, or early-stage mastitis';
            urgency = 'Immediate veterinary inspection recommended';
            actions = [
                'Isolate the animal from the herd to prevent potential spread',
                'Provide fresh water and shade — check for signs of dehydration',
                'Contact veterinarian for physical examination within 4 hours'
            ];
        } else if (parseFloat(hr) > 90) {
            condition = 'tachycardia — possible stress, pain, or cardiovascular issue';
            urgency = 'Schedule veterinary check within 24 hours';
            actions = [
                'Move animal to a calm, shaded area',
                'Monitor heart rate every 30 minutes',
                'Check for signs of injury or distress'
            ];
        } else {
            condition = 'abnormal vital sign patterns detected by ML model';
            urgency = 'Monitor closely over next 12 hours';
            actions = [
                'Increase monitoring frequency for this animal',
                'Record any changes in feeding behavior or milk production',
                'Schedule routine veterinary check if patterns persist'
            ];
        }

        return `**Current Health State:** ${name} (${breed}) is showing ${condition}. ` +
            `Body temperature: ${temp}°C, Heart rate: ${hr} bpm. ` +
            `The ML model assigned a ${prob}% disease probability.\n\n` +
            `**Likely Condition:** ${condition}.\n\n` +
            `**Recommended Actions:**\n` +
            actions.map((a, i) => `${i + 1}. ${a}`).join('\n') + '\n\n' +
            `**Urgency:** ${urgency}\n\n` +
            `**Educational Note:** Regular monitoring of body temperature and heart rate helps detect health issues early, ` +
            `reducing treatment costs and improving animal welfare.`;
    }

    /**
     * Subscribe to alert events from the event bus
     */
    subscribeToAlerts(eventBus) {
        this.eventBus = eventBus;
        eventBus.on('alert:saved', async (envelope) => {
            const data = envelope.data || envelope;
            this.log.info({
                service: LOG_SERVICE,
                action: 'ALERT_RECEIVED',
                alertId: data.alertId
            }, `[LLMAgent] Received alert event for ${data.alertId}`);

            // Process asynchronously so we don't block the event bus
            setImmediate(() => this.processAlert(data));
        });

        this.log.info({
            service: LOG_SERVICE,
            action: 'SUBSCRIBED'
        }, '[LLMAgent] Subscribed to alert:saved events');
    }
}

module.exports = LLMExplanationAgent;
