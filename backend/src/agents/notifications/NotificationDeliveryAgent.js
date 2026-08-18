const logger = require('../../../utils/logger');
const twilio = require('twilio');
const twilioClient = (process.env.TWILIO_ACCOUNT_SID && process.env.TWILIO_AUTH_TOKEN)
    ? twilio(process.env.TWILIO_ACCOUNT_SID, process.env.TWILIO_AUTH_TOKEN)
    : null;
const User = require('../../../models/User');

class NotificationDeliveryAgent {
    constructor(bus) {
        this.bus = bus;
    }

    start() {
        logger.info('[NotificationDeliveryAgent] Started. Awaiting verified alerts to dispatch via SMS...');

        // Only listen for alerts that survived the Deduplication phase in the AlertAgent
        this.bus.on('alert:saved', async (payload) => {
            const alertData = payload.data;
            const traceId = payload.traceId;
            const { alertId, livestockId, severity, message } = alertData;

            try {
                // Find users who own this livestock or Staff
                // In a real multi-tenant app, we fetch the specific Farm owners.
                // For MVP, we alert Admins or Vets.
                const recipients = await User.find({ role: { $in: ['Admin', 'Vet'] } });

                if (recipients.length === 0) {
                    logger.info(`[NotificationDeliveryAgent] No Admins/Vets found to notify for alert ${alertId}.`);
                    return;
                }

                const shortMessage = message.length > 50 ? message.substring(0, 47) + '...' : message;
                const smsBody = `GoMata Alert (${severity}): Livestock ${livestockId}\n${shortMessage}\nLog in to Dashboard.`;

                // Dispatch SMS
                recipients.forEach(async (user) => {
                    if (user.phone) {
                        logger.info(`[NotificationDeliveryAgent] 📱 Sending SMS to ${user.name} (${user.phone}) -> ${severity}`);

                        try {
                            if (!twilioClient) {
                                logger.warn(`[NotificationDeliveryAgent] ⚠️ Twilio not configured. Skipping SMS to ${user.phone}`);
                                return;
                            }
                            const msg = await twilioClient.messages.create({
                                body: smsBody,
                                from: process.env.TWILIO_PHONE_NUMBER,
                                to: user.phone
                            });
                            logger.info(`[NotificationDeliveryAgent] ✅ SMS sent successfully. Twilio SID: ${msg.sid}`);
                        } catch (smsError) {
                            logger.error(`[NotificationDeliveryAgent] ❌ Twilio SMS Error for ${user.phone}:`, smsError.message);
                        }
                    }
                });

            } catch (error) {
                logger.error(`[NotificationDeliveryAgent] DB Lookup Failed:`, error);
            }
        });
    }
}

module.exports = NotificationDeliveryAgent;
