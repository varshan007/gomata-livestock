const logger = require('../../../utils/logger');
/**
 * OnboardingAgent
 * Handles one-time farm configuration, livestock registration, and hardware pairings.
 */
class OnboardingAgent {
    constructor(bus, models) {
        this.bus = bus;
        this.models = models; // Expects { Livestock, User, Geofence }
    }

    start() {
        // Listen for requests to register new livestock
        this.bus.on('onboarding:register_livestock', async (payload) => {
            const { animalData, farmId, generatedBy } = payload.data;
            try {
                // Ensure deviceId is unique and registered
                const newAnimal = await this.models.Livestock.create({
                    ...animalData,
                    status: 'Active'
                });

                logger.info(`[OnboardingAgent] Successfully registered ${newAnimal.name} (${newAnimal.tagNumber})`);

                // Emit success so Sync Agent updates the UI
                this.bus.emit('onboarding:livestock_registered', {
                    animal: newAnimal,
                    farmId
                }, payload.traceId);

            } catch (error) {
                logger.error(`[OnboardingAgent] Failed to register livestock:`, error);
                this.bus.emit('onboarding:error', { error: error.message }, payload.traceId);
            }
        });

        logger.info('[OnboardingAgent] Started.');
    }
}

module.exports = OnboardingAgent;
