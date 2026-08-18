const logger = require('../utils/logger');
const express = require('express');
const { successResponse, errorResponse } = require('../utils/apiResponse');
const router = express.Router();
const { chatWithVetAssistant } = require('../services/aiService');
const { protect } = require('../middleware/authMiddleware');
const User = require('../models/User');
const LivestockMaster = require('../models/LivestockMaster');
const Farm = require('../models/Farm');
const Zone = require('../models/Zone');

// @desc    Chat with AI Assistant (Voice Context)
// @route   POST /api/ai/chat
// @access  Private
router.post('/chat', protect, async (req, res) => {
    const { query, history } = req.body;

    try {
        // Fetch full user context including farm and livestock
        const user = await User.findById(req.user._id);

        if (!user) {
            return errorResponse(res, 'USER_NOT_FOUND', 'User not found', 404);
        }

        // Fetch all livestock using the correct model
        const allLivestock = await LivestockMaster.find({ userId: req.user.tenantId });

        // Map LivestockMaster to context
        const animalsContext = allLivestock.map(animal => ({
            name: animal.name || animal.livestock_id,
            type: animal.breed || animal.species || 'Livestock',
            status: animal.health_status || 'Unknown',
            temp: animal.temperature || 'N/A',
            location: animal.zone_name || 'Unknown',
            battery: animal.battery || 'N/A'
        }));

        // Fetch Farms and Zones
        const allFarms = await Farm.find({ userId: req.user.tenantId });
        const allZones = await Zone.find({ farmId: { $in: allFarms.map(f => f._id) } });

        const farmContext = {
            farmName: user.farm?.name || 'Default Farm',
            livestockType: user.farm?.livestockType || 'Livestock',
            count: allLivestock.length,
            location: user.farm?.location || 'Unknown',
            animals: animalsContext,
            farms: allFarms.map(f => f.name),
            zones: allZones.map(z => z.name)
        };

        let aiResponseText = await chatWithVetAssistant(farmContext, query, history);
        logger.info("🤖 RAW AI RESPONSE:", aiResponseText); // DEBUG LOG

        let refresh = false;

        // Clean up code blocks if present
        let cleanText = aiResponseText.replace(/```json/g, '').replace(/```/g, '').trim();
        logger.info("🧹 CLEANED TEXT:", cleanText); // DEBUG LOG

        // Check availability of Action (JSON)
        if (cleanText.startsWith('{') && cleanText.includes('"action":')) {
            try {
                const actionObj = JSON.parse(cleanText);
                logger.info("🧩 PARSED ACTION:", actionObj); // DEBUG LOG

                if (actionObj.action === 'ADD_ANIMAL' && actionObj.data) {
                    logger.info("⚡ Executing AI Action: ADD_ANIMAL", actionObj.data);

                    // Create new livestock entry
                    const newAnimal = new Livestock({
                        name: actionObj.data.name || 'Unknown',
                        tagNumber: actionObj.data.tagNumber || `TAG-${Math.floor(Math.random() * 10000)}`,
                        breed: actionObj.data.breed || 'Mixed',
                        age: actionObj.data.age || 1,
                        weight: actionObj.data.weight || 100,
                        gender: actionObj.data.gender || 'Female',
                        deviceId: `AI-GEN-${Date.now()}`,
                        photoUrl: null,
                        userId: req.user.tenantId
                    });

                    await newAnimal.save();
                    logger.info("✅ ANIMAL SAVED TO DB"); // DEBUG LOG
                    aiResponseText = actionObj.response; // Use the confirmation message
                    refresh = true; // Signal frontend to refresh
                }
            } catch (e) {
                logger.error("❌ Failed to parse AI Action JSON:", e);
                // Fallback: just send the raw text if parsing fails
            }
        } else {
            logger.info("ℹ️ No JSON detected, treating as plain text.");
        }

        return successResponse(res, { response: aiResponseText, refresh });

    } catch (error) {
        logger.error('AI Chat Error:', error);
        return errorResponse(res, 'AI_CHAT_ERROR', 'Failed to process voice query', 500, error.message);
    }
});

module.exports = router;
