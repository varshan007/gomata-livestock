const { GoogleGenerativeAI } = require("@google/generative-ai");
const axios = require('axios');
require('dotenv').config();

const GOOGLE_API_KEY = process.env.GOOGLE_API_KEY;

async function listModels() {
    try {
        console.log("Listing models via REST API...");
        const response = await axios.get(
            `https://generativelanguage.googleapis.com/v1beta/models?key=${GOOGLE_API_KEY}`
        );

        const models = response.data.models;
        console.log(`Found ${models.length} models:`);

        models.forEach(model => {
            if (model.supportedGenerationMethods && model.supportedGenerationMethods.includes("generateContent")) {
                console.log(`- ${model.name} (${model.displayName})`);
            }
        });

    } catch (error) {
        console.error("Error listing models:", error.response ? error.response.data : error.message);
    }
}

listModels();
