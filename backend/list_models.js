const axios = require('axios');
require('dotenv').config();

const XAI_API_KEY = process.env.XAI_API_KEY;

async function listModels() {
    try {
        const response = await axios.get('https://api.x.ai/v1/models', {
            headers: {
                'Authorization': `Bearer ${XAI_API_KEY}`,
                'Content-Type': 'application/json'
            }
        });

        console.log("Available Models:", JSON.stringify(response.data, null, 2));
    } catch (error) {
        console.error("Error listing models:", error.response ? error.response.data : error.message);
    }
}

listModels();
