const axios = require('axios');
require('dotenv').config();

const API_KEY = process.env.GOOGLE_API_KEY;

async function testRaw() {
    console.log("Testing raw API access...");
    try {
        const url = `https://generativelanguage.googleapis.com/v1beta/models?key=${API_KEY}`;
        const response = await axios.get(url);
        console.log("✅ Success! Models found:", response.data.models.length);
        console.log("First 3 models:", response.data.models.slice(0, 3).map(m => m.name));
    } catch (error) {
        console.error("❌ Error:", error.response ? error.response.data : error.message);
    }
}

testRaw();
