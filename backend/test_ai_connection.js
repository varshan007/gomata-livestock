const { GoogleGenerativeAI } = require("@google/generative-ai");
require('dotenv').config();

const GOOGLE_API_KEY = process.env.GOOGLE_API_KEY;

async function testConnection() {
    console.log("Testing Gemini API Connection...");
    console.log("API Key present:", !!GOOGLE_API_KEY);

    if (!GOOGLE_API_KEY) {
        console.error("❌ ERROR: GOOGLE_API_KEY is missing in .env");
        return;
    }

    try {
        const genAI = new GoogleGenerativeAI(GOOGLE_API_KEY);
        // Trying 'gemini-pro' as fallback to test key validity generally
        const model = genAI.getGenerativeModel({ model: "gemini-pro" });

        const prompt = "Hello, are you online?";
        console.log(`Sending prompt: "${prompt}" to model: gemini-pro`);

        const result = await model.generateContent(prompt);
        const response = await result.response;
        const text = response.text();

        console.log("✅ API Success!");
        console.log("Response:", text);

    } catch (error) {
        console.error("❌ API Failed!");
        console.error("Error Message:", error.message);
        console.error("Full Error:", error);
    }
}

testConnection();
