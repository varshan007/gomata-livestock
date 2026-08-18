const { GoogleGenerativeAI } = require("@google/generative-ai");
require('dotenv').config();

const GOOGLE_API_KEY = process.env.GOOGLE_API_KEY;

const modelsToTest = [
    "gemini-1.5-flash",
    "gemini-1.5-pro",
    "gemini-2.0-flash-exp",
    "gemini-pro",
    "gemini-1.0-pro"
];

async function testAllModels() {
    console.log("🔍 Testing Gemini Model Availability...");
    console.log("🔑 API Key Present:", !!GOOGLE_API_KEY);

    if (!GOOGLE_API_KEY) {
        console.error("❌ NO API KEY FOUND.");
        return;
    }

    const genAI = new GoogleGenerativeAI(GOOGLE_API_KEY);
    let workingModel = null;

    for (const modelName of modelsToTest) {
        console.log(`\n👉 Testing: ${modelName}...`);
        try {
            const model = genAI.getGenerativeModel({ model: modelName });
            const result = await model.generateContent("Test connection. Reply 'OK'.");
            const response = await result.response;
            const text = response.text();

            console.log(`✅ SUCCESS: ${modelName} is working!`);
            console.log(`   Response: ${text.trim()}`);
            workingModel = modelName;
            break; // Stop after finding the first working one

        } catch (error) {
            console.log(`❌ FAILED: ${modelName}`);
            if (error.message.includes("404")) {
                console.log("   Reason: 404 Not Found (Model not supported by this key/region or invalid name)");
            } else if (error.message.includes("403") || error.message.includes("API key not valid")) {
                console.log("   Reason: 403 Invalid API Key");
            } else {
                console.log(`   Reason: ${error.message.split('\n')[0]}`);
            }
        }
    }

    if (workingModel) {
        console.log(`\n🎉 CONCLUSION: Please update aiService.js to use: '${workingModel}'`);
    } else {
        console.log("\n💀 CONCLUSION: No models worked. The API Key is likely invalid, restricted, or expired.");
    }
}

testAllModels();
