const logger = require('../utils/logger');
const { GoogleGenerativeAI } = require("@google/generative-ai");
require('dotenv').config();

const GOOGLE_API_KEY = process.env.GOOGLE_API_KEY;

const sleep = (ms) => new Promise(resolve => setTimeout(resolve, ms));

async function generateHealthAdvisory(livestockData) {
    const genAI = new GoogleGenerativeAI(GOOGLE_API_KEY);
    // Use flash-latest for better availability/quota management
    const model = genAI.getGenerativeModel({ model: "gemini-flash-latest" });

    const { name, breed, age, gender, temperature, activityLevel, weight } = livestockData;

    const prompt = `
    You are a veterinary AI assistant. Analyze the following livestock health data and provide a concise health advisory.
    
    Animal Profile:
    - Name: ${name}
    - Species/Breed: ${breed}
    - Gender: ${gender}
    - Age: ${age} years
    - Weight: ${weight} kg
    
    Current Vitals:
    - Body Temperature: ${temperature}°C
    - Activity Level: ${activityLevel}
    
    Rules:
    1. Normal cattle temp is approx 38-39°C.
    2. Provide a status (Healthy, Warning, Critical).
    3. Give 2-3 specific, actionable recommendations.
    4. Keep it under 100 words.
    
    Response Format:
    Status: [Status]
    Advisory: [Analysis]
    Recommendations:
    - [Rec 1]
    - [Rec 2]
    `;

    const maxRetries = 3;
    let retryCount = 0;

    while (retryCount <= maxRetries) {
        try {
            const result = await model.generateContent(prompt);
            const response = await result.response;
            return response.text();

        } catch (error) {
            logger.error(`AI Service Error (Attempt ${retryCount + 1}/${maxRetries + 1}):`, error.message);

            if (error.message.includes("429") || error.message.includes("Quota exceeded")) {
                if (retryCount < maxRetries) {
                    const delay = Math.pow(2, retryCount) * 1000; // Exponential backoff: 1s, 2s, 4s
                    logger.info(`⏳ Quota exceeded. Retrying in ${delay}ms...`);
                    await sleep(delay);
                    retryCount++;
                    continue;
                } else {
                    return "⚠️ AI Service Busy: Quota exceeded. Please try again later.";
                }
            }

            if (error.message.includes("API key not valid")) {
                return "⚠️ AI Error: Invalid Google API Key.";
            }

            return `⚠️ AI Service Error: ${error.message}`;
        }
    }
}

async function chatWithVetAssistant(farmContext, userQuery, history = [], language = 'en-IN') {
    const genAI = new GoogleGenerativeAI(GOOGLE_API_KEY);

    // User requested "Gemini 2.5 Flash" (Confirmed available in user's project)
    let model;
    try {
        model = genAI.getGenerativeModel({ model: "gemini-2.5-flash" });
    } catch (e) {
        logger.error("Error initializing Gemini 2.5 Flash:", e);
        // Fallback
        model = genAI.getGenerativeModel({ model: "gemini-2.0-flash-exp" });
    }

    // Limit history
    const recentHistory = history.slice(-6);
    let historyContext = "";
    if (recentHistory.length > 0) {
        historyContext = "Previous conversation:\n" + recentHistory.map(msg => `${msg.role === 'user' ? 'User' : 'AI'}: ${msg.content}`).join("\n") + "\n";
    }

    const targetLanguage = language === 'hi-IN' ? 'Hindi' : 'English';

    const educationPrompt = `
    You are GoMata's AI Orchestrator Agent. You coordinate 4 specialized agents:
    1. **Orchestrator Agent**: Routes queries and manages general tasks.
    2. **Health Agent**: Monitors temperature/vitals. Detects fever (>40°C).
    3. **Movement Agent**: Tracks location, grazing patterns, and step count.
    4. **Production Agent**: Analyzes milk yield, weight gain, and profitability.

    **Current Farm Context**:
    - Farm Name: ${farmContext.farmName}
    - Livestock Count: ${farmContext.count}
    - Active Farms: ${farmContext.farms ? farmContext.farms.join(', ') : 'None'}
    - Active Zones: ${farmContext.zones ? farmContext.zones.join(', ') : 'None'}
    - Alerts: ${farmContext.animals.filter(a => a.status !== 'Normal').length} Active
    - Real-time Data:
      ${farmContext.animals.map(a => `- ${a.name} (${a.type}): ${a.status}, Temp ${a.temp}°C, Loc: ${a.location}`).join('\n      ')}

    **User Query**: "${userQuery}"

    **Guidelines**:
    1. **Identify Agent**: Internalize which agent you are (Health, Movement, Production, or Orchestrator) based on the query.
    2. **Direct Answer**: Provide the answer DIRECTLY. Do NOT prefix your response with "[Agent Name]".
    3. **Tone**: Be professional, helpful, and concise. Use a natural, conversational tone.
    4. **Health**: If asking about health, analyze vitals and provide assessment.
    5. **Movement/Location**: If asking about location/walking, provide tracking details.
    6. **Production/Profit**: If asking about milk/money, provide financial/yield analysis.
    7. **Language**: Answer STRICTLY in ${targetLanguage}.

    **Tool / Action capabilities**:
    - **ADD_ANIMAL**: If the user explicitly wants to add/register a new animal, you **MUST** return a **valid JSON object** (and **ONLY** the JSON object). DO NOT include any conversational text outside the JSON.
    
    **Format**:
    \`\`\`json
    {
      "action": "ADD_ANIMAL",
      "data": {
        "name": "Name",
        "breed": "Breed",
        "type": "Cow", 
        "age": 2, 
        "weight": 300,
        "gender": "Female",
        "tagNumber": "TAG-[Random4]" 
      },
      "response": "I have successfully registered [Name] the [Breed] to your herd."
    }
    \`\`\`
    
    **Example Response (Action)**:
    \`\`\`json
    { "action": "ADD_ANIMAL", "data": { "name": "Bessie", "tagNumber": "TAG-1024", "breed": "Jersey" }, "response": "Added Bessie." }
    \`\`\`
    `;

    const maxRetries = 2;
    let retryCount = 0;

    while (retryCount <= maxRetries) {
        try {
            console.time(`GeminiChatTime-${retryCount}`);
            const result = await model.generateContent(educationPrompt);
            const response = await result.response;
            console.timeEnd(`GeminiChatTime-${retryCount}`);
            return response.text();
        } catch (error) {
            logger.error(`AI Chat Error (Attempt ${retryCount + 1}):`, error);
            if (retryCount < maxRetries) {
                await sleep(1000 * (retryCount + 1));
                retryCount++;
                continue;
            }
            // Return concise error for the frontend
            if (error.message.includes('404')) return `Error: Model '${model.model}' not found. Check API Key or Model Name.`;
            return "I am having trouble connecting to the farm data right now. Please try again.";
        }
    }
}

module.exports = { generateHealthAdvisory, chatWithVetAssistant };
