const axios = require('axios');
require('dotenv').config();

const OPENAI_API_KEY = process.env.OPENAI_API_KEY;

console.log("Testing OpenAI API...");

async function testOpenAI() {
    try {
        const response = await axios.post('https://api.openai.com/v1/chat/completions', {
            model: "gpt-4o-mini",
            messages: [
                { role: "system", content: "Test" },
                { role: "user", content: "Hi" }
            ],
            max_tokens: 5
        }, {
            headers: {
                'Authorization': `Bearer ${OPENAI_API_KEY}`,
                'Content-Type': 'application/json'
            }
        });

        console.log("✅ Success! Response:", response.data.choices[0].message.content);
    } catch (error) {
        console.error("❌ Error:", error.response ? JSON.stringify(error.response.data, null, 2) : error.message);
    }
}

testOpenAI();
