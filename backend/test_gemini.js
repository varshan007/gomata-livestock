require('dotenv').config();
const axios = require('axios');

async function test() {
    try {
        const url = `https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:generateContent?key=${process.env.GOOGLE_API_KEY}`;
        const response = await axios.post(url, {
            contents: [{ parts: [{ text: "Hello" }] }]
        });
        console.log("Success:", response.data.candidates[0].content.parts[0].text);
    } catch (e) {
        console.log("Error:", e.response ? e.response.data : e.message);
    }
}
test();
