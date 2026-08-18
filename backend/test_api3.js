const axios = require('axios');
const jwt = require('jsonwebtoken');

const token = jwt.sign({ id: '6a840f80bf8bde1bb70b956b', role: 'admin' }, 'supersecret_livestock_key_12345', { expiresIn: '1h' });

async function test() {
    try {
        const livestockRes = await axios.get('http://localhost:8000/api/livestock', {
            headers: { Authorization: `Bearer ${token}` }
        });
        
        const animals = livestockRes.data.data;
        console.log('Total locally:', animals.length);
        if(animals.length > 0) {
            console.log('First animal IDs:', animals[0]._id, animals[0].livestock.tagNumber);
        }
    } catch (e) {
        console.error('Error:', e.message);
    }
}
test();
