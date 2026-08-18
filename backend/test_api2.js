const axios = require('axios');
const jwt = require('jsonwebtoken');

const token = jwt.sign({ id: '6a840f80bf8bde1bb70b956b', role: 'admin' }, 'supersecret_livestock_key_12345', { expiresIn: '1h' });

async function test() {
    try {
        const livestockRes = await axios.get('http://localhost:8000/api/livestock', {
            headers: { Authorization: `Bearer ${token}` }
        });
        
        console.log('Livestock status:', livestockRes.status);
        console.log('Livestock data type:', Array.isArray(livestockRes.data) ? 'Array' : typeof livestockRes.data);
        console.log('Livestock data:', JSON.stringify(livestockRes.data, null, 2).substring(0, 200));
        
        if (livestockRes.data.data) {
            console.log('Inner data type:', Array.isArray(livestockRes.data.data) ? 'Array' : typeof livestockRes.data.data);
            console.log('Inner data length:', livestockRes.data.data.length);
        }
    } catch (e) {
        console.error('Error:', e.response ? e.response.data : e.message);
    }
}
test();
