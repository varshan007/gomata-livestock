const axios = require('axios');
(async () => {
  try {
    const login = await axios.post('https://gomata-backend.onrender.com/api/auth/login', {
      email: 'admin@gomata.com',
      password: 'gMata_Prod_992xQ'
    });
    
    const livestock = await axios.get('https://gomata-backend.onrender.com/api/livestock', {
      headers: { Authorization: `Bearer ${login.data.data.token}` }
    });
    console.log("Raw Response Data keys:", Object.keys(livestock.data));
    console.log("Success:", livestock.data.success);
    console.log("Is Data Array?", Array.isArray(livestock.data.data));
    console.log("Data length:", livestock.data.data.length);
    if (livestock.data.data.length > 0) {
        console.log("First item:", JSON.stringify(livestock.data.data[0], null, 2).substring(0, 500));
    }
  } catch (err) {
    console.log("Error:", err.response ? err.response.data : err.message);
  }
})();
