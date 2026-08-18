const axios = require('axios');
(async () => {
  try {
    const login = await axios.post('https://gomata-backend.onrender.com/api/auth/login', {
      email: 'admin@gomata.com',
      password: 'gMata_Prod_992xQ'
    });
    console.log("Login Success, Token:", login.data.data.token.substring(0,20) + "...");
  } catch (err) {
    console.log("Error:", err.response ? err.response.data : err.message);
  }
})();
