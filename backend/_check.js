const http = require('http');

function req(method, path, token, body) {
    return new Promise((resolve, reject) => {
        const opts = {
            hostname: 'localhost', port: 8000,
            path, method,
            headers: { 'Content-Type': 'application/json' }
        };
        if (token) opts.headers['Authorization'] = 'Bearer ' + token;
        const r = http.request(opts, res => {
            let data = '';
            res.on('data', c => data += c);
            res.on('end', () => { try { resolve(JSON.parse(data)); } catch (e) { resolve(data); } });
        });
        r.on('error', reject);
        if (body) r.write(JSON.stringify(body));
        r.end();
    });
}

(async () => {
    // Login
    const login = await req('POST', '/api/auth/login', null, { email: 'varshananand31@gmail.com', password: 'Kanna' });
    const token = login.data?.token;
    if (!token) { console.log('Login failed:', JSON.stringify(login)); return; }
    console.log('Logged in as Kanna, token:', token.substring(0, 20) + '...');

    // Test endpoints
    const livestock = await req('GET', '/api/livestock', token);
    console.log('\n=== LIVESTOCK ===');
    console.log('Count:', (livestock.data || []).length);
    (livestock.data || []).forEach(l => console.log(' -', l.livestock?.name));

    const farms = await req('GET', '/api/farms', token);
    console.log('\n=== FARMS ===');
    console.log('Count:', (farms.data || []).length);
    (farms.data || []).forEach(f => console.log(' -', f.name, '| zones:', (f.zones || []).length));

    const devices = await req('GET', '/api/devices', token);
    console.log('\n=== DEVICES ===');
    console.log('Count:', (devices.data || []).length);
    (devices.data || []).forEach(d => console.log(' -', d.id, d.animal, d.status));

    const alerts = await req('GET', '/api/alerts', token);
    console.log('\n=== ALERTS ===');
    console.log('Count:', (alerts.data || []).length);
    (alerts.data || []).forEach(a => console.log(' -', a.alertType, a.severity, (a.message || '').substring(0, 70)));

    const breeds = await req('GET', '/api/livestock/breeds/summary', token);
    console.log('\n=== BREEDS ===');
    console.log('Count:', (breeds.data || []).length);
    (breeds.data || []).forEach(b => console.log(' -', b.breed, b.species, 'count:', b.count));
})();
