const aedes = require('aedes')();
const server = require('net').createServer(aedes.handle);
const PORT = 1883;

if (process.env.NODE_ENV !== 'production') {
    server.listen(PORT, function () {
        console.log('✅ Local MQTT Broker started and listening on TCP port', PORT);
    });
} else {
    console.log('✅ Local MQTT Broker initialized (TCP listener disabled for production)');
}

aedes.on('client', function (client) {
    console.log(`[MQTT Local Broker] Client Connected: \x1b[33m${(client ? client.id : client)}\x1b[0m`);
});

aedes.on('clientDisconnect', function (client) {
    console.log(`[MQTT Local Broker] Client Disconnected: \x1b[33m${(client ? client.id : client)}\x1b[0m`);
});
