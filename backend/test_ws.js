const { io } = require("socket.io-client");

const socket = io("http://localhost:8001");

socket.on("connect", () => {
    console.log("Mock UI Client Connected: " + socket.id);
    console.log("Waiting for 'alert:new' events...");
});

socket.on("alert:new", (data) => {
    console.log("\n!!! RECEIVED alert:new !!!");
    console.log(JSON.stringify(data, null, 2));
});

setTimeout(() => {
    console.log("Test timed out.");
    process.exit(0);
}, 20000);
