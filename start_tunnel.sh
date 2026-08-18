#!/bin/bash
while true; do
    echo "Starting Localtunnel..."
    npx -y localtunnel --port 8000 --subdomain livestock-varshan
    echo "Localtunnel crashed or disconnected. Restarting in 5 seconds..."
    sleep 5
done
