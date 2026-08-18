import React, { createContext, useContext, useEffect, useState, useRef } from 'react';
import { io } from 'socket.io-client';

const BACKEND_URL = (process.env.REACT_APP_API_URL && process.env.REACT_APP_API_URL.startsWith('http'))
    ? process.env.REACT_APP_API_URL.replace('/api', '')
    : 'http://localhost:8000';

// 1. Create the Context
const LiveTelemetryContext = createContext();

// 2. Custom Hook for easy access inside any Component
export const useLiveTelemetry = () => useContext(LiveTelemetryContext);

// 3. The Provider Component that wraps the App
export const LiveTelemetryProvider = ({ children }) => {
    // liveData Dictionary maps deviceId/tagNumber to Real-Time Stats
    const [liveData, setLiveData] = useState({});

    // Store global alerts that haven't been dismissed
    const [liveAlerts, setLiveAlerts] = useState([]);

    const socketRef = useRef(null);

    useEffect(() => {
        // Initialize single, persistent Socket Connection
        socketRef.current = io(BACKEND_URL, {
            reconnectionAttempts: 10,
            reconnectionDelay: 1000,
            transports: ['websocket', 'polling']
        });

        const socket = socketRef.current;

        socket.on('connect', () => {
            console.log('[LiveTelemetryContext] Connected to Global SyncAgent ✅');

            // ── Join tenant room for multi-tenant isolation ──
            // Read user info from localStorage to get tenantId
            try {
                const userRaw = localStorage.getItem('user');
                if (userRaw) {
                    const user = JSON.parse(userRaw);
                    const tenantId = user.tenantId || user._id || user.id;
                    if (tenantId) {
                        socket.emit('join_tenant', tenantId);
                        console.log(`[LiveTelemetryContext] Joined tenant room: tenant:${tenantId}`);
                    }
                }
            } catch (e) {
                console.warn('[LiveTelemetryContext] Could not read user for tenant room:', e);
            }
        });

        // Listen for live telemetry (now scoped to tenant room by backend)
        socket.on('sync:telemetry_live', (data) => {
            setLiveData(prevData => ({
                ...prevData,
                [data.hwId]: {
                    temperature: parseFloat(data.temp),
                    heartRate: data.heartRate,
                    location: { lat: data.lat, lng: data.lng },
                    battery: data.battery,
                    lastUpdate: data.ts
                }
            }));
        });

        // Listen for alerts (now scoped to tenant room by backend)
        socket.on('alert:new', (alertData) => {
            console.log("[Global] Alert Received:", alertData);
            setLiveAlerts(prev => [alertData, ...prev]);
        });

        return () => {
            console.log('[LiveTelemetryContext] Disconnecting Global Socket...');
            socket.disconnect();
        };
    }, []);

    return (
        <LiveTelemetryContext.Provider value={{ liveData, liveAlerts, socket: socketRef.current }}>
            {children}
        </LiveTelemetryContext.Provider>
    );
};
