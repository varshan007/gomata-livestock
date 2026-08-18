import { useEffect, useRef } from 'react';
import { io } from 'socket.io-client';

const BACKEND_URL = process.env.REACT_APP_API_URL || 'http://localhost:8000';

/**
 * useAgentSocket
 * A custom React hook that connects to the GoMata SinkAgent via WebSockets.
 * Passes real-time payload updates directly to the provided callbacks.
 */
export function useAgentSocket(callbacks) {
    const socketRef = useRef(null);

    useEffect(() => {
        // Initialize Socket Connection
        socketRef.current = io(BACKEND_URL, {
            reconnectionAttempts: 5,
            reconnectionDelay: 1000,
        });

        const socket = socketRef.current;

        socket.on('connect', () => {
            console.log('[AgentSocket] Connected to GoMata SyncAgent ✅');
        });

        // Listen for live telemetry bypass updates
        if (callbacks.onTelemetry) {
            socket.on('sync:telemetry_live', callbacks.onTelemetry);
        }

        // Listen for new critical alerts
        if (callbacks.onAlert) {
            socket.on('alert:new', callbacks.onAlert);
        }

        // Listen for health state updates (from Model Agent)
        if (callbacks.onHealth) {
            socket.on('health:updated', callbacks.onHealth);
        }

        // Listen for dynamic database syncing 
        if (callbacks.onDbSync) {
            socket.on('sync:db_Livestock', callbacks.onDbSync);
        }

        return () => {
            console.log('[AgentSocket] Disconnecting...');
            socket.disconnect();
        };
    }, [callbacks]);

    return socketRef.current;
}
