# GoMata: Detailed Multi-Agent Architecture & Infrastructure
### Comprehensive Agent Definitions & Production-Ready Infrastructure Blueprint

This document outlines the detailed architecture for the 15-agent GoMata intelligence system, specifically integrating the **robust infrastructure enhancements** required for a fault-tolerant, scalable production environment.

---

## 🏗️ 1. Core Infrastructure & Networking 

To support 15 independent AI agents seamlessly exchanging thousands of events per minute, the infrastructure must move beyond basic Node.js in-memory operations.

### The Message Broker (Event Bus)
*   **Technology:** Redis Pub/Sub (or Apache Kafka for extreme scale).
*   **Why:** Replaces the Node.js `EventEmitter`. An external broker ensures that if the main server restarts, queued events are not lost in volatile memory. It allows agents to be split across multiple servers (horizontal scaling) in the future.
*   **Implementation:** Agents publish standardized JSON payloads containing a `traceId` (for logging) and a `timestamp` to specific topic channels (e.g., `telemetry:received`, `alert:critical`).

### Temporal State Management
*   **Technology:** BullMQ (backed by Redis).
*   **Why:** Replaces standard `setTimeout()`. Used heavily by the **Staff Assignment Agent** for escalation delays (e.g., "Wait 10 minutes, if task is unacknowledged, escalate"). BullMQ persists these schedules in Redis; if the Node container crashes during the 10-minute wait, the timer safely resumes on reboot.

### Processing Offload
*   **Technology:** Node.js Worker Threads or FastAPI (Python) Microservices.
*   **Why:** The **Data Processing Agent** handles complex matrices and time-series arrays. Moving this off the main thread keeps the REST APIs and WebSockets lightning-fast.

### LLM Gateway & Rate Limiting
*   **Technology:** Token Buckets & Circuit Breakers via Redis.
*   **Why:** A runaway hardware glitch emitting false alerts could cause the **Health Agent** to fire 10,000 Gemini requests, causing excessive billing and API bans. The Gateway batches prompts, caches identical queries, and enforces hard rate limits.

---

## 🤖 2. Detailed Agent Definitions (The 15 Agents)

### LAYER 0 — Security & Access
#### `AGENT 00` · Auth & Access Agent
*   **Role:** Issues Role-Based Access Control (RBAC) tokens (Farmer, Vet, Staff, Admin) guarding every agent interaction.
*   **Infrastructure Need:** JWT verification middleware. Tokens dictate the permissible limits for the Automation Agent (e.g., a Staff token cannot delete an animal profile).

### LAYER 1 — Data Ingestion
#### `AGENT 01` · Hardware Agent
*   **Role:** Listens to IoT devices (collars/sensors) via MQTT and emits raw telemetry.
*   **Infrastructure Need:** MQTT Broker (e.g., Eclipse Mosquitto or AWS IoT Core). Crucially relies on an **Edge Buffer** (local database queue) to replay up to 72 hours of data if the connection drops.
#### `AGENT 02` · Onboarding Agent
*   **Role:** Handles one-time farm configuration, hardware pairings, and breed definitions.
*   **Infrastructure Need:** Standard CRUD operations hitting MongoDB.
#### `AGENT 03` · Interaction Agent
*   **Role:** Captures ongoing mid-use manual input (vet diagnostic overrides, feed records) and feeds correction signals back to the ML models.
*   **Infrastructure Need:** API endpoints linked to the Event Bus to emit `feedback:received` events.

### LAYER 2 — Data Management
#### `AGENT 04` · Data Management Agent
*   **Role:** The central source of truth. Normalizes data, resolves entity relations (Hardware → Animal), and pushes to the Time-Series store.
*   **Infrastructure Need:** MongoDB Atlas (Time-Series Collections) for highly performant storage of fast-moving sequential sensor data.

### LAYER 3 — Preprocessing
#### `AGENT 05` · Data Processing Agent
*   **Role:** Transforms raw telemetry into ML-ready feature sets (imputation, windowing, z-scores).
*   **Infrastructure Need:** **Node.js Worker Threads** or a native Python microservice to prevent blocking the main event loop during intense array operations.

### LAYER 4 — ML Inference & Feedback
#### `AGENT 06` · Model Agent (The LLM Core)
*   **Role:** Hosts 4 specialized inference models (Health Forecast, Movement Analytics, Device Status, Disease Risk).
*   **Infrastructure Need:** Protected by the **LLM Gateway (Rate Limiter)** to prevent API quota exhaustion. Utilizes vector embeddings if comparing historical health precedents.

### LAYER 5 — Orchestration Hub
#### `AGENT 07` · Orchestrator Agent
*   **Role:** Intelligently routes insights from the Model Agent to downstream domain agents based on priority queues.
*   **Infrastructure Need:** Subscribes to Redis channels. Features a fallback mechanism: if the Orchestrator goes offline, domain agents directly subscribe to the Model Agent's output via Redis.

### LAYER 6 — Domain Experts
#### `AGENT 08` · Health Agent
*   **Role:** Generates clinical insights and triggers severity-graded health alerts. 
#### `AGENT 09` · Location Agent
*   **Role:** Handles spatial intelligence, checks GPS vs geofence coordinates, and fires breach alerts.
#### `AGENT 10` · Sync Agent (Event-Driven)
*   **Role:** Syncs backend state changes to the frontend UI instantly.
*   **Infrastructure Need:** **WebSockets (Socket.io)**. Replaces frontend HTTP polling with direct, instant pushes.

### LAYER 7 — Alerts, Delivery & Action
#### `AGENT 11` · Alert Agent
*   **Role:** Aggregates, deduplicates, and prioritizes anomalies across the system.
*   **Infrastructure Need:** Redis caching layer to quickly check if a similar alert fired in the last 5 minutes (deduplication).
#### `AGENT 12` · Notification Delivery Agent
*   **Role:** Multi-channel outbound delivery (SMS, Email, Push).
*   **Infrastructure Need:** Integrated with Twilio (SMS/WhatsApp Business API) and Firebase Cloud Messaging (FCM) for push notifications.
#### `AGENT 13` · Staff Assignment Agent
*   **Role:** Creates actionable tasks for staff based on critical alerts.
*   **Infrastructure Need:** **BullMQ (Redis)** for durable delay timers. Ensures a task escalates to a manager if unacknowledged, surviving any server restarts.

### LAYER 8 — Conversational Automation
#### `AGENT 14` · Automation Agent (GoMata AI Assistant)
*   **Role:** LLM-powered conversational interface capable of tool-calling to execute real app functions (e.g., generating reports, adding animals).
*   **Infrastructure Need:** LangChain or custom Gemini Tool schemas to translate natural language into strict JSON API executions. Scoped securely by the Auth Agent.

---

## 🔄 Tracing the Lifecycle of a Critical Event
To visualize the robustness of this pipeline:

1. **(Agent 01)** ESP32 collar sends a low heart-rate reading via MQTT. The Hardware Agent attaches `traceId: 104-A` and publishes to Redis.
2. **(Agent 05)** The Data Processing worker thread normalizes the drop without crashing the API server.
3. **(Agent 06)** The Model Agent (safeguarded by the Gateway) queries Gemini, concluding a 90% risk of cardiac distress.
4. **(Agent 08 & 11)** The Health Agent registers the severity; the Alert Agent ensures it hasn't alerted on this cow in the last 5 minutes.
5. **(Agent 10, 12, & 13)** The Sync Agent pushes a red flashing light to the React frontend via WebSockets. The Notification Agent pings the farmer's WhatsApp. The Staff Assignment Agent sets a BullMQ 10-minute escalation timer for the on-site Vet.
