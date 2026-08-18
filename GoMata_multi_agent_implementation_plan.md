# GoMata: Multi-Agent Implementation & Production Rollout Plan
### Phase-by-Phase Roadmap to a Robust AI Infrastructure

This document outlines the step-by-step implementation plan to build the 15-agent architecture discussed previously. This roadmap is designed to prevent "big bang" deployments, ensuring that every phase delivers immediate functionality to the UI without breaking existing systems.

---

## 🛠️ Phase 1: Infrastructure Foundations & Data Ingestion
**Goal:** Establish the bulletproof core message bus and the reliable ingestion of IoT telemetry.
**Duration:** Week 1-2

### Step 1: Deploy Core Infrastructure
- Provision an external **Redis Server** (hosted on Redis Labs or locally via Docker) for Pub/Sub events and BullMQ queues.
- **Outcome:** The foundation for agent decoupling and state survival over restarts.

### Step 2: Implement the Event Bus
- Create `/agents/bus/RedisEventBus.js` wrapper class handling `publish` and `subscribe` to Redis channels with serialized JSON (ensuring cross-language compatibility if needed).
- Add `TraceID` injection logic to stamp IDs on entering events.

### Step 3: Auth & Onboarding Agents (Agents 00, 02)
- Separate existing API logic into an `AuthAgent` generating scoped JWTs (Admin/Vet/Staff).
- Structure existing registration logic into an `OnboardingAgent`.

### Step 4: Hardware Agent with Edge Buffer (Agent 01)
- Wrap current MQTT listener in a Class structure.
- Implement an Edge Buffer: when MQTT drops connection, save raw JSON to a MongoDB `buffer` collection. On reconnect, flush buffer to the Event Bus with retroactive timestamps.

### Step 5: Sync Agent (Agent 10)
- Install `Socket.io` and configure a `SyncAgent` listening for `db:write` events.
- Update React Dashboard to remove HTTP polling and listen entirely to WebSocket pushes.

*✔️ Testing Checkpoint: Hardware telemetry hits MQTT -> Redis Bus -> Sync Agent -> UI Live update.*

---

## 🧠 Phase 2: Processing, Intelligence, and Delivery
**Goal:** Ingest telemetry, discover anomalies, and notify farmers securely.
**Duration:** Week 3-4

### Step 1: Data Processing Agent (Agent 05)
- Set up a **Node.js Worker Thread** to pick up `telemetry:received` payloads.
- Normalization (filling gaps, z-scores) running securely off the main event loop.

### Step 2: Health Agent (Agent 08)
- Structure Gemini logic to process the normalized array, returning structured JSON indicating status and severity.
- If severity is `Warning/Critical`, emit an `alert:create` event to the bus.

### Step 3: Alert Agent (Agent 11)
- Listen for `alert:create`.
- Check Redis Cache keys `(animalID-alertType)` to ensure no duplicates within a 5-minute sliding window. Store to MongoDB.

### Step 4: Notification Delivery Agent (Agent 12)
- Separate Nodemailer, FCM Push, and Twilio SMS/WhatsApp logic.
- Listen for `notification:send` and execute reliable third-party API dispatches.

*✔️ Testing Checkpoint: Spiking a mock cow's temperature instantly triggers a single WhatsApp message without duplicating, driven entirely by event-chains.*

---

## 🗺️ Phase 3: Spatial Insight and Operations
**Goal:** Track livestock physically and assign critical tasks to farm personnel gracefully.
**Duration:** Week 5

### Step 1: Location Agent (Agent 09)
- Run geospatial polygon intersections (e.g., Turfs.js) to detect if an incoming GPS point has exited the MongoDB `Geofence` boundaries.
- Emit `geofence:breach` to the Alert Agent.

### Step 2: Staff Assignment Agent (Agent 13)
- Install and configure **BullMQ**.
- On critical alerts, assign task to on-shift veterinarians in the DB.
- Create a delayed Redis job (`queue.add('escalate-task', data, { delay: 600000 })`).
- If task acknowledged within 10 minutes from React UI via REST API, cleanly remove job from queue. Otherwise, escalate.

*✔️ Testing Checkpoint: Forcing a Node.js restart during a 10-minute wait window to ensure BullMQ correctly resumes the escalation timer post-reboot.*

---

## 🤖 Phase 4: Full ML Automation & Feedback Loops
**Goal:** Empower the orchestration of all intelligence into predictive capabilities and automated interactions.
**Duration:** Week 6-7

### Step 1: LLM Gateway & Rate Limiting
- Build a throttling middleware on the Gemini SDK. Ensure the system cannot exceed 15 model calls per second to protect quotas.

### Step 2: The Model Agent (Agent 06)
- Operationalize the 4 discrete prediction models (Health, Movement, Battery, Disease Risk) behind the Gateway.

### Step 3: Orchestrator Agent (Agent 07)
- Deploy central routing. Have Health/Location listen to Orchestrator rather than processing events directly.
- Build fallback direct-subscriptions in Domain agents utilizing Redis pub/sub patterns.

### Step 4: Interaction Agent & Feedback Loop (Agent 03)
- Build UI flows where Vets can "Correct" a Gemini prediction.
- Save to a `correction_signals` DB collection where an automated script bundles training data for Vertex AI fine-tuning periodically.

### Step 5: Automation Agent (Agent 14)
- Give Chat UI access to a locked-down LangChain/Gemini interface defining explicit tools (e.g. `addAnimal(name, breed)`, `getTemperature(id)`). Verify tokens via Auth Agent before execution.

*✔️ Testing Checkpoint: Chatting naturally with GoMata AI to register an animal, and observing database persistence successfully completed.*

---

## 🔒 Phase 5: Hardening & Testing Strategy
**Goal:** Ensure the system survives stress, failures, and production volumes.
**Duration:** Week 8

### Step 1: Load Testing (Artillery / K6)
- Simulate 2,000 MQTT collars transmitting every 10 seconds.
- Monitor Node V8 Heap memory usage and Worker Thread event loop lag.

### Step 2: Chaos Engineering
- Deliberately kill the Orchestrator Node.js process. Ensure Domain agents gracefully fallback to Redis pub/sub.
- Drop network access to MongoDB for 5 minutes; rely on the Hardware Agent's Edge Buffer queue to preserve IoT telemetry.

### Step 3: End-to-End Test Automation (Cypress + Jest)
- Write Jest suites mocking `eventBus.emit()` to test business logic independently of DB states.

---

## 🚀 Phase 6: Production Deployment Architecture
**Goal:** Launching efficiently, securely, and scalably to the Cloud.

### Architecture
- **Web App / APIs:** Node.js App containerized effectively with **Docker**. Deployed scaling linearly over **AWS ECS (Fargate)** or **Google Cloud Run**.
- **Frontend App:** Static React build hosted via CDN on **Vercel** or **AWS S3/CloudFront** for lightning-fast delivery.
- **Message Bus & Session:** Managed **Redis Cluster** (e.g., Upstash or AWS ElastiCache) handling pub/sub and BullMQ logic reliably.
- **Database:** **MongoDB Atlas Serverless** cluster utilizing dedicated Time-Series Collections for fast insertions and compression of telemetry data.
- **Ingestion:** Managed **AWS IoT Core** or **HiveMQ Cloud** acting as the resilient MQTT Broker managing TLS 1.2 traffic exclusively from the ESP32 sensors.

### CI/CD Pipeline
- **GitHub Actions** Pipeline:
  - Linting -> Jest Tests -> Build Docker Image -> Push to Registry -> Zero-Downtime Rolling Deployment.

---

By the conclusion of Phase 6, GoMata will operate as a fully realized, self-healing, agentic ecosystem, seamlessly managing tens of thousands of livestock interactions with human-like precision and robust technical endurance.
