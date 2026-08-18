# GoMata — MERN → Multi-Agent Integration Guide
### How to layer the AI agent system on your existing GoMata backend
> GoMata · Technical Architecture Team · Confidential · 2026

---

| Stat | Value |
|------|-------|
| Already Built | ~50% |
| Phases | 4 |
| Timeline | ~6 weeks |
| Runtime | Node.js |

---

## Table of Contents
1. [What Your Existing MERN Stack Already Covers](#1-what-your-existing-mern-stack-already-covers)
2. [How the Agent Layer Sits in Your MERN App](#2-how-the-agent-layer-sits-in-your-mern-app)
3. [Phase-by-Phase Integration Roadmap](#3-phase-by-phase-integration-roadmap)
4. [Agent Implementation Blueprints](#4-agent-implementation-blueprints)
5. [Connecting Your React Frontend to the Agent Layer](#5-connecting-your-react-frontend-to-the-agent-layer)
6. [Quick Wins — Start Here on Day 1](#6-quick-wins--start-here-on-day-1)

---

## 1. What Your Existing MERN Stack Already Covers

Your existing GoMata backend is more ready for the multi-agent architecture than you might think. Roughly **50–60% of the agent capabilities already exist** as standard MERN features — they just need to be wrapped in an agent pattern, given clear boundaries, and wired together.

**Legend:**
- ✅ **Done** — Already built, just needs agent wrapper
- 🔶 **Partial** — Exists but needs extension/refactoring
- ❌ **New** — Needs to be built from scratch

| Agent | Status | Your Existing MERN Asset | Gap / What's Missing |
|-------|--------|--------------------------|----------------------|
| Auth & Access (00) | ✅ Done | JWT + bcryptjs middleware | Add RBAC role scopes per agent call |
| Hardware (01) | ✅ Done | MQTT client listening to collar topics | Add edge buffer + reconnect sync logic |
| Onboarding (02) | 🔶 Partial | User registration + animal create APIs | Extract into isolated onboarding flow |
| Interaction (03) | 🔶 Partial | Manual update endpoints exist | Add correction signal pipeline to ML |
| Data Management (04) | ✅ Done | Mongoose models: Animal, Device, Telemetry | Add write-event emitter for Sync Agent |
| Data Processing (05) | ❌ New | Raw telemetry saved to MongoDB as-is | Build preprocessing + feature pipeline |
| Model / ML (06) | 🔶 Partial | Gemini SDK for health insights | Add structured 4-model output format |
| Orchestrator (07) | ❌ New | No orchestration layer exists | Build central router service |
| Health Agent (08) | 🔶 Partial | Gemini breed-range anomaly detection | Wrap into Health Agent with alert emit |
| Location Agent (09) | ❌ New | Leaflet is frontend-only | Build geofence logic on backend |
| Sync Agent (10) | ❌ New | Frontend polls REST APIs | Replace polling with event-driven sync |
| Alert Agent (11) | 🔶 Partial | Alert model + basic alert creation | Add classification + deduplication |
| Notification (12) | ✅ Done | Nodemailer (email) + Twilio (SMS) | Decouple from Alert Agent, add WhatsApp |
| Staff Assignment (13) | ❌ New | Staff model exists, no task assignment | Build assignment + escalation logic |
| Automation Agent (14) | 🔶 Partial | Gemini SDK integrated for AI chat | Expand to agentic actions + tool calls |

---

## 2. How the Agent Layer Sits in Your MERN App

> **Key decision: do not rewrite your existing backend.**

Build the agent system as a separate set of modules within the same codebase that reads from and writes to your existing MongoDB database. Your REST APIs remain unchanged — the frontend keeps calling them. The agents work in the background, enriching the data those APIs serve.

### Recommended Architecture Pattern

> **Pattern: Agent-as-Service** — Each agent is a Node.js class or module. A lightweight orchestrator imports and coordinates them. Agents share access to MongoDB via Mongoose but have clearly defined responsibilities. Communication between agents uses Node.js `EventEmitter` (lightweight) or a Redis pub/sub channel (scalable).

### Recommended Project Structure

```
gomata-backend/
├── src/
│   ├── api/              ← Your existing Express routes (unchanged)
│   ├── models/           ← Your existing Mongoose models (unchanged)
│   ├── mqtt/             ← Your existing MQTT client (minor changes)
│   │
│   ├── agents/           ← NEW: Agent layer lives here
│   │   ├── index.js      ← Agent registry + bootstrapper
│   │   ├── orchestrator/ ← Orchestrator Agent
│   │   ├── hardware/     ← Hardware Agent (wraps MQTT client)
│   │   ├── health/       ← Health Agent (wraps Gemini health calls)
│   │   ├── location/     ← Location Agent (geofence logic)
│   │   ├── dataProc/     ← Data Processing Agent
│   │   ├── model/        ← Model Agent (Gemini + future ML models)
│   │   ├── alert/        ← Alert Agent
│   │   ├── notification/ ← Notification Agent (wraps Nodemailer/Twilio)
│   │   ├── staff/        ← Staff Assignment Agent
│   │   ├── sync/         ← Sync Agent (EventEmitter-based)
│   │   └── automation/   ← Automation Agent (Gemini tool calls)
│   │
│   ├── bus/              ← NEW: Agent event bus
│   │   └── eventBus.js
│   │
│   └── server.js         ← Existing (add agent bootstrap here)
└── package.json
```

### The Agent Event Bus — How Agents Talk to Each Other

Instead of agents calling each other directly (tight coupling), they communicate by emitting and listening to named events on a shared event bus. This means agents are completely independent — you can add, remove, or replace any agent without touching the others.

```js
// src/bus/eventBus.js
const EventEmitter = require("events");

class AgentEventBus extends EventEmitter {}
const bus = new AgentEventBus();
bus.setMaxListeners(50);

module.exports = bus;

// ── How agents use the bus ──────────────────────────────────

// Hardware Agent → emits telemetry
bus.emit("telemetry:received", { animalId, temp, heartRate, location, ts });

// Data Processing Agent → listens + processes
bus.on("telemetry:received", async (data) => {
  const features = await processFeatures(data);
  bus.emit("features:ready", features);
});

// Model Agent → listens + runs inference
bus.on("features:ready", async (features) => {
  const prediction = await runInference(features);
  bus.emit("prediction:ready", prediction);
});

// Orchestrator → routes predictions to Health + Location agents
bus.on("prediction:ready", (prediction) => {
  bus.emit("health:update", prediction.health);
  bus.emit("location:update", prediction.location);
});
```

---

## 3. Phase-by-Phase Integration Roadmap

Build the agent system in 4 phases, from lowest effort to highest. Each phase delivers immediate value and leaves you in a working state — **no "big bang" migration required.**

---

### Phase 1 — Foundation: Event Bus + Wrap Existing Code
**Timeline: Week 1–2**
**Agents: Hardware Agent · Auth Agent · Data Mgmt Agent · Notification Agent · Sync Agent**

- Create the `EventBus` (`eventBus.js`) — the backbone of the entire agent system
- Wrap your MQTT listener in a `HardwareAgent` class that emits `"telemetry:received"` events
- Wrap your Mongoose write operations to emit `"db:write"` events → triggers Sync Agent
- Create `SyncAgent` that listens to `"db:write"` and pushes updates via WebSocket or SSE to frontend
- Move Nodemailer + Twilio into a `NotificationAgent` that listens to `"notification:send"` events
- **Verify:** telemetry flows → DB → sync event → frontend refreshes without polling

---

### Phase 2 — Intelligence: Health + Alert Agents
**Timeline: Week 2–3**
**Agents: Health Agent · Alert Agent · Data Processing Agent**

- Build `DataProcessingAgent`: normalizes raw telemetry, computes rolling averages and z-scores
- Create `HealthAgent` that wraps your existing Gemini breed-range health logic
- `HealthAgent` emits `"alert:create"` events with severity (info / warning / critical) and context
- Build `AlertAgent`: listens to `"alert:create"`, deduplicates by `animalId+type` within 5-min windows
- `AlertAgent` writes deduplicated alerts to MongoDB and emits `"notification:send"` to `NotificationAgent`
- **Verify:** a spike in temperature triggers alert → Twilio SMS fires within seconds

---

### Phase 3 — Spatial: Location + Staff Agents + Orchestrator
**Timeline: Week 3–4**
**Agents: Location Agent · Orchestrator · Staff Assignment Agent**

- Build `LocationAgent`: receives GPS coordinates from `HardwareAgent`, checks against farm geofences in MongoDB
- `LocationAgent` emits `"geofence:breach"` events when animals cross zone boundaries
- Build `StaffAssignmentAgent`: listens to critical health + geofence alerts, creates task records
- `StaffAssignmentAgent` assigns tasks to available vets/staff, sets 10-min escalation timer
- Build `OrchestratorAgent`: central router that coordinates event flow between all agents
- Orchestrator adds fallback: if it restarts, agents re-subscribe to `ModelAgent` directly

---

### Phase 4 — ML + Automation: Full Agent Completion
**Timeline: Week 4–6**
**Agents: Model Agent · Automation Agent · Interaction Agent · Feedback Loop**

- Build `ModelAgent` with 4 structured Gemini prompt templates: health, movement, device, disease risk
- Each model returns structured JSON: `{ prediction, confidence, 7-day-forecast[] }`
- Build `InteractionAgent`: captures vet overrides from API endpoints, stores as labelled corrections
- Correction samples accumulate in a `model_corrections` collection for future fine-tuning
- Upgrade `AutomationAgent`: add `tool_call` definitions so Gemini can call your APIs as actions
- `AutomationAgent` can now add animals, navigate (return route hints), generate reports autonomously

---

## 4. Agent Implementation Blueprints

Each agent follows the same Node.js class pattern: it takes the event bus and any needed Mongoose models as dependencies, registers its listeners in `start()`, and emits events rather than calling other agents directly.

---

### Hardware Agent (01)
**Wraps your existing MQTT client**

Your MQTT client already works. The only change is to wrap it in an agent class that emits events to the bus instead of directly calling functions. Also add an edge buffer — a simple in-memory queue (or MongoDB collection) that stores messages during connectivity loss.

```js
// agents/hardware/HardwareAgent.js
const mqtt = require("mqtt");
const EdgeBuffer = require("./EdgeBuffer");

class HardwareAgent {
  constructor(bus, mqttUrl) {
    this.bus = bus;
    this.client = mqtt.connect(mqttUrl);
    this.buffer = new EdgeBuffer();  // 72h local queue
  }

  start() {
    this.client.on("connect", () => {
      this.client.subscribe("gomata/+/telemetry");
      this.buffer.flush(this.bus);  // replay buffered msgs
    });

    this.client.on("message", (topic, payload) => {
      const hwId = topic.split("/")[1];
      const data = JSON.parse(payload.toString());
      // Emit to agent bus instead of calling DB directly
      this.bus.emit("telemetry:received", {
        hwId, temp: data.temp, heartRate: data.hr,
        lat: data.lat, lng: data.lng, ts: new Date(),
      });
    });

    this.client.on("offline", () => {
      this.client.on("message", (t, p) => this.buffer.push(t, p));
    });
  }
}
module.exports = HardwareAgent;
```

---

### Health Agent (08)
**Wraps your existing Gemini health logic**

You already call Gemini for breed-specific health insights. The Health Agent wraps this in a structured way: it listens to processed telemetry, calls Gemini with a consistent prompt, and emits typed alert events rather than returning data ad-hoc.

```js
// agents/health/HealthAgent.js
const { GoogleGenerativeAI } = require("@google/generative-ai");

class HealthAgent {
  constructor(bus, Animal) {
    this.bus = bus; this.Animal = Animal;
    this.ai = new GoogleGenerativeAI(process.env.GEMINI_KEY);
    this.model = this.ai.getGenerativeModel({ model: "gemini-pro" });
  }

  start() {
    this.bus.on("features:ready", async ({ animalId, features }) => {
      const animal = await this.Animal.findById(animalId).lean();
      const analysis = await this._analyzeHealth(animal, features);

      // Emit structured health update
      this.bus.emit("health:updated", { animalId, analysis });

      // Emit alert if threshold breached
      if (analysis.severity !== "healthy") {
        this.bus.emit("alert:create", {
          type: "health", animalId, severity: analysis.severity,
          message: analysis.summary, data: analysis,
        });
      }
    });
  }

  async _analyzeHealth(animal, features) {
    const prompt = `
      Breed: ${animal.breed}. Temp: ${features.avgTemp}C.
      Heart Rate: ${features.avgHR} bpm. Trend: ${features.trend}.
      Return JSON: { severity, summary, forecast_7d, recommendations }
    `;
    const result = await this.model.generateContent(prompt);
    return JSON.parse(result.response.text());
  }
}
module.exports = HealthAgent;
```

---

### Alert Agent (11)
**Deduplicates + classifies all alerts from all agents**

The Alert Agent is the single point that all alert events flow through. It deduplicates alerts (same animal, same type, within 5 minutes = one alert), classifies severity, saves to MongoDB, and hands delivery to the Notification Agent.

```js
// agents/alert/AlertAgent.js
class AlertAgent {
  constructor(bus, Alert) {
    this.bus = bus; this.Alert = Alert;
    this.recentKeys = new Map();  // dedup cache
  }

  start() {
    this.bus.on("alert:create", async (alertData) => {
      const key = `${alertData.animalId}-${alertData.type}`;
      const last = this.recentKeys.get(key);

      // Deduplicate: skip if same alert within 5 minutes
      if (last && Date.now() - last < 5 * 60 * 1000) return;
      this.recentKeys.set(key, Date.now());

      // Save alert to MongoDB
      const saved = await this.Alert.create({
        ...alertData,
        status: "unread",
        createdAt: new Date(),
      });

      // Route to notification + staff assignment
      if (alertData.severity === "critical") {
        this.bus.emit("notification:send", {
          channels: ["sms", "email", "push"],
          subject: `CRITICAL: ${alertData.message}`,
          alert: saved,
        });
        this.bus.emit("staff:assign", { alert: saved });
      }
    });
  }
}
module.exports = AlertAgent;
```

---

### Automation Agent (14)
**Upgrades your Gemini chat to agentic tool calls**

You already have Gemini integrated. The upgrade here is adding **tool definitions** — structured functions that Gemini can call to take real actions in your app, like adding an animal, fetching health data, or generating a report.

```js
// agents/automation/AutomationAgent.js
class AutomationAgent {
  constructor(bus, models, ai) {
    this.bus = bus; this.models = models; this.ai = ai;
    this.gemini = ai.getGenerativeModel({
      model: "gemini-pro",
      tools: [{ functionDeclarations: this._getTools() }],
    });
  }

  _getTools() {
    return [
      {
        name: "add_animal",
        description: "Register a new livestock animal",
        parameters: {
          type: "object",
          properties: {
            name:  { type: "string" },
            breed: { type: "string" },
            dob:   { type: "string" },
            hwId:  { type: "string", description: "Hardware collar ID" },
          },
          required: ["name", "breed"],
        },
      },
      {
        name: "get_health_summary",
        description: "Get health status for an animal or entire herd",
        parameters: {
          type: "object",
          properties: { animalId: { type: "string" } },
        },
      },
    ];
  }

  async chat(userMessage, userId) {
    const result = await this.gemini.generateContent(userMessage);
    const response = result.response;

    // Handle tool call if Gemini wants to take an action
    if (response.functionCalls()?.length) {
      const call = response.functionCalls()[0];
      const output = await this._executeTool(call.name, call.args, userId);
      return { type: "action", tool: call.name, result: output };
    }
    return { type: "text", text: response.text() };
  }

  async _executeTool(name, args, userId) {
    if (name === "add_animal") {
      return await this.models.Animal.create({ ...args, owner: userId });
    }
    if (name === "get_health_summary") {
      const animal = await this.models.Animal.findById(args.animalId);
      return { name: animal.name, status: animal.healthStatus };
    }
  }
}
module.exports = AutomationAgent;
```

---

### Bootstrapping All Agents in `server.js`

Add the following to your existing `server.js` to start all agents when the app boots. **This is the only change needed to your server entry point.**

```js
// server.js (additions only — your existing code stays the same)
const bus                = require("./bus/eventBus");
const HardwareAgent      = require("./agents/hardware/HardwareAgent");
const DataProcAgent      = require("./agents/dataProc/DataProcAgent");
const HealthAgent        = require("./agents/health/HealthAgent");
const AlertAgent         = require("./agents/alert/AlertAgent");
const NotificationAgent  = require("./agents/notification/NotificationAgent");
const LocationAgent      = require("./agents/location/LocationAgent");
const StaffAgent         = require("./agents/staff/StaffAgent");
const SyncAgent          = require("./agents/sync/SyncAgent");
const OrchestratorAgent  = require("./agents/orchestrator/OrchestratorAgent");
const AutomationAgent    = require("./agents/automation/AutomationAgent");

async function bootstrapAgents() {
  const models = require("./models");  // your existing Mongoose models
  const ai = new GoogleGenerativeAI(process.env.GEMINI_KEY);

  const agents = [
    new HardwareAgent(bus, process.env.MQTT_URL),
    new DataProcAgent(bus, models),
    new HealthAgent(bus, models.Animal, ai),
    new AlertAgent(bus, models.Alert),
    new NotificationAgent(bus),           // wraps Nodemailer + Twilio
    new LocationAgent(bus, models),
    new StaffAgent(bus, models),
    new SyncAgent(bus, io),               // io = your Socket.io instance
    new OrchestratorAgent(bus, agents),
    new AutomationAgent(bus, models, ai),
  ];

  agents.forEach(agent => agent.start());
  console.log(`[GoMata] ${agents.length} agents online`);
}

// Call after DB connects (you already have this)
mongoose.connection.once("open", bootstrapAgents);
```

---

## 5. Connecting Your React Frontend to the Agent Layer

Your React frontend currently polls REST APIs. With the agent layer, you can replace most polling with **real-time push via WebSocket (Socket.io)**. The Sync Agent bridges the backend event bus to the frontend — when any agent updates data, the frontend is notified instantly.

### What Stays the Same vs. What Changes

| Area | Before (Current) | After (With Agents) |
|------|-----------------|---------------------|
| REST API calls (Axios) | ✅ Stays the same | All existing Axios calls still work unchanged |
| Animal / Device CRUD | ✅ Stays the same | Same endpoints, same response format |
| Dashboard data refresh | Polling every 10–30s | Socket.io push → instant, no polling |
| Alerts display | Fetched on page load | Real-time push when AlertAgent fires |
| AI chat (Automation) | Gemini text reply only | Tool-call results + action confirmations |
| Health analytics charts | Recharts with fetched data | Recharts data pushed in real-time by HealthAgent |
| Map / geofence | Static after load | LocationAgent pushes GPS updates live to Leaflet |

### React Hook — Subscribes to All Agent Events

```js
// React — src/hooks/useAgentSocket.js
import { useEffect } from "react";
import { io } from "socket.io-client";

const socket = io(process.env.REACT_APP_API_URL);

export function useAgentSocket(dispatch) {
  useEffect(() => {
    // Health updates from HealthAgent
    socket.on("health:updated", (data) => {
      dispatch({ type: "UPDATE_ANIMAL_HEALTH", payload: data });
    });

    // New alert from AlertAgent
    socket.on("alert:new", (alert) => {
      dispatch({ type: "ADD_ALERT", payload: alert });
    });

    // GPS update from LocationAgent
    socket.on("location:updated", ({ animalId, lat, lng }) => {
      dispatch({ type: "UPDATE_ANIMAL_LOCATION", payload: { animalId, lat, lng } });
    });

    // Staff task assigned
    socket.on("staff:task_assigned", (task) => {
      dispatch({ type: "ADD_STAFF_TASK", payload: task });
    });

    return () => socket.off();  // cleanup
  }, [dispatch]);
}
```

---

## 6. Quick Wins — Start Here on Day 1

These five changes take 1–3 hours each and deliver immediate, visible value. Do these **first** before building the more complex agents.

---

### ① Create the EventBus
Create `eventBus.js` (8 lines of code). This enables every agent to be built independently from this point forward. Zero risk — it's just a module, nothing breaks.

> ⏱ **Time:** 30 min · **Risk:** None · **Value:** Enables everything else

---

### ② Wrap MQTT in HardwareAgent
Your MQTT listener already works. Add 20 lines to make it emit `"telemetry:received"` events to the bus instead of calling functions directly. Instantly decoupled.

> ⏱ **Time:** 1 hour · **Risk:** Very Low · **Value:** Telemetry pipeline begins

---

### ③ Move Nodemailer/Twilio to NotificationAgent
Extract your existing Nodemailer and Twilio calls into a `NotificationAgent` that listens to `"notification:send"` events. Immediately decouples alert creation from delivery — adding WhatsApp later is trivial.

> ⏱ **Time:** 1–2 hours · **Risk:** Low · **Value:** Channel-agnostic alerts

---

### ④ Add Write-Event to Mongoose Models
After any `Animal.save()`, `Alert.save()`, or `Device.save()`, emit a `"db:write"` event. Build a basic `SyncAgent` that listens and broadcasts via Socket.io. Replace one polling call in React with a socket listener.

> ⏱ **Time:** 2 hours · **Risk:** Low · **Value:** Real-time dashboard without polling

---

### ⑤ Structure Gemini into HealthAgent
Move your existing Gemini health call into a `HealthAgent` class with a consistent JSON output format. This makes it trivial to add the 7-day forecast and disease risk features later without refactoring.

> ⏱ **Time:** 2–3 hours · **Risk:** Low · **Value:** Foundation for all ML features

---

> 💡 **WhatsApp Priority:** For Indian farmers, WhatsApp is the #1 communication channel. Once the `NotificationAgent` is built, add WhatsApp via the WhatsApp Business API (Meta) or Twilio's WhatsApp sandbox. This is a single new channel in the `NotificationAgent` — no other agents need to know or change.

---

## Summary Timeline

| Phase | What You Build | When | Immediate Value |
|-------|---------------|------|----------------|
| Phase 1 | EventBus + Wrap existing code as agents | Week 1–2 | Real-time dashboard, no more polling |
| Phase 2 | HealthAgent + AlertAgent + DataProc | Week 2–3 | Instant critical alerts with deduplication |
| Phase 3 | LocationAgent + StaffAgent + Orchestrator | Week 3–4 | Geofence alerts + auto staff assignment |
| Phase 4 | ModelAgent + AutomationAgent + ML loop | Week 4–6 | Agentic AI chat + 7-day health forecast |

---

*GoMata Intelligence System · Agent Integration Technical Guide · MERN Stack · Confidential · 2026*