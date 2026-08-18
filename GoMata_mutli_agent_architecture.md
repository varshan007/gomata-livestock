# GoMata — Multi-Agent Architecture v2.0
### Livestock Intelligence Platform · Confidential · Internal Architecture Document · 2026

---

## Table of Contents
1. [Executive Summary](#1-executive-summary)
2. [Layer-by-Layer Agent Breakdown](#2-layer-by-layer-agent-breakdown)
3. [v2.0 Changes & Improvements](#3-v20-changes--improvements)
4. [Data Flow & Inter-Agent Communication](#4-data-flow--inter-agent-communication)
5. [UI Pages & Agent Mapping](#5-ui-pages--agent-mapping)
6. [Recommended Next Steps](#6-recommended-next-steps)

---

## 1. Executive Summary

GoMata is a livestock intelligence platform designed to help farmers monitor animal health, location, device status, and behavioral patterns in real time. To power this intelligence, GoMata employs a **multi-agent architecture** — a coordinated network of 15 specialized AI agents, each responsible for a distinct domain, working together to transform raw IoT sensor data into actionable farm insights.

Version 2.0 introduces **4 new agents** and **modifies 4 existing ones**, addressing critical gaps identified in v1: the absence of role-based security, notification delivery, staff task assignment, a model learning feedback loop, and edge-side data buffering.

| Stat | Value |
|------|-------|
| Total Agents | 15 |
| Layers | 8 |
| ML Models | 4 |
| New in v2 | 4 Agents |
| Modified in v2 | 4 Agents |

### Design Principles

| Principle | Description |
|-----------|-------------|
| **Separation of Concerns** | Each agent owns exactly one domain — ingestion, processing, inference, routing, or delivery. |
| **Fault Tolerance** | No single point of failure. Orchestrator has pub/sub fallback; Hardware Agent has 72h edge buffer. |
| **Continuous Learning** | Manual vet overrides feed back into model fine-tuning via the Interaction Agent feedback loop. |
| **Event-Driven Sync** | The Sync Agent fires only on confirmed database writes — no polling, no wasted cycles. |
| **Role-Scoped Actions** | The Auth Agent gates every downstream agent — users can only do what their role permits. |
| **Decoupled Delivery** | Alert creation and notification delivery are separate agents — channels can evolve independently. |

---

## 2. Layer-by-Layer Agent Breakdown

---

### LAYER 0 — Security & Access
> Foundation of every agent interaction

#### `AGENT 00` · Auth & Access Agent 🆕 NEW
**Identity, Roles & Permission Scopes**

Guards every agent interaction in the system. Manages role-based access control (RBAC) with four roles: Farmer, Vet, Staff, and Admin. Issues scoped tokens to downstream agents so the Automation Agent can only perform actions the logged-in user is authorized for. All agent-to-agent calls carry a JWT with the user's permission scope attached.

**Capabilities:**
- JWT / Session Tokens
- RBAC: Farmer / Vet / Staff / Admin
- Per-Agent Permission Scopes
- Audit Log
- Token Refresh
- Scoped Automation Actions

> ✅ *Authenticated session context passed to all layers*

---

### LAYER 1 — Data Ingestion
> 3 agents collect all raw data entering the system

#### `AGENT 01` · Hardware Agent 🔄 UPDATED
**IoT Device Telemetry Collector**

Continuously polls IoT devices — collars, sensors, and GPS trackers — using unique hardware IDs. Now ships with an on-device **edge buffer** that stores up to 72 hours of data locally during connectivity loss, syncing with timestamp correction when the connection is restored.

**Capabilities:**
- Temperature
- GPS Location
- Heart Rate
- Battery Level
- Device Status
- Activity Metrics

> 💡 *Edge Buffer: 72h local storage. Auto-syncs with timestamp correction on reconnect — no data gaps.*

---

#### `AGENT 02` · Onboarding Agent 🔄 UPDATED
**One-Time Farm Setup Handler**

Handles the one-time farm configuration flow — animal registration, hardware ID pairings, farm zone definitions, breed details, and staff assignments. Formerly part of the v1 User Agent; split to isolate setup logic from ongoing interactions.

**Capabilities:**
- Animal Registration
- Hardware ID Pairing
- Farm Zone Setup
- Breed Configuration
- Staff Onboarding

> 💡 *Split from v1 User Agent. Only active during setup and major reconfigurations.*

---

#### `AGENT 03` · Interaction Agent 🆕 NEW
**Ongoing Mid-Use Input & Feedback Capture**

Captures ongoing farmer and vet input during regular use — manual health observations, vet diagnostic overrides, feed records, and corrections to AI predictions. These corrections are routed to the Model Agent's feedback loop, making the ML models progressively smarter over time as the farm provides real-world ground truth.

**Capabilities:**
- Manual Health Observations
- Vet Diagnostic Overrides
- Feed & Medication Records
- Model Correction Signals
- Real-Time Annotations

> 💡 *New in v2. Correction signals feed back into ML model fine-tuning via the Model Agent feedback loop.*

> ✅ *Raw telemetry + user-annotated data flows to Layer 2*

---

### LAYER 2 — Data Management
> Single source of truth — maps, stores, and validates all data

#### `AGENT 04` · Data Management Agent
**Normalized Time-Series Database Manager**

The central source of truth for the entire system. Merges hardware telemetry with animal profiles by matching hardware IDs, resolves all entity relationships (animals, devices, farms, users, staff), and maintains the normalized time-series database. On every confirmed write, it emits a sync event to the Sync Agent to propagate changes across the UI.

**Capabilities:**
- Hardware ↔ Animal Mapping
- Time-Series Store
- User ↔ Farm Relations
- Entity Resolution
- Write-Triggered Sync Events
- Input Validation Layer
- Data Versioning

> ✅ *Structured, validated time-series data flows to Layer 3*

---

### LAYER 3 — Preprocessing
> Transforms raw data into ML-ready feature sets

#### `AGENT 05` · Data Processing Agent
**Feature Engineering & ML Preparation**

Transforms raw time-series livestock data into model-ready feature sets. Handles normalization, missing value imputation, sliding window construction, and feature engineering specific to each prediction task. Also preprocesses manual correction signals from the Interaction Agent as labelled training samples for model improvement.

**Capabilities:**
- Time-Series Windowing
- Missing Value Imputation
- Normalization & Scaling
- Per-Animal Feature Sequences
- Temporal Alignment
- Correction Sample Labelling
- Per-Task Feature Engineering

> ✅ *ML-ready feature vectors per animal per task flows to Layer 4*

---

### LAYER 4 — ML Inference & Feedback Loop
> 4 specialized models + continuous learning

#### `AGENT 06` · Model Agent 🔄 UPDATED
**4 Specialized ML Models + Feedback Loop**

The Model Agent hosts four specialized ML/neural network models. Each receives preprocessed feature vectors for its domain and returns structured prediction outputs with confidence scores.

| Model | Description & Output |
|-------|---------------------|
| 🌡️ **Health Forecasting** | 7-day temperature, fever risk, vitals forecasting per animal. Predicts which animals will trend critical before it happens. |
| 🚶 **Movement Analytics** | Grazing pattern analysis, daily distance traveled, behavioral anomaly detection. Flags unusual stillness or erratic movement. |
| 📡 **Device Forecasting** | Battery life prediction, signal quality forecasting, offline risk detection. Prevents unexpected device failures. |
| 🦠 **Disease Risk Model** | Herd-level infection spread scoring. Early warning system for infectious disease propagation using spatial + health data. |

> 🔁 **Feedback Loop (NEW in v2):** Manual vet overrides from the Interaction Agent are accumulated as labelled correction samples. These are periodically used to fine-tune each model — improving accuracy over time without requiring manual retraining. The more your vets correct the system, the better it gets.

> ✅ *Prediction outputs + confidence scores flow to Layer 5*

---

### LAYER 5 — Orchestration Hub
> Central nervous system — routes everything, fails gracefully

#### `AGENT 07` · Orchestrator Agent 🔄 UPDATED
**Smart Routing, Priority Queuing & Fallback**

The central coordinator. Receives outputs from the Model Agent and Data Management Agent, then intelligently routes processed insights to every downstream domain agent based on data type, priority, and agent availability.

**New in v2:** Fallback pub/sub routing ensures that if the Orchestrator goes offline, domain agents can subscribe directly to the Model Agent output stream — eliminating a critical single point of failure.

**Capabilities:**
- Smart Routing Engine
- Priority Queue
- Agent Coordination
- Load Balancing
- Context Window Management
- Fallback: Direct Pub/Sub Subscriptions
- Health Monitoring

> ⚠️ *v2 Fallback: If Orchestrator is unavailable, domain agents subscribe directly to Model Agent via pub/sub bus.*

> ✅ *Routed intelligence streams flow to Layer 6 domain agents*

---

### LAYER 6 — Domain Agents
> Health, Location & Sync — 3 specialized domain experts

#### `AGENT 08` · Health Agent
**Clinical Insights & Health Analytics**

Processes health metrics from the Orchestrator, generates clinical insights, and keeps the Health Analytics page updated in real time. Triggers severity-graded alerts (Warning / Critical) to the Alert Agent when thresholds are breached.

**Capabilities:**
- Vitals Monitoring
- Disease Signal Detection
- Health Analytics Page
- 7-Day Health Forecast Display
- Alert Triggering

---

#### `AGENT 09` · Location Agent
**Spatial Intelligence & Geofence Monitoring**

Handles all spatial intelligence — herd zone segregation, geofence boundary monitoring, zone health scoring, and live map layer updates. Fires breach alerts when any animal exits a defined geofence radius.

**Capabilities:**
- Geofence Monitoring
- Herd Zone Segregation
- Zone Health Scoring
- Map Intelligence Page
- Breach Alert Triggers

---

#### `AGENT 10` · Sync Agent 🔄 UPDATED
**Event-Driven UI Synchronization** *(formerly Update Agent)*

Renamed from v1's Update Agent and redesigned to be purely event-driven. Fires only when the Data Management Agent confirms a successful write — no continuous polling. Propagates changes to animal profiles, device records, breed info, and staff views across the entire application.

**Capabilities:**
- Animal Profile Sync
- Device Registry Updates
- Breed Record Propagation
- Staff View Updates
- Write-Event Triggered Only

> 💡 *Renamed + redesigned in v2. Fires on write events only — not polling. Far more efficient.*

> ✅ *Cross-system events, alerts, and data changes flow to Layer 7*

---

### LAYER 7 — Alerts, Delivery & Staff Coordination
> 3 agents handle what happens after an alert fires

#### `AGENT 11` · Alert Agent
**Alert Aggregation, Classification & Deduplication**

Aggregates alert events from every upstream agent — health, location, device, system, and user. Classifies each alert by type and severity, deduplicates redundant signals, and prioritizes them before publishing to the Alerts Centre UI and passing delivery-ready alerts to the Notification Delivery Agent.

**Capabilities:**
- Livestock Alerts
- Farm Alerts
- Device Alerts
- Staff Alerts
- System & App Alerts
- Severity Classification
- Deduplication

---

#### `AGENT 12` · Notification Delivery Agent 🆕 NEW
**Multi-Channel Alert Delivery**

Receives classified alerts from the Alert Agent and delivers them through the right channel based on user preferences and alert severity. Completely decoupled from alert logic — new delivery channels (e.g., Telegram, voice calls) can be added without touching the Alert Agent.

**Capabilities:**
- Push Notifications
- WhatsApp / SMS *(key for India)*
- Email Delivery
- In-App Notifications
- Critical Escalation Calls

---

#### `AGENT 13` · Staff Assignment Agent 🆕 NEW
**Task Auto-Assignment & Escalation**

Receives critical health and location alerts and automatically assigns actionable tasks to available staff members based on role, availability, and proximity. Tracks acknowledgement status and escalates unactioned tasks if not confirmed within a configurable threshold time.

**Capabilities:**
- Auto Task Assignment
- Vet Routing by Availability
- Escalation Timer
- Acknowledgement Tracking
- Staff Notification
- Task Status Dashboard

> 📝 *Example: "Aman has fever (39.5°C) → Assigned to Dr. Rao. Unacknowledged after 10 min → escalate to Farm Manager."*

> ✅ *Full context, permissions, and insights flow to Layer 8*

---

### LAYER 8 — Conversational Automation
> LLM-powered agent that acts on the farmer's behalf

#### `AGENT 14` · Automation Agent
**LLM-Powered Conversational AI Interface**

The user-facing conversational AI layer. Synthesizes insights from all 14 upstream agents to answer natural language queries, perform agentic UI actions (add livestock, fill forms, navigate pages), generate farm reports, and complete complex tasks on behalf of the user — all scoped by the Auth Agent's permission token.

**Capabilities:**
- LLM Chat Interface
- Add / Edit Animal Records
- In-App Navigation
- Report Generation
- Voice Command Support
- Auth-Scoped Actions Only
- Cross-Agent Insight Synthesis

---

## 3. v2.0 Changes & Improvements

### 🆕 New Agents Added

| Agent | What It Does |
|-------|-------------|
| **Auth & Access Agent (00)** | Secures all agent interactions with RBAC. Farmer, vet, staff, and admin roles each get scoped tokens. Critical for the Automation Agent which can perform real actions. |
| **Interaction Agent (03)** | Split from v1's over-broad User Agent. Handles all ongoing mid-use farmer and vet input — and critically, routes correction signals back to the ML models. |
| **Notification Delivery Agent (12)** | Alert creation and delivery are now separate. This agent handles all outbound channels — push, WhatsApp, SMS, email — independently of alert logic. |
| **Staff Assignment Agent (13)** | Auto-assigns actionable tasks to vets and staff when critical alerts fire. Includes escalation timers and acknowledgement tracking — closing the alert-to-action loop. |

### 🔄 Modified Agents

| Agent | What Changed |
|-------|-------------|
| **Hardware Agent (01)** | Edge buffer added — stores up to 72h of data locally when offline. Syncs with timestamp correction on reconnect. |
| **User Agent (02) → Onboarding Agent** | Split into Onboarding Agent (setup) and Interaction Agent (ongoing). Cleaner scope, better separation of cadences. |
| **Model Agent (06)** | Feedback loop added. Vet correction signals from Interaction Agent accumulate as labelled samples and trigger periodic fine-tuning. |
| **Update Agent (10) → Sync Agent** | Renamed and redesigned to be write-event-triggered only. No more continuous polling — dramatically more efficient. |
| **Orchestrator Agent (07)** | Fallback pub/sub routing added. If the Orchestrator fails, domain agents subscribe directly to Model Agent outputs — no single point of failure. |

---

## 4. Data Flow & Inter-Agent Communication

Data flows through the system in three primary streams:

### 🔵 Telemetry Pipeline
```
Hardware Agent → Data Management Agent → Data Processing Agent → Model Agent → Orchestrator → Domain Agents
```

1. Hardware Agent collects raw sensor data from IoT devices (edge-buffered for resilience)
2. Data Management Agent merges telemetry with animal profiles via hardware ID matching
3. Data Processing Agent cleans, windows, and engineers features for each ML task
4. Model Agent runs inference, generating 7-day forecasts across 4 models
5. Orchestrator routes prediction outputs to Health Agent, Location Agent, and Sync Agent

---

### 🟣 Feedback Pipeline
```
Interaction Agent → Data Processing Agent → Model Agent → Fine-Tuned Models (hot-swap)
```

1. Farmer or vet provides a manual observation or override via the Interaction Agent
2. Interaction Agent formats the correction as a labelled sample *(e.g., status='Critical', temp=39.5)*
3. Data Processing Agent packages the sample into the model's training format
4. Model Agent accumulates correction samples and triggers fine-tuning on a rolling schedule
5. Improved models are hot-swapped in — no downtime required

---

### 🔴 Alert & Action Stream
```
Health/Location Agent → Alert Agent → Notification Agent + Staff Assignment Agent → Automation Agent
```

1. Health Agent or Location Agent detects a threshold breach and emits an alert event
2. Alert Agent classifies, deduplicates, and prioritizes the alert by severity
3. Notification Delivery Agent sends the alert via the appropriate channel (push, WhatsApp, email)
4. Staff Assignment Agent creates and assigns a task to the relevant vet or staff member
5. Automation Agent can query any part of this stream to answer farmer questions in real time

---

## 5. UI Pages & Agent Mapping

| UI Page | Primary Agent | Supporting Agents |
|---------|---------------|-------------------|
| Overview Dashboard | Orchestrator | Health Agent, Location Agent, Alert Agent, Sync Agent |
| Animals | Sync Agent | Data Management Agent, Health Agent |
| Map Intelligence | Location Agent | Orchestrator, Data Management Agent |
| Health Analytics | Health Agent | Model Agent (Health Forecast), Orchestrator |
| Alerts Centre | Alert Agent | Health Agent, Location Agent, Notification Delivery Agent |
| Predictions | Model Agent | Data Processing Agent, Orchestrator |
| AI Orchestrator | Orchestrator | All agents — dashboard of agent activity |
| Behaviour Analysis | Model Agent (Movement) | Health Agent, Orchestrator |
| Disease Risk | Model Agent (Disease) | Health Agent, Orchestrator |
| Devices | Sync Agent | Hardware Agent, Data Management Agent |
| Farms & Locations | Location Agent | Sync Agent, Data Management Agent |
| Reports | Automation Agent | All domain agents provide data |
| Staff | Staff Assignment Agent | Sync Agent, Auth Agent |
| Profile | Auth Agent | Onboarding Agent, Sync Agent |

---

## 6. Recommended Next Steps

### 1. Define the Message Bus
Consider Apache Kafka or a lightweight **Redis pub/sub** as the backbone for agent-to-agent communication. This enables the Orchestrator fallback and makes the system horizontally scalable.

### 2. Establish Agent API Contracts
Each agent should expose a typed interface — input schema, output schema, and error codes. This makes testing, mocking, and replacing individual agents straightforward.

### 3. Build the Model Agent Incrementally
Start with the **Health Forecasting model** (highest immediate value), validate against real farm data, then add Movement, Device, and Disease models sequentially.

### 4. Prototype the Feedback Loop Early
Even a simple manual-label collection mechanism from day one will give you enough correction data to meaningfully fine-tune by month 3.

### 5. WhatsApp Integration is High Priority
For Indian farmers, **WhatsApp is the primary communication channel**. The Notification Delivery Agent should prioritize this integration above email and push. Use the WhatsApp Business API (Meta) or Twilio's WhatsApp sandbox.

---

*GoMata Intelligence System · Architecture v2.0 · 15 Agents · Confidential · 2026*