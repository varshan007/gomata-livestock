#!/usr/bin/env node
/**
 * GoMata Digital Twin v2 — Data Flow Verification Script
 * 
 * Checks that telemetry is flowing correctly through:
 *   1. MongoDB (DeviceTelemetry, TrainingEvent collections)
 *   2. Redis (Feature Store, Risk Scores)
 *   3. BullMQ (ML Prediction queue)
 * 
 * Usage:
 *   node scripts/verifyDataFlow.js
 */

'use strict';

const path = require('path');
const envFile = process.env.NODE_ENV === 'production'
    ? '.env.production' : '.env.development';
require('dotenv').config({ path: path.join(__dirname, '../', envFile) });

const mongoose = require('mongoose');
const Redis = require('ioredis');

// ── Color helpers ───────────────────────────────────────────────────────────
const GREEN = '\x1b[32m';
const RED = '\x1b[31m';
const YELLOW = '\x1b[33m';
const CYAN = '\x1b[36m';
const BOLD = '\x1b[1m';
const RESET = '\x1b[0m';

const pass = (msg) => console.log(`  ${GREEN}✅ ${msg}${RESET}`);
const fail = (msg) => console.log(`  ${RED}❌ ${msg}${RESET}`);
const warn = (msg) => console.log(`  ${YELLOW}⚠️  ${msg}${RESET}`);
const info = (msg) => console.log(`  ${CYAN}ℹ  ${msg}${RESET}`);
const header = (msg) => console.log(`\n${BOLD}${CYAN}═══ ${msg} ═══${RESET}`);

// ═════════════════════════════════════════════════════════════════════════════

async function main() {
    console.log(`
${BOLD}${CYAN}🧬 GoMata Digital Twin v2 — Data Flow Verification${RESET}
${CYAN}═══════════════════════════════════════════════════${RESET}
`);

    let redis = null;
    let mongoConnected = false;

    try {
        // ── 1. MONGODB ──────────────────────────────────────────────
        header('1. MongoDB Connection & Collections');

        const mongoUri = process.env.MONGO_URI || 'mongodb://localhost:27017/livestock';
        info(`Connecting to: ${mongoUri.replace(/\/\/.*@/, '//<hidden>@')}`);

        await mongoose.connect(mongoUri);
        mongoConnected = true;
        pass('MongoDB connected');

        // Check DeviceTelemetry
        const DeviceTelemetry = require('../models/DeviceTelemetry');
        const telemetryCount = await DeviceTelemetry.countDocuments();
        const recentTelemetry = await DeviceTelemetry.find()
            .sort({ timestamp: -1 })
            .limit(5)
            .lean();

        if (telemetryCount > 0) {
            pass(`DeviceTelemetry: ${telemetryCount} documents`);

            if (recentTelemetry.length > 0) {
                const latest = recentTelemetry[0];
                const age = (Date.now() - new Date(latest.timestamp).getTime()) / 1000;

                console.log(`\n  ${BOLD}Latest telemetry (${age.toFixed(0)}s ago):${RESET}`);
                console.log(`    Device:      ${latest.deviceId}`);
                console.log(`    Animal:      ${latest.animalId}`);
                console.log(`    Tenant:      ${latest.tenantId}`);
                console.log(`    Temperature: ${latest.temperature}°C`);
                console.log(`    Heart Rate:  ${latest.heartRate} bpm`);
                console.log(`    Respiration: ${latest.respiration || 'N/A'} bpm`);
                console.log(`    Activity:    ${latest.activity}`);
                console.log(`    Rumination:  ${latest.rumination || 'N/A'} min`);
                console.log(`    THI:         ${latest.thi || 'N/A'}`);
                console.log(`    Location:    [${latest.location?.coordinates?.join(', ') || 'N/A'}]`);

                // Check if v2 fields are present
                if (latest.respiration !== undefined || latest.thi !== undefined) {
                    pass('Digital Twin v2 fields detected (respiration, THI)');
                } else {
                    info('V1 telemetry format (no respiration/THI). Switch to SIMULATION_VERSION=v2');
                }

                if (age < 60) {
                    pass(`Fresh data — latest entry is ${age.toFixed(0)}s old`);
                } else if (age < 300) {
                    warn(`Data is ${(age / 60).toFixed(1)} minutes old`);
                } else {
                    warn(`Data is ${(age / 3600).toFixed(1)} hours old — is simulator running?`);
                }
            }
        } else {
            fail('DeviceTelemetry: 0 documents — no telemetry in DB');
        }

        // Check TrainingEvent
        const TrainingEvent = require('../models/TrainingEvent');
        const trainingCount = await TrainingEvent.countDocuments();
        const recentTraining = await TrainingEvent.find()
            .sort({ createdAt: -1 })
            .limit(3)
            .lean();

        if (trainingCount > 0) {
            pass(`TrainingEvent: ${trainingCount} documents`);

            if (recentTraining.length > 0) {
                const latest = recentTraining[0];
                console.log(`\n  ${BOLD}Latest training event:${RESET}`);
                console.log(`    Source:   ${latest.source}`);
                console.log(`    Type:     ${latest.eventType}`);
                console.log(`    Label:    ${latest.label}`);

                if (latest.hiddenLabels) {
                    pass('Hidden labels present (Digital Twin v2 training events)');
                    console.log(`    Infection:  ${latest.hiddenLabels.infectionLoad}`);
                    console.log(`    Severity:   ${latest.hiddenLabels.severityLevel}`);
                    console.log(`    Phase:      ${latest.hiddenLabels.episodePhase}`);
                }
            }
        } else {
            warn('TrainingEvent: 0 documents (training events are sampled at 8%)');
        }

        // Check LivestockMaster
        const LivestockMaster = require('../models/LivestockMaster');
        const livestockCount = await LivestockMaster.countDocuments();
        pass(`LivestockMaster: ${livestockCount} animals registered`);

        // Check Alerts
        const Alert = require('../models/Alert');
        const alertCount = await Alert.countDocuments();
        const recentAlerts = await Alert.find()
            .sort({ createdAt: -1 })
            .limit(3)
            .lean();

        if (alertCount > 0) {
            pass(`Alerts: ${alertCount} total`);
            if (recentAlerts.length > 0) {
                const latest = recentAlerts[0];
                const age = (Date.now() - new Date(latest.createdAt).getTime()) / 1000;
                console.log(`    Latest: "${latest.type}" for ${latest.animalName || 'Unknown'} (${(age / 60).toFixed(0)} min ago)`);
            }
        } else {
            info('Alerts: 0 — HealthAgent may not have triggered any yet');
        }

        // ── 2. REDIS ────────────────────────────────────────────────
        header('2. Redis Connection & Feature Store');

        const redisUrl = process.env.REDIS_URL || 'redis://localhost:6379';
        info(`Connecting to: ${redisUrl}`);

        redis = new Redis(redisUrl);
        await redis.ping();
        pass('Redis connected (PONG received)');

        // Check feature store keys
        const featureKeys = await redis.keys('features:v3:*');
        if (featureKeys.length > 0) {
            pass(`Feature Store: ${featureKeys.length} cached feature sets`);

            // Show a sample
            const sampleKey = featureKeys[0];
            const sampleData = await redis.get(sampleKey);
            if (sampleData) {
                const features = JSON.parse(sampleData);
                const parts = sampleKey.split(':');
                console.log(`\n  ${BOLD}Sample feature set (${parts[2]}/${parts[3]}):${RESET}`);
                console.log(`    temp_current:  ${features.temp_current}°C`);
                console.log(`    hr_current:    ${features.hr_current} bpm`);
                console.log(`    temp_6h_avg:   ${features.temp_6h_avg}`);
                console.log(`    temp_zscore:   ${features.temp_zscore}`);
                console.log(`    version:       ${features.feature_version}`);

                const ttl = await redis.ttl(sampleKey);
                console.log(`    TTL:           ${ttl}s`);
            }
        } else {
            warn('Feature Store: 0 keys — FeatureStoreWorker may not have run yet');
        }

        // Check risk score keys
        const riskKeys = await redis.keys('healthRisk:*');
        const riskKeys2 = await redis.keys('risk:*');
        const totalRiskKeys = riskKeys.length + riskKeys2.length;

        if (totalRiskKeys > 0) {
            pass(`Risk Scores: ${totalRiskKeys} cached (healthRisk:${riskKeys.length}, risk:${riskKeys2.length})`);
        } else {
            info('Risk Scores: 0 cached — HealthAgent may not have run predictions yet');
        }

        // Check LLM explanation cache
        const llmKeys = await redis.keys('llm:explanation:*');
        if (llmKeys.length > 0) {
            pass(`LLM Explanations: ${llmKeys.length} cached`);
        } else {
            info('LLM Explanations: 0 cached');
        }

        // Overall Redis key count
        const allKeysCount = await redis.dbsize();
        info(`Total Redis keys: ${allKeysCount}`);

        // ── 3. BULLMQ ───────────────────────────────────────────────
        header('3. BullMQ Job Queues');

        // BullMQ stores data in Redis with prefix 'bull:'
        const bullKeys = await redis.keys('bull:*');

        if (bullKeys.length > 0) {
            pass(`BullMQ: ${bullKeys.length} Redis keys`);

            // Check ML Predictions queue
            const mlWaiting = await redis.llen('bull:ml-predictions:wait') || 0;
            const mlActive = await redis.llen('bull:ml-predictions:active') || 0;
            const mlCompleted = await redis.get('bull:ml-predictions:id') || '0';

            console.log(`\n  ${BOLD}ML Predictions Queue:${RESET}`);
            console.log(`    Waiting:    ${mlWaiting}`);
            console.log(`    Active:     ${mlActive}`);
            console.log(`    Total Jobs: ~${mlCompleted}`);

            if (parseInt(mlCompleted) > 0) {
                pass('ML prediction jobs have been processed');
            } else if (mlWaiting > 0 || mlActive > 0) {
                warn('Jobs in queue but none completed — is ML worker running?');
            }
        } else {
            warn('BullMQ: No queue keys found — queues may not be initialized');
        }

        // ── 4. DATA FLOW SUMMARY ────────────────────────────────────
        header('4. Data Flow Summary');

        const flowStatus = {
            'Simulator → MongoDB': telemetryCount > 0,
            'MongoDB → FeatureStore (Redis)': featureKeys.length > 0,
            'FeatureStore → HealthAgent → Risk (Redis)': totalRiskKeys > 0,
            'HealthAgent → Alerts (MongoDB)': alertCount > 0,
            'Simulator → BullMQ': bullKeys.length > 0,
        };

        console.log('');
        console.log(`  ${BOLD}Pipeline Stage                              Status${RESET}`);
        console.log(`  ${'─'.repeat(56)}`);

        for (const [stage, ok] of Object.entries(flowStatus)) {
            const icon = ok ? `${GREEN}✅` : `${YELLOW}⏳`;
            const status = ok ? 'ACTIVE' : 'PENDING';
            console.log(`  ${icon} ${stage.padEnd(44)} ${status}${RESET}`);
        }

        console.log(`\n  ${BOLD}Simulation Version:${RESET} ${process.env.SIMULATION_VERSION || 'v1 (default)'}`);

        // ── 5. QUICK HEALTH CHECK ───────────────────────────────────
        header('5. Recommendations');

        if (process.env.SIMULATION_VERSION !== 'v2') {
            info('To use Digital Twin v2: set SIMULATION_VERSION=v2 in your .env or start command');
        }
        if (featureKeys.length === 0) {
            info('Wait 60s for FeatureStoreWorker to cache first features');
        }
        if (totalRiskKeys === 0 && featureKeys.length > 0) {
            info('Wait ~70s for HealthSchedulerAgent to run first prediction cycle');
        }
        if (alertCount === 0 && totalRiskKeys > 0) {
            info('Alerts trigger when disease_prob >= threshold — may take a few minutes');
        }

        console.log(`\n${GREEN}${BOLD}✅ Verification complete${RESET}\n`);

    } catch (err) {
        console.error(`\n${RED}${BOLD}Error: ${err.message}${RESET}`);
        console.error(err.stack);
    } finally {
        if (redis) await redis.quit();
        if (mongoConnected) await mongoose.disconnect();
    }
}

main().catch(console.error);
