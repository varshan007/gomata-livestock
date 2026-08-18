const express = require('express');
const mongoose = require('mongoose');
const cors = require('cors');
const path = require('path');
// Load environment-specific variables
const envFile = process.env.NODE_ENV === 'production'
  ? '.env.production'
  : process.env.NODE_ENV === 'staging'
    ? '.env.staging'
    : process.env.NODE_ENV === 'accelerated'
      ? '.env.accelerated'
      : '.env.development';

require('dotenv').config({ path: path.join(__dirname, envFile) });

// Initialize Logger and Services
const logger = require('./utils/logger');
const redisConnection = require('./config/redis');
const mlServiceClient = require('./services/mlServiceClient');

const http = require('http');
const { Server } = require('socket.io');

// Import Agent Event Bus
const eventBus = require('./src/bus/RedisEventBus');

// Import Agents (Phase 1)
const AuthAgent = require('./src/agents/auth/AuthAgent');
const OnboardingAgent = require('./src/agents/onboarding/OnboardingAgent');
const HardwareAgent = require('./src/agents/hardware/HardwareAgent');
const SyncAgent = require('./src/agents/sync/SyncAgent');

// --- Phase 2: Intelligence Agents ---
const DataProcessingAgent = require('./src/agents/processing/DataProcessingAgent');
const HealthAgent = require('./src/agents/health/HealthAgent');
const LocationAgent = require('./src/agents/location/LocationAgent');
const AlertAgent = require('./src/agents/alerts/AlertAgent');
const NotificationDeliveryAgent = require('./src/agents/notifications/NotificationDeliveryAgent');
const StaffAssignmentAgent = require('./src/agents/staff/StaffAssignmentAgent');
const LLMExplanationAgent = require('./src/agents/health/LLMExplanationAgent');

logger.info("🚀 SERVER RESTARTED - MULTI-AGENT PHASE 1 ONLINE 🚀");

const app = express();
const server = http.createServer(app);

// Setup Socket.io for SyncAgent
const io = new Server(server, {
  cors: { origin: '*', methods: ['GET', 'POST'] }
});

// Middleware
app.use(cors({
  origin: '*', // Allow all origins for development/ngrok
  methods: ['GET', 'POST', 'PUT', 'DELETE', 'PATCH', 'OPTIONS'],
  allowedHeaders: ['Content-Type', 'Authorization']
}));
app.use(express.json());
app.use('/uploads', express.static('uploads')); // Serve uploaded images

// --- PRODUCTION: HEALTH CHECK & KEEP-WARM ---
app.get('/health', (req, res) => res.status(200).send('OK'));

// Asynchronously ping ML Service on boot to wake it up
setTimeout(() => {
    const mlUrl = process.env.ML_SERVICE_URL || 'http://localhost:8001';
    const axios = require('axios');
    axios.get(`${mlUrl}/health`).then(() => {
        logger.info({ action: 'keep_warm', service: 'ml_service' }, 'Successfully pinged ML Service /health on boot');
    }).catch(err => {
        logger.warn({ action: 'keep_warm', service: 'ml_service', error: err.message }, 'Failed to ping ML Service on boot');
    });
}, 3000);
// --------------------------------------------

// Import Error Handlers
const { errorHandler, notFoundHandler } = require('./middleware/errorHandler');

// Import Request Logger
const requestLogger = require('./middleware/requestLogger');
app.use(requestLogger);

// MongoDB Connection
if (!process.env.MONGO_URI) {
  logger.error('❌ FATAL ERROR: MONGO_URI is not defined.');
  process.exit(1);
}

mongoose.connect(process.env.MONGO_URI, {
  useNewUrlParser: true,
  useUnifiedTopology: true
})
  .then(async () => {
    const logger = require('./utils/logger');
    logger.info({ action: 'db_connect', result: 'success', dbName: mongoose.connection.name, host: mongoose.connection.host }, 'MongoDB Connected');

    // Import Data Models for Agents
    const Livestock = require('./models/Livestock');
    const User = require('./models/User');
    const Geofence = require('./models/Geofence');
    const models = { Livestock, User, Geofence };

    // ── AgentOrchestrator Setup ─────────────────────────────────────────
    const AgentOrchestrator = require('./src/core/AgentOrchestrator');
    const orchestrator = new AgentOrchestrator(redisConnection);
    global.orchestrator = orchestrator; // accessible from route handlers

    // --- Instantiate Phase 1 Agents ---
    const authAgent = new AuthAgent(eventBus);
    const onboardingAgent = new OnboardingAgent(eventBus, models);
    const hardwareAgent = new HardwareAgent(eventBus, process.env.MQTT_BROKER_URL, models);
    const syncAgent = new SyncAgent(eventBus, io);

    // --- Instantiate Phase 2 Intelligence Agents ---
    const dataProcessingAgent = new DataProcessingAgent(eventBus);
    const healthAgent = new HealthAgent({
      redis: redisConnection,
      eventBus: eventBus,
      mlServiceUrl: process.env.ML_SERVICE_URL || 'http://localhost:8001',
      operatingMode: process.env.HEALTH_AGENT_MODE || 'default',
      concurrency: parseInt(process.env.HEALTH_AGENT_CONCURRENCY || '15', 10)
    });

    // HealthSchedulerAgent — wraps HealthAgent for multi-tenant batch execution
    const HealthSchedulerAgent = require('./src/agents/health/healthSchedulerAgent');
    const healthScheduler = new HealthSchedulerAgent({
      healthAgent,
      tenantBatch: parseInt(process.env.HEALTH_TENANT_BATCH || '5', 10)
    });

    // LLMExplanationAgent — subscribes to alert:saved, generates LLM explanations
    const llmAgent = new LLMExplanationAgent({
      redis: redisConnection,
      eventBus: eventBus
    });
    llmAgent.subscribeToAlerts(eventBus);

    const locationAgent = new LocationAgent(eventBus);
    const alertAgent = new AlertAgent(eventBus);
    const notificationAgent = new NotificationDeliveryAgent(eventBus);
    const staffAssignmentAgent = new StaffAssignmentAgent(eventBus);

    // Simulation Worker (Bypasses MQTT, injects directly to Event Bus)
    const SimulationWorker = require('./src/workers/SimulationWorker');
    const simulationWorker = new SimulationWorker(eventBus);

    // ── Register Phase 1 agents (event-driven, no cron) ──────────────
    orchestrator.register('AuthAgent', async () => authAgent.start(), {
      timeoutMs: 5000, maxRetries: 1, distributed: false
    });
    orchestrator.register('OnboardingAgent', async () => onboardingAgent.start(), {
      timeoutMs: 5000, maxRetries: 1, distributed: false
    });
    orchestrator.register('HardwareAgent', async () => hardwareAgent.start(), {
      timeoutMs: 10000, maxRetries: 2, distributed: false
    });
    orchestrator.register('SyncAgent', async () => syncAgent.start(), {
      timeoutMs: 5000, maxRetries: 1, distributed: false
    });

    // ── Register Phase 2 intelligence agents ─────────────────────────
    orchestrator.register('DataProcessingAgent', async () => dataProcessingAgent.start(), {
      timeoutMs: 10000, maxRetries: 2, distributed: false
    });
    // HealthSchedulerAgent — multi-tenant batch scan, cron-scheduled
    orchestrator.register('HealthSchedulerAgent', async () => healthScheduler.run(), {
      timeoutMs: 300000,   // 5 min timeout (processes all tenants × all animals)
      maxRetries: 1,
      distributed: true    // only one instance runs the full scan
    });
    orchestrator.register('LocationAgent', async () => locationAgent.start(), {
      timeoutMs: 10000, maxRetries: 2, distributed: false
    });
    orchestrator.register('AlertAgent', async () => alertAgent.start(), {
      timeoutMs: 10000, maxRetries: 2, distributed: false
    });
    orchestrator.register('NotificationAgent', async () => notificationAgent.start(), {
      timeoutMs: 10000, maxRetries: 2, distributed: false
    });
    orchestrator.register('StaffAssignmentAgent', async () => staffAssignmentAgent.start(), {
      timeoutMs: 10000, maxRetries: 2, distributed: false
    });

    // ── Start all agents through orchestrator ────────────────────────
    await orchestrator.execute('AuthAgent');
    await orchestrator.execute('OnboardingAgent');
    await orchestrator.execute('HardwareAgent');
    await orchestrator.execute('SyncAgent');
    await orchestrator.execute('DataProcessingAgent');
    
    if (process.env.NODE_ENV === 'production') {
        simulationWorker.start();
    }
    // HealthSchedulerAgent — every 60s interval for testing (change to cron for prod)
    orchestrator.scheduleInterval('HealthSchedulerAgent', 60000, null, false);

    // Optional: startup test run (controlled by env var)
    if (process.env.RUN_STARTUP_TEST === 'true') {
      setTimeout(() => {
        logger.info({ service: 'health_scheduler', action: 'STARTUP_TEST' },
          '[HealthSchedulerAgent] Startup test triggered');
        orchestrator.execute('HealthSchedulerAgent').catch(err => {
          logger.error({ service: 'health_scheduler', error: err.message },
            '[HealthSchedulerAgent] Startup test failed');
        });
      }, 70000); // 70s delay — wait for simulation ticks + feature caching
    }
    await orchestrator.execute('LocationAgent');
    await orchestrator.execute('AlertAgent');
    await orchestrator.execute('NotificationAgent');
    await orchestrator.execute('StaffAssignmentAgent');

    // ── Register & schedule Feature Store (cron-style, distributed) ──
    const featureStoreWorker = require('./workers/featureStoreWorker');
    orchestrator.register('FeatureStoreWorker', async () => {
      await featureStoreWorker.refreshAllFeatures();
    }, {
      timeoutMs: 60000,    // 60s timeout for full refresh
      maxRetries: 2,
      distributed: true    // only one instance runs the refresh
    });
    orchestrator.scheduleInterval('FeatureStoreWorker', 60000, null, true);

    // Start Real-Time Hardware Simulation Engine
    const simVersion = process.env.SIMULATION_VERSION || 'v1';
    if (simVersion === 'v2') {
      const DigitalTwinSimulator = require('./services/digitalTwin/DigitalTwinSimulator');
      const twinSim = new DigitalTwinSimulator();
      twinSim.start();
      logger.info({ action: 'SIMULATION_VERSION', version: 'v2' }, '🧬 Digital Twin Simulator v2 activated');
    } else {
      require('./services/HardwareSimulationService').start();
      logger.info({ action: 'SIMULATION_VERSION', version: 'v1' }, '📡 Legacy Hardware Simulation v1 activated');
    }

    logger.info({ action: 'agent_init', result: 'success' }, 'All Phase 1 & Phase 2 AI Agents Initialized via AgentOrchestrator.');
  })
  .catch(err => {
    const logger = require('./utils/logger');
    logger.error({ action: 'db_connect', result: 'error', err }, 'MongoDB Connection Error');
  });

// Import Routes
const livestockRouter = require('./routes/livestock');
const sensorDataRouter = require('./routes/sensorData');
const alertsRouter = require('./routes/alerts');
const geofencesRouter = require('./routes/geofences');
const authRoutes = require('./routes/auth');
const systemRoutes = require('./routes/system');
const farmsRouter = require('./routes/farms');
const devicesRouter = require('./routes/devices');
const staffRouter = require('./routes/staff');
const staffAuthRouter = require('./routes/staffAuth');

// Use Routes
app.use('/api/livestock', livestockRouter);
app.use('/api/dashboard', require('./routes/dashboard'));
app.use('/api/sensor-data', sensorDataRouter);
app.use('/api/alerts', alertsRouter);
app.use('/api/geofences', geofencesRouter);
app.use('/api/auth', authRoutes);
app.use('/api/system', systemRoutes);
app.use('/api/ai', require('./routes/aiRoutes')); // Voice Assistant Route
app.use('/api/farms', farmsRouter);
app.use('/api/devices', devicesRouter);
app.use('/api/staff', staffRouter);
app.use('/api/staffAuth', staffAuthRouter);
app.use('/api/health-prediction', require('./routes/healthPrediction'));

// Health Check Endpoint — includes per-agent health from orchestrator
app.get('/api/health', async (req, res) => {
  try {
    const redisStatus = redisConnection.status === 'ready' ? 'connected' : 'disconnected';

    let mlStatus = 'disabled';
    if (process.env.ML_SERVICE_URL) {
      try {
        const mlRes = await mlServiceClient.getHealth();
        mlStatus = mlRes.status || 'connected';
      } catch (err) {
        mlStatus = 'disconnected';
      }
    }

    // Include orchestrator agent health if available
    const agentHealth = global.orchestrator
      ? global.orchestrator.getHealthStatus()
      : null;

    res.status(200).json({
      success: true,
      data: {
        status: 'ok',
        database: mongoose.connection.readyState === 1 ? 'connected' : 'disconnected',
        redis: redisStatus,
        mlService: mlStatus,
        orchestrator: agentHealth,
        timestamp: new Date().toISOString()
      },
      meta: { version: '2.0' }
    });
  } catch (error) {
    res.status(500).json({
      success: false,
      error: { message: 'Health check failed', details: error.message }
    });
  }
});

// Provide a basic response for root/non-API requests instead of serving frontend files
app.use((req, res, next) => {
  if (req.path.startsWith('/api/')) {
    return next();
  }
  res.status(200).json({ message: "GoMata API is running. Frontend is hosted separately." });
});

// Setup Error Handling Middleware
app.use(notFoundHandler);
app.use(errorHandler);

// Start server
const PORT = process.env.PORT || 8000;
server.listen(PORT, () => {
  const logger = require('./utils/logger');
  logger.info({ action: 'server_start', port: PORT }, `Server running on port ${PORT}`);
});

// ── Graceful Shutdown ─────────────────────────────────────────────────────
async function gracefulShutdown(signal) {
  const logger = require('./utils/logger');
  logger.info({ action: 'shutdown_signal', signal }, `Received ${signal} — initiating graceful shutdown`);

  // 1. Shutdown orchestrator (stops crons, closes workers, releases locks)
  if (global.orchestrator) {
    await global.orchestrator.shutdown();
  }

  // 2. Stop feature store worker
  try {
    require('./workers/featureStoreWorker').stop();
  } catch (e) { /* already stopped */ }

  // 3. Close HTTP server
  server.close(() => {
    logger.info({ action: 'server_stopped' }, 'HTTP server closed');

    // 4. Close MongoDB
    mongoose.connection.close(false).then(() => {
      logger.info({ action: 'db_disconnected' }, 'MongoDB disconnected');
      process.exit(0);
    });
  });

  // Force exit after 10 seconds if graceful shutdown hangs
  setTimeout(() => {
    logger.error({ action: 'forced_exit' }, 'Forced exit after 10s timeout');
    process.exit(1);
  }, 10000);
}

process.on('SIGTERM', () => gracefulShutdown('SIGTERM'));
process.on('SIGINT', () => gracefulShutdown('SIGINT'));

