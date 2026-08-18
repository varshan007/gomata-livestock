#!/usr/bin/env node
/**
 * GoMata Digital Twin — Batch Dataset Generator
 * 
 * Generates ML training datasets using the Hidden-State Digital Twin Simulator.
 * 
 * Usage:
 *   node scripts/generateDataset.js --cows 100 --days 30
 *   node scripts/generateDataset.js --cows 1000 --days 180 --output ./data/v2_full
 * 
 * Outputs:
 *   telemetry.csv           — Raw sensor readings + hidden labels
 *   features_6h.csv         — Engineered features for XGBoost
 *   herd_metrics.csv        — Herd-level epidemic curves
 *   intervention_outcomes.csv — Strategy comparison (5000+ records)
 */

'use strict';

const path = require('path');
const envFile = process.env.NODE_ENV === 'production'
    ? '.env.production' : '.env.development';
require('dotenv').config({ path: path.join(__dirname, '../', envFile) });

const DatasetExporter = require('../services/digitalTwin/DatasetExporter');

// ── Parse CLI args ──────────────────────────────────────────────────────────

function parseArgs() {
    const args = {};
    const argv = process.argv.slice(2);
    for (let i = 0; i < argv.length; i++) {
        if (argv[i] === '--cows') args.numCows = parseInt(argv[++i]);
        else if (argv[i] === '--days') args.days = parseInt(argv[++i]);
        else if (argv[i] === '--interval') args.samplingMinutes = parseInt(argv[++i]);
        else if (argv[i] === '--output') args.outputDir = argv[++i];
        else if (argv[i] === '--tenant') args.tenantId = argv[++i];
        else if (argv[i] === '--farm') args.farmId = argv[++i];
        else if (argv[i] === '--help') {
            console.log(`
GoMata Digital Twin — Batch Dataset Generator

Usage:
  node scripts/generateDataset.js [OPTIONS]

Options:
  --cows <N>         Number of cows to simulate (default: 100)
  --days <N>         Simulation duration in days (default: 30)
  --interval <M>     Sampling interval in minutes (default: 5)
  --output <DIR>     Output directory (default: ./data/exports)
  --tenant <ID>      Tenant ID (default: batch_export)
  --farm <ID>        Farm ID (default: FM-BATCH)
  --help             Show this help message

Examples:
  node scripts/generateDataset.js --cows 100 --days 30
  node scripts/generateDataset.js --cows 1000 --days 180 --output ./data/v2_full
`);
            process.exit(0);
        }
    }
    return args;
}

// ── Main ────────────────────────────────────────────────────────────────────

async function main() {
    const args = parseArgs();

    const config = {
        numCows: args.numCows || 100,
        days: args.days || 30,
        samplingMinutes: args.samplingMinutes || 5,
        outputDir: args.outputDir || path.join(__dirname, '../data/exports'),
        tenantId: args.tenantId || 'batch_export',
        farmId: args.farmId || 'FM-BATCH'
    };

    const totalRows = config.numCows * Math.floor((config.days * 24 * 60) / config.samplingMinutes);

    console.log(`
🧬 GoMata Digital Twin — Batch Dataset Generator
═════════════════════════════════════════════════
  Cows:       ${config.numCows}
  Days:       ${config.days}
  Interval:   ${config.samplingMinutes} min
  Total rows: ${(totalRows / 1e6).toFixed(1)}M
  Output:     ${config.outputDir}
═════════════════════════════════════════════════
`);

    const exporter = new DatasetExporter(config);
    const files = await exporter.generateAll();

    console.log(`
✅ Dataset generation complete!

  📊 Telemetry:     ${files.telemetryFile}
  📈 Features:      ${files.featureFile}
  🐄 Herd Metrics:  ${files.herdFile}
  💉 Interventions: ${files.interventionFile}
`);
}

main().catch(err => {
    console.error('Fatal error:', err);
    process.exit(1);
});
