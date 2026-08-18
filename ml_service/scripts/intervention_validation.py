#!/usr/bin/env python3
"""
intervention_validation.py — GoMata Pilot Validation Framework v1, Part 3
Intervention Simulation Validation

Validates that treatment timing changes biological + economic outcomes:
  A) Antibiotic at onset (severity=1)  → faster recovery, lower milk loss
  B) Antibiotic at peak (severity=3)   → longer duration, higher milk loss

Expected: milk loss difference ≥20% between early and late treatment.

Usage:
  python intervention_validation.py [--cows 50] [--runs 5]
"""

import os
import sys
import json
import logging
import subprocess
import numpy as np
import pandas as pd

# ── Logging ───────────────────────────────────────────────────────────────────
log_dir = os.path.join(os.path.dirname(__file__), "../logs")
os.makedirs(log_dir, exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(os.path.join(log_dir, "intervention_validation.log")),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger("InterventionValidation")

BACKEND_DIR = os.path.join(os.path.dirname(__file__), "../../backend")


# ═════════════════════════════════════════════════════════════════════════════════
# SIMULATION SCENARIOS
# ═════════════════════════════════════════════════════════════════════════════════

def run_intervention_scenario(trigger_severity, num_cows=50):
    """
    Run intervention timing scenario via Node.js subprocess.
    
    trigger_severity: 1 = onset, 3 = peak
    Returns per-cow episode metrics.
    """
    script = f"""
    const CowPhysiologyEngine = require('./services/digitalTwin/CowPhysiologyEngine');
    const EnvironmentModel = require('./services/digitalTwin/EnvironmentModel');
    const EpisodeScheduler = require('./services/digitalTwin/EpisodeScheduler');
    const ProductionModel = require('./services/digitalTwin/ProductionModel');
    const FarmProfile = require('./services/digitalTwin/FarmProfile');

    const TICK_MINUTES = 5;
    const TICKS_PER_DAY = 288;
    const SIM_DAYS = 60;
    const TOTAL_TICKS = SIM_DAYS * TICKS_PER_DAY;
    const NUM_COWS = {num_cows};
    const TRIGGER_SEV = {trigger_severity};

    const env = new EnvironmentModel();
    const farmProfile = FarmProfile.get('dairy');
    const scheduler = new EpisodeScheduler({{
        totalTicks: TOTAL_TICKS, tickMinutes: TICK_MINUTES,
        numCows: NUM_COWS, farmProfile
    }});

    const breeds = ['Holstein', 'Jersey', 'Gir', 'Sahiwal'];
    const cowIds = Array.from({{length: NUM_COWS}}, (_, i) => 'cow_' + String(i).padStart(4, '0'));
    const engines = new Map();
    const prodModels = new Map();
    const cowState = new Map();  // Track per-cow episode data

    for (const id of cowIds) {{
        const breed = breeds[Math.floor(Math.random() * breeds.length)];
        const baseMilk = breed === 'Holstein' ? 35 : breed === 'Jersey' ? 25 : 15;
        engines.set(id, new CowPhysiologyEngine(id, {{ age: 4 }}));
        prodModels.set(id, new ProductionModel({{
            breed, lactationStage: 'mid', parity: 2,
            baselineMilkYield: baseMilk, baselineWeight: 500,
            calvingDate: new Date(Date.now() - 90 * 86400000)
        }}));
        cowState.set(id, {{
            episodes: [],
            currentEpisode: null,
            abxActive: false,
            abxStartTick: 0,
            baselineMilk: baseMilk
        }});
    }}

    const schedule = scheduler.generateSchedule(cowIds);

    for (let tick = 1; tick <= TOTAL_TICKS; tick++) {{
        const envSnap = env.getEnvironment(tick, TICK_MINUTES, NUM_COWS);
        const baseStress = env.computeStressLoad(envSnap);
        const currentDay = tick / TICKS_PER_DAY;

        for (const cowId of cowIds) {{
            const engine = engines.get(cowId);
            const prod = prodModels.get(cowId);
            const state = cowState.get(cowId);

            // Episode triggers
            const triggered = scheduler.getTriggeredEpisodes(cowId, tick, schedule);
            for (const ep of triggered) {{
                if (ep.type === 'infection' && engine.isSusceptible()) {{
                    engine.seedInfection(ep);
                    state.currentEpisode = {{
                        startTick: tick,
                        maxSeverity: 0,
                        totalMilkLoss: 0,
                        tickCount: 0,
                        abxStarted: false,
                        abxDelay: 0
                    }};
                }}
            }}

            const stressBoost = scheduler.getActiveStressBoost(cowId, tick, schedule);

            // Determine severity
            const I = engine.state.I || 0;
            const currentSev = I < 0.01 ? 0 : I < 0.15 ? 1 : I < 0.5 ? 2 : 3;

            // Antibiotic trigger logic
            if (state.currentEpisode && !state.currentEpisode.abxStarted && currentSev >= TRIGGER_SEV) {{
                state.currentEpisode.abxStarted = true;
                state.currentEpisode.abxDelay = tick - state.currentEpisode.startTick;
                state.abxActive = true;
                state.abxStartTick = tick;
            }}

            // Deactivate after 7 days
            if (state.abxActive && (tick - state.abxStartTick) > 7 * TICKS_PER_DAY) {{
                state.abxActive = false;
            }}

            const context = {{
                antibioticActive: state.abxActive,
                stressSpike: stressBoost
            }};

            const hidden = engine.evolve(baseStress, context);
            const production = prod.generate(hidden, envSnap, tick, TICK_MINUTES, currentSev);

            // Track episode
            if (state.currentEpisode && hidden.infectionLoad > 0.01) {{
                state.currentEpisode.tickCount++;
                state.currentEpisode.maxSeverity = Math.max(state.currentEpisode.maxSeverity, currentSev);
                const milkLoss = Math.max(0, state.baselineMilk * 0.7 - production.milkYield);
                state.currentEpisode.totalMilkLoss += milkLoss;
            }}

            // Episode ended
            if (state.currentEpisode && hidden.infectionLoad < 0.01 && state.currentEpisode.tickCount > 10) {{
                state.currentEpisode.durationHours = (state.currentEpisode.tickCount * TICK_MINUTES) / 60;
                state.currentEpisode.durationDays = state.currentEpisode.durationHours / 24;
                state.currentEpisode.recoveryTick = tick;
                state.episodes.push(state.currentEpisode);
                state.currentEpisode = null;
                state.abxActive = false;
            }}
        }}
    }}

    // Aggregate results
    const allEpisodes = [];
    for (const [cowId, state] of cowState) {{
        for (const ep of state.episodes) {{
            allEpisodes.push({{
                cowId,
                durationHours: Math.round(ep.durationHours * 10) / 10,
                durationDays: Math.round(ep.durationDays * 10) / 10,
                maxSeverity: ep.maxSeverity,
                totalMilkLoss: Math.round(ep.totalMilkLoss * 100) / 100,
                abxDelay: Math.round((ep.abxDelay * TICK_MINUTES) / 60 * 10) / 10,
                abxStarted: ep.abxStarted
            }});
        }}
    }}

    const avgDuration = allEpisodes.length > 0
        ? allEpisodes.reduce((s, e) => s + e.durationHours, 0) / allEpisodes.length : 0;
    const avgMilkLoss = allEpisodes.length > 0
        ? allEpisodes.reduce((s, e) => s + e.totalMilkLoss, 0) / allEpisodes.length : 0;
    const avgMaxSev = allEpisodes.length > 0
        ? allEpisodes.reduce((s, e) => s + e.maxSeverity, 0) / allEpisodes.length : 0;

    console.log(JSON.stringify({{
        triggerSeverity: TRIGGER_SEV,
        totalEpisodes: allEpisodes.length,
        avgDurationHours: Math.round(avgDuration * 10) / 10,
        avgDurationDays: Math.round(avgDuration / 24 * 10) / 10,
        avgMilkLoss: Math.round(avgMilkLoss * 100) / 100,
        avgMaxSeverity: Math.round(avgMaxSev * 100) / 100,
        episodes: allEpisodes.slice(0, 20)
    }}));
    """

    result = subprocess.run(
        ['node', '-e', script],
        capture_output=True, text=True, cwd=BACKEND_DIR,
        timeout=120
    )

    if result.returncode != 0:
        logger.error(f"Simulation failed: {result.stderr[:500]}")
        return None

    try:
        return json.loads(result.stdout.strip())
    except json.JSONDecodeError:
        logger.error(f"Parse error: {result.stdout[:500]}")
        return None


# ═════════════════════════════════════════════════════════════════════════════════
# ANALYSIS
# ═════════════════════════════════════════════════════════════════════════════════

def run_intervention_analysis(num_cows=50, runs=5):
    """Run early vs late treatment scenarios and aggregate."""
    scenarios = [
        ('Early Treatment (sev=1)', 1),
        ('Late Treatment (sev=3)', 3),
    ]

    all_results = {}

    for name, trigger_sev in scenarios:
        logger.info(f"\n── Scenario: {name} ──")
        run_data = []

        for r in range(runs):
            logger.info(f"  Run {r+1}/{runs}...")
            result = run_intervention_scenario(trigger_sev, num_cows)
            if result:
                run_data.append(result)
                logger.info(f"    Episodes: {result['totalEpisodes']}, "
                          f"Avg Duration: {result['avgDurationHours']}h, "
                          f"Avg Milk Loss: {result['avgMilkLoss']}")

        if run_data:
            avg_duration = np.mean([r['avgDurationHours'] for r in run_data])
            avg_milk_loss = np.mean([r['avgMilkLoss'] for r in run_data])
            avg_max_sev = np.mean([r['avgMaxSeverity'] for r in run_data])
            avg_episodes = np.mean([r['totalEpisodes'] for r in run_data])

            all_results[name] = {
                'trigger_severity': trigger_sev,
                'avg_duration_hours': round(avg_duration, 1),
                'avg_duration_days': round(avg_duration / 24, 1),
                'avg_milk_loss': round(avg_milk_loss, 2),
                'avg_max_severity': round(avg_max_sev, 2),
                'avg_episodes': round(avg_episodes, 1),
                'runs': len(run_data)
            }

    return all_results


def generate_report(results):
    """Generate intervention comparison table and verdict."""
    output_dir = os.path.join(os.path.dirname(__file__), "../training_data")
    os.makedirs(output_dir, exist_ok=True)

    rows = []
    for name, data in results.items():
        rows.append({
            'Scenario': name,
            'Trigger Severity': data['trigger_severity'],
            'Avg Duration (hours)': data['avg_duration_hours'],
            'Avg Duration (days)': data['avg_duration_days'],
            'Avg Milk Loss': data['avg_milk_loss'],
            'Avg Max Severity': data['avg_max_severity'],
            'Episodes': data['avg_episodes'],
            'Runs': data['runs']
        })

    table = pd.DataFrame(rows)
    csv_path = os.path.join(output_dir, "intervention_results.csv")
    table.to_csv(csv_path, index=False)

    logger.info("\n── Intervention Comparison Table ──")
    logger.info(f"\n{table.to_string(index=False)}")

    # Compute milk loss difference
    early = results.get('Early Treatment (sev=1)', {})
    late = results.get('Late Treatment (sev=3)', {})

    if early and late:
        early_loss = early['avg_milk_loss']
        late_loss = late['avg_milk_loss']
        if late_loss > 0:
            pct_diff = ((late_loss - early_loss) / late_loss) * 100
        else:
            pct_diff = 0

        logger.info(f"\n── Treatment Impact Analysis ──")
        logger.info(f"  Early treatment milk loss: {early_loss:.2f}")
        logger.info(f"  Late treatment milk loss:  {late_loss:.2f}")
        logger.info(f"  Difference: {pct_diff:.1f}%  (target ≥20%)")

        duration_early = early['avg_duration_hours']
        duration_late = late['avg_duration_hours']
        logger.info(f"  Early duration: {duration_early:.1f}h")
        logger.info(f"  Late duration:  {duration_late:.1f}h")

        passed = pct_diff >= 20 and duration_late > duration_early
        logger.info(f"\n  VERDICT: {'✅ PASS' if passed else '❌ FAIL'}")
        logger.info(f"    Milk loss diff ≥20%: {'✅' if pct_diff >= 20 else '❌'} ({pct_diff:.1f}%)")
        logger.info(f"    Late duration > early: {'✅' if duration_late > duration_early else '❌'}")

        return table, passed, pct_diff
    else:
        logger.error("Missing scenario data")
        return table, False, 0


def generate_plots(results):
    """Generate comparison bar charts."""
    output_dir = os.path.join(os.path.dirname(__file__), "../training_data")

    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt

        names = list(results.keys())
        durations = [results[n]['avg_duration_hours'] for n in names]
        milk_losses = [results[n]['avg_milk_loss'] for n in names]
        max_sevs = [results[n]['avg_max_severity'] for n in names]

        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        colors = ['#27ae60', '#e74c3c']

        axes[0].bar(range(len(names)), durations, color=colors)
        axes[0].set_xticks(range(len(names)))
        axes[0].set_xticklabels(['Early (sev=1)', 'Late (sev=3)'])
        axes[0].set_ylabel('Hours')
        axes[0].set_title('Episode Duration', fontsize=13, fontweight='bold')
        axes[0].grid(True, alpha=0.3)

        axes[1].bar(range(len(names)), milk_losses, color=colors)
        axes[1].set_xticks(range(len(names)))
        axes[1].set_xticklabels(['Early (sev=1)', 'Late (sev=3)'])
        axes[1].set_ylabel('Liters')
        axes[1].set_title('Total Milk Loss per Episode', fontsize=13, fontweight='bold')
        axes[1].grid(True, alpha=0.3)

        axes[2].bar(range(len(names)), max_sevs, color=colors)
        axes[2].set_xticks(range(len(names)))
        axes[2].set_xticklabels(['Early (sev=1)', 'Late (sev=3)'])
        axes[2].set_ylabel('Severity (0-3)')
        axes[2].set_title('Avg Max Severity', fontsize=13, fontweight='bold')
        axes[2].grid(True, alpha=0.3)

        plt.tight_layout()
        plot_path = os.path.join(output_dir, "intervention_comparison.png")
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        plt.close()
        logger.info(f"  Plot saved: {plot_path}")

    except ImportError:
        logger.warning("matplotlib not available — skipping plots")


# ═════════════════════════════════════════════════════════════════════════════════
# MAIN
# ═════════════════════════════════════════════════════════════════════════════════

def main():
    import argparse
    parser = argparse.ArgumentParser(description='Intervention Simulation Validation')
    parser.add_argument('--cows', type=int, default=50)
    parser.add_argument('--runs', type=int, default=5)
    args = parser.parse_args()

    logger.info("=" * 60)
    logger.info("🧬 GoMata Intervention Validation Engine — v1")
    logger.info(f"   Cows: {args.cows}, Runs/scenario: {args.runs}")
    logger.info("=" * 60)

    results = run_intervention_analysis(args.cows, args.runs)

    if len(results) < 2:
        logger.error("Not all scenarios completed. Check simulator.")
        sys.exit(1)

    generate_plots(results)
    table, passed, pct_diff = generate_report(results)

    if passed:
        logger.info("\n🟢 INTERVENTION VALIDATION PASSED — Treatment dynamics are causal")
    else:
        logger.info("\n🔴 INTERVENTION VALIDATION FAILED — Treatment timing not differentiated")

    logger.info("=" * 60)


if __name__ == "__main__":
    main()
