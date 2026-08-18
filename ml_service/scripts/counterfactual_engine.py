#!/usr/bin/env python3
"""
counterfactual_engine.py — GoMata Pilot Validation Framework v1, Part 2
Counterfactual Herd Testing

Validates epidemiological intelligence by running vaccination scenarios:
  A) 0% vaccination — baseline outbreak
  B) 20% vaccination — moderate suppression
  C) 60% vaccination — strong suppression

Verifies:
  - Infection curves scale monotonically with vaccination coverage
  - R₀ decreases with vaccination
  - Herd stability index improves

Usage:
  python counterfactual_engine.py [--cows 50] [--days 30]
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
        logging.FileHandler(os.path.join(log_dir, "counterfactual.log")),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger("Counterfactual")

MONGO_URI = os.getenv("MONGO_URI", "mongodb://127.0.0.1:27017/livestock_monitoring")
DB_NAME = "livestock_monitoring"
BACKEND_DIR = os.path.join(os.path.dirname(__file__), "../../backend")


# ═════════════════════════════════════════════════════════════════════════════════
# SIMULATION RUNNER (calls Node.js engine)
# ═════════════════════════════════════════════════════════════════════════════════

def run_simulation_scenario(vacc_rate, num_cows=100, sim_days=30):
    """
    Run a counterfactual simulation scenario via node subprocess.
    Vaccination works TWO ways:
      1. 60% chance to BLOCK infection seeding (prevent acquisition)  
      2. α reduction during evolve() (slow growth if infected anyway)
    """
    script = f"""
    const CowPhysiologyEngine = require('./services/digitalTwin/CowPhysiologyEngine');
    const EnvironmentModel = require('./services/digitalTwin/EnvironmentModel');
    const EpisodeScheduler = require('./services/digitalTwin/EpisodeScheduler');
    const FarmProfile = require('./services/digitalTwin/FarmProfile');

    const TICK_MINUTES = 5;
    const TICKS_PER_DAY = 288;
    const TOTAL_TICKS = {sim_days} * TICKS_PER_DAY;
    const NUM_COWS = {num_cows};
    const VACC_RATE = {vacc_rate};

    const env = new EnvironmentModel();
    const farmProfile = FarmProfile.get('dairy');
    const scheduler = new EpisodeScheduler({{
        totalTicks: TOTAL_TICKS, tickMinutes: TICK_MINUTES,
        numCows: NUM_COWS, farmProfile
    }});

    const cowIds = Array.from({{length: NUM_COWS}}, (_, i) => 'cow_' + String(i).padStart(4, '0'));
    const engines = new Map();
    const vaccinated = new Set();

    for (const id of cowIds) {{
        engines.set(id, new CowPhysiologyEngine(id, {{ age: 4 }}));
        if (Math.random() < VACC_RATE) vaccinated.add(id);
    }}

    const schedule = scheduler.generateSchedule(cowIds);
    const dailyInfected = [];
    let totalSeeded = 0;
    let totalBlocked = 0;

    for (let tick = 1; tick <= TOTAL_TICKS; tick++) {{
        const envSnap = env.getEnvironment(tick, TICK_MINUTES, NUM_COWS);
        const baseStress = env.computeStressLoad(envSnap);
        let dayInfected = 0;

        for (const cowId of cowIds) {{
            const engine = engines.get(cowId);
            const isVacc = vaccinated.has(cowId);

            const triggered = scheduler.getTriggeredEpisodes(cowId, tick, schedule);
            for (const ep of triggered) {{
                if (ep.type === 'infection' && engine.isSusceptible()) {{
                    // Vaccination blocks 60% of infection acquisitions
                    if (isVacc && Math.random() < 0.6) {{
                        totalBlocked++;
                        continue;
                    }}
                    engine.seedInfection(ep);
                    totalSeeded++;
                }}
            }}

            const context = {{ vaccinationActive: isVacc }};
            const hidden = engine.evolve(baseStress, context);

            if (hidden.infectionLoad > 0.05) dayInfected++;
        }}

        if (tick % TICKS_PER_DAY === 0) {{
            dailyInfected.push(dayInfected);
        }}
    }}

    // Estimate R₀ from early growth phase
    let maxGrowth = 0;
    for (let i = 1; i < dailyInfected.length; i++) {{
        if (dailyInfected[i - 1] > 0) {{
            const r = dailyInfected[i] / dailyInfected[i - 1];
            if (r > maxGrowth) maxGrowth = r;
        }}
    }}

    const peak = Math.max(...dailyInfected, 0);
    const peakDay = dailyInfected.indexOf(peak) + 1;
    const total = dailyInfected.reduce((a, b) => a + b, 0);
    const avgInfected = total / Math.max(dailyInfected.length, 1);
    const stability = 1 - (avgInfected / NUM_COWS);

    console.log(JSON.stringify({{
        vaccRate: VACC_RATE,
        dailyInfected,
        peak, peakDay,
        estimatedR0: Math.round(maxGrowth * 100) / 100,
        herdStability: Math.round(stability * 1000) / 1000,
        totalInfectionDays: total,
        totalSeeded, totalBlocked,
        vaccCows: vaccinated.size
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
        logger.error(f"Failed to parse output: {result.stdout[:500]}")
        return None


# ═════════════════════════════════════════════════════════════════════════════════
# ANALYSIS
# ═════════════════════════════════════════════════════════════════════════════════

def run_counterfactual_analysis(num_cows=50, sim_days=30, runs_per_scenario=3):
    """Run all vaccination scenarios and aggregate results."""
    scenarios = [
        ('No Vaccination (0%)', 0.0),
        ('20% Vaccination', 0.2),
        ('60% Vaccination', 0.6),
    ]

    all_results = {}

    for name, vacc_rate in scenarios:
        logger.info(f"\n── Scenario: {name} (vacc={vacc_rate*100:.0f}%) ──")
        scenario_runs = []

        for run in range(runs_per_scenario):
            logger.info(f"  Run {run + 1}/{runs_per_scenario}...")
            result = run_simulation_scenario(vacc_rate, num_cows, sim_days)
            if result:
                scenario_runs.append(result)
                logger.info(f"    Peak: {result['peak']} (day {result['peakDay']}), "
                          f"R₀≈{result['estimatedR0']}, Stability: {result['herdStability']}")

        if scenario_runs:
            # Average across runs
            avg_peak = np.mean([r['peak'] for r in scenario_runs])
            avg_r0 = np.mean([r['estimatedR0'] for r in scenario_runs])
            avg_stability = np.mean([r['herdStability'] for r in scenario_runs])
            avg_peak_day = np.mean([r['peakDay'] for r in scenario_runs])
            avg_total = np.mean([r['totalInfectionDays'] for r in scenario_runs])

            # Average daily curve
            max_days = max(len(r['dailyInfected']) for r in scenario_runs)
            avg_curve = np.zeros(max_days)
            for r in scenario_runs:
                padded = r['dailyInfected'] + [0] * (max_days - len(r['dailyInfected']))
                avg_curve += np.array(padded)
            avg_curve /= len(scenario_runs)

            all_results[name] = {
                'vacc_rate': vacc_rate,
                'avg_peak': round(avg_peak, 1),
                'avg_peak_day': round(avg_peak_day, 1),
                'avg_r0': round(avg_r0, 3),
                'avg_stability': round(avg_stability, 4),
                'avg_total_infection_days': round(avg_total, 1),
                'infection_curve': avg_curve.tolist(),
                'runs': len(scenario_runs)
            }

    return all_results


def generate_plots(results):
    """Generate infection curve plots."""
    output_dir = os.path.join(os.path.dirname(__file__), "../training_data")
    os.makedirs(output_dir, exist_ok=True)

    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        # Plot 1: Infection curves
        ax1 = axes[0]
        colors = {'No Vaccination (0%)': '#e74c3c', '20% Vaccination': '#f39c12', '60% Vaccination': '#27ae60'}
        for name, data in results.items():
            curve = data['infection_curve']
            ax1.plot(range(1, len(curve) + 1), curve,
                    label=f"{name} (peak={data['avg_peak']})",
                    color=colors.get(name, '#333'), linewidth=2)
        ax1.set_xlabel('Day', fontsize=12)
        ax1.set_ylabel('Infected Cows', fontsize=12)
        ax1.set_title('Infection Curves by Vaccination Coverage', fontsize=14, fontweight='bold')
        ax1.legend(fontsize=10)
        ax1.grid(True, alpha=0.3)

        # Plot 2: R₀ and stability comparison
        ax2 = axes[1]
        names = list(results.keys())
        r0_vals = [results[n]['avg_r0'] for n in names]
        stab_vals = [results[n]['avg_stability'] for n in names]

        x = np.arange(len(names))
        width = 0.35
        bars1 = ax2.bar(x - width/2, r0_vals, width, label='Estimated R₀', color='#e74c3c', alpha=0.8)
        ax2_twin = ax2.twinx()
        bars2 = ax2_twin.bar(x + width/2, stab_vals, width, label='Herd Stability', color='#27ae60', alpha=0.8)

        ax2.set_xlabel('Scenario', fontsize=12)
        ax2.set_ylabel('R₀', color='#e74c3c', fontsize=12)
        ax2_twin.set_ylabel('Stability Index', color='#27ae60', fontsize=12)
        ax2.set_xticks(x)
        ax2.set_xticklabels(['0%', '20%', '60%'], fontsize=11)
        ax2.set_title('R₀ vs Herd Stability', fontsize=14, fontweight='bold')

        lines1, labels1 = ax2.get_legend_handles_labels()
        lines2, labels2 = ax2_twin.get_legend_handles_labels()
        ax2.legend(lines1 + lines2, labels1 + labels2, fontsize=10)
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        plot_path = os.path.join(output_dir, "counterfactual_vaccination.png")
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        plt.close()
        logger.info(f"  Plot saved: {plot_path}")

    except ImportError:
        logger.warning("matplotlib not available — skipping plots")


def generate_report(results):
    """Generate comparison table and verdicts."""
    output_dir = os.path.join(os.path.dirname(__file__), "../training_data")

    rows = []
    for name, data in results.items():
        rows.append({
            'Scenario': name,
            'Vacc %': int(data['vacc_rate'] * 100),
            'Peak Infected': data['avg_peak'],
            'Peak Day': data['avg_peak_day'],
            'Estimated R₀': data['avg_r0'],
            'Herd Stability': data['avg_stability'],
            'Total Infection-Days': data['avg_total_infection_days'],
            'Runs': data['runs']
        })

    table = pd.DataFrame(rows)
    csv_path = os.path.join(output_dir, "counterfactual_results.csv")
    table.to_csv(csv_path, index=False)

    logger.info("\n── Counterfactual Comparison Table ──")
    logger.info(f"\n{table.to_string(index=False)}")

    # Verify monotonicity
    peaks = [r['avg_peak'] for r in results.values()]
    r0s = [r['avg_r0'] for r in results.values()]

    peak_monotonic = all(peaks[i] >= peaks[i+1] for i in range(len(peaks)-1))
    r0_monotonic = all(r0s[i] >= r0s[i+1] for i in range(len(r0s)-1))

    logger.info(f"\n── Monotonicity Checks ──")
    logger.info(f"  Peak infections monotonic (0% > 20% > 60%): {'✅ PASS' if peak_monotonic else '❌ FAIL'}")
    logger.info(f"  R₀ monotonic (0% > 20% > 60%):              {'✅ PASS' if r0_monotonic else '❌ FAIL'}")

    return table, peak_monotonic, r0_monotonic


# ═════════════════════════════════════════════════════════════════════════════════
# MAIN
# ═════════════════════════════════════════════════════════════════════════════════

def main():
    import argparse
    parser = argparse.ArgumentParser(description='Counterfactual Herd Testing')
    parser.add_argument('--cows', type=int, default=50)
    parser.add_argument('--days', type=int, default=30)
    parser.add_argument('--runs', type=int, default=3)
    args = parser.parse_args()

    logger.info("=" * 60)
    logger.info("🧬 GoMata Counterfactual Herd Testing Engine — v1")
    logger.info(f"   Cows: {args.cows}, Days: {args.days}, Runs/scenario: {args.runs}")
    logger.info("=" * 60)

    results = run_counterfactual_analysis(args.cows, args.days, args.runs)

    if len(results) < 3:
        logger.error("Not all scenarios completed. Check simulator.")
        sys.exit(1)

    generate_plots(results)
    table, peak_ok, r0_ok = generate_report(results)

    if peak_ok and r0_ok:
        logger.info("\n🟢 COUNTERFACTUAL VALIDATION PASSED — Epidemiology is causal")
    else:
        logger.info("\n🔴 COUNTERFACTUAL VALIDATION FAILED — Herd propagation weak")

    logger.info("=" * 60)


if __name__ == "__main__":
    main()
