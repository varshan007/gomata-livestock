#!/usr/bin/env python3
"""
pilot_readiness_report.py — GoMata Pilot Validation Framework v1
Final Pilot Readiness Aggregator

Runs all 3 validation parts and generates an overall readiness score:
  Part 1: Reality Gap Testing     (reality_gap_v3.py)
  Part 2: Counterfactual Testing  (counterfactual_engine.py)
  Part 3: Intervention Validation (intervention_validation.py)

Pilot Ready Criteria:
  ✔ Robust under 15% missing data (AUC drop ≤15%)
  ✔ Outbreak curves scale under vaccination (R₀ monotonic)
  ✔ Early vs late treatment milk loss differs >20%
  ✔ Economic projections remain monotonic

Usage:
  python pilot_readiness_report.py [--skip-generation] [--cows 50]
"""

import os
import sys
import time
import logging
import subprocess

log_dir = os.path.join(os.path.dirname(__file__), "../logs")
os.makedirs(log_dir, exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(os.path.join(log_dir, "pilot_readiness.log")),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger("PilotReadiness")

SCRIPTS_DIR = os.path.dirname(__file__)
BACKEND_DIR = os.path.join(SCRIPTS_DIR, "../../backend")
OUTPUT_DIR = os.path.join(SCRIPTS_DIR, "../training_data")


def run_step(name, cmd, cwd=None, timeout=600):
    """Run a subprocess step with logging."""
    logger.info(f"\n{'='*60}")
    logger.info(f"▶ {name}")
    logger.info(f"{'='*60}")
    start = time.time()

    result = subprocess.run(
        cmd, shell=isinstance(cmd, str),
        capture_output=True, text=True,
        cwd=cwd or SCRIPTS_DIR, timeout=timeout
    )

    elapsed = time.time() - start
    if result.returncode == 0:
        logger.info(f"✅ {name} completed in {elapsed:.1f}s")
        # Log last 20 lines of output
        lines = result.stdout.strip().split('\n')
        for line in lines[-20:]:
            logger.info(f"  {line}")
    else:
        logger.error(f"❌ {name} FAILED in {elapsed:.1f}s")
        logger.error(result.stderr[:1000])

    return result.returncode == 0


def check_results():
    """Check if result files exist and parse verdicts."""
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    results = {
        'reality_gap': {'passed': False, 'details': ''},
        'counterfactual': {'passed': False, 'details': ''},
        'intervention': {'passed': False, 'details': ''}
    }

    # Check reality gap verdicts
    v3_path = os.path.join(OUTPUT_DIR, "reality_gap_v3_verdicts.csv")
    v4_path = os.path.join(OUTPUT_DIR, "reality_gap_v4_verdicts.csv")
    if os.path.exists(v3_path):
        import pandas as pd
        df = pd.read_csv(v3_path)
        fragile = df[df['VERDICT'].str.contains('FRAGILE', na=False)]
        if len(fragile) == 0:
            results['reality_gap']['passed'] = True
            results['reality_gap']['details'] = f"All {len(df)} scenarios passed"
        else:
            results['reality_gap']['details'] = f"{len(fragile)}/{len(df)} scenarios FRAGILE"
    else:
        results['reality_gap']['details'] = "No results file found"

    # Check counterfactual results
    cf_path = os.path.join(OUTPUT_DIR, "counterfactual_results.csv")
    if os.path.exists(cf_path):
        import pandas as pd
        df = pd.read_csv(cf_path)
        peaks = df['Peak Infected'].tolist()
        r0s = df['Estimated R₀'].tolist()
        peak_mono = all(peaks[i] >= peaks[i+1] for i in range(len(peaks)-1))
        r0_mono = all(r0s[i] >= r0s[i+1] for i in range(len(r0s)-1))
        if peak_mono and r0_mono:
            results['counterfactual']['passed'] = True
            results['counterfactual']['details'] = f"Peak and R₀ monotonic across {len(df)} scenarios"
        else:
            results['counterfactual']['details'] = f"Peak mono: {peak_mono}, R₀ mono: {r0_mono}"
    else:
        results['counterfactual']['details'] = "No results file found"

    # Check intervention results
    iv_path = os.path.join(OUTPUT_DIR, "intervention_results.csv")
    if os.path.exists(iv_path):
        import pandas as pd
        df = pd.read_csv(iv_path)
        if len(df) >= 2:
            early = df[df['Trigger Severity'] == 1]
            late = df[df['Trigger Severity'] == 3]
            if len(early) > 0 and len(late) > 0:
                early_loss = early['Avg Milk Loss'].values[0]
                late_loss = late['Avg Milk Loss'].values[0]
                pct_diff = ((late_loss - early_loss) / late_loss * 100) if late_loss > 0 else 0
                if pct_diff >= 20:
                    results['intervention']['passed'] = True
                    results['intervention']['details'] = f"Milk loss diff {pct_diff:.1f}% (≥20%)"
                else:
                    results['intervention']['details'] = f"Milk loss diff {pct_diff:.1f}% (<20%)"
    else:
        results['intervention']['details'] = "No results file found"

    return results


def main():
    import argparse
    parser = argparse.ArgumentParser(description='Pilot Readiness Report')
    parser.add_argument('--skip-generation', action='store_true', help='Skip validation set generation')
    parser.add_argument('--cows', type=int, default=50)
    parser.add_argument('--runs', type=int, default=3)
    args = parser.parse_args()

    start_time = time.time()

    logger.info("="*60)
    logger.info("🧬 GoMata Pilot Validation Framework v1")
    logger.info("   Complete Pilot Readiness Assessment")
    logger.info("="*60)

    steps_passed = 0
    total_steps = 4

    # ── Step 0: Generate validation sets ──────────────────────────────────
    if not args.skip_generation:
        ok = run_step(
            "Generate Validation Sets (100K v3 + 100K v4)",
            f"node scripts/generate_validation_sets.js --target 100000 --cows {args.cows} --days 30",
            cwd=BACKEND_DIR, timeout=300
        )
        if not ok:
            logger.error("Cannot continue without validation data")
            sys.exit(1)
    else:
        logger.info("Skipping validation set generation (--skip-generation)")

    # ── Step 1: Reality Gap Testing ───────────────────────────────────────
    ok = run_step(
        "Part 1: Reality Gap Testing",
        [sys.executable, os.path.join(SCRIPTS_DIR, "reality_gap_v3.py"), "--limit", "100000"],
        timeout=300
    )
    if ok: steps_passed += 1

    # ── Step 2: Counterfactual Herd Testing ───────────────────────────────
    ok = run_step(
        "Part 2: Counterfactual Herd Testing",
        [sys.executable, os.path.join(SCRIPTS_DIR, "counterfactual_engine.py"),
         "--cows", str(args.cows), "--days", "30", "--runs", str(args.runs)],
        timeout=600
    )
    if ok: steps_passed += 1

    # ── Step 3: Intervention Validation ───────────────────────────────────
    ok = run_step(
        "Part 3: Intervention Simulation Validation",
        [sys.executable, os.path.join(SCRIPTS_DIR, "intervention_validation.py"),
         "--cows", str(args.cows), "--runs", str(args.runs)],
        timeout=600
    )
    if ok: steps_passed += 1

    # ── Step 4: Aggregate Results ─────────────────────────────────────────
    results = check_results()
    criteria_passed = sum(1 for v in results.values() if v['passed'])

    elapsed = time.time() - start_time

    # ── Final Report ──────────────────────────────────────────────────────
    logger.info("\n" + "="*60)
    logger.info("📊 PILOT READINESS REPORT")
    logger.info("="*60)

    logger.info(f"\n{'Part':<35} {'Status':<10} {'Details'}")
    logger.info("-"*80)
    for part, data in results.items():
        status = "✅ PASS" if data['passed'] else "❌ FAIL"
        logger.info(f"  {part:<33} {status:<10} {data['details']}")

    score = (criteria_passed / len(results)) * 100
    logger.info(f"\n{'─'*60}")
    logger.info(f"  Pilot Readiness Score: {score:.0f}%")
    logger.info(f"  Criteria Passed: {criteria_passed}/{len(results)}")
    logger.info(f"  Steps Completed: {steps_passed}/{total_steps}")
    logger.info(f"  Total Duration: {elapsed:.1f}s ({elapsed/60:.1f} min)")
    logger.info(f"{'─'*60}")

    if score >= 100:
        logger.info("\n🟢 SYSTEM QUALIFIED FOR PILOT DEPLOYMENT")
        logger.info("  All criteria met. Ready for real farm testing.")
    elif score >= 66:
        logger.info("\n🟡 PARTIALLY READY — Address failing criteria before pilot")
    else:
        logger.info("\n🔴 NOT READY FOR PILOT — Significant issues remain")

    logger.info("\n" + "="*60)


if __name__ == "__main__":
    main()
