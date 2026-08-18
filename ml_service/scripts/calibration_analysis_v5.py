#!/usr/bin/env python3
"""
calibration_analysis_v5.py — GoMata Phase 6, Parts 1 & 2
Prevalence baseline, PR lift, reliability diagram, ECE, Brier score.
"""

import os, sys, json, logging, time
import numpy as np
import pandas as pd
import joblib
from sklearn.metrics import (roc_auc_score, average_precision_score,
                             brier_score_loss, precision_recall_curve)

sys.path.insert(0, os.path.dirname(__file__))
from v5_config import ALL_FEATURES, load_and_fuse, DATA_DIR, MODEL_DIR
from split_strategy_v2 import animal_time_split

log_dir = os.path.join(os.path.dirname(__file__), "../logs")
os.makedirs(log_dir, exist_ok=True)
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.FileHandler(os.path.join(log_dir, "calibration_v5.log"), mode='w'),
              logging.StreamHandler(sys.stdout)])
logger = logging.getLogger("CalibrationV5")


def compute_calibration(y_true, y_prob, n_bins=10):
    bins = np.linspace(0, 1, n_bins + 1)
    bin_data = []
    ece = 0
    mce = 0
    for lo, hi in zip(bins[:-1], bins[1:]):
        mask = (y_prob >= lo) & (y_prob < hi)
        count = mask.sum()
        if count == 0:
            bin_data.append({"bin": f"{lo:.1f}-{hi:.1f}", "count": 0,
                "avg_confidence": 0, "avg_accuracy": 0, "gap": 0})
            continue
        avg_conf = float(y_prob[mask].mean())
        avg_acc = float(y_true[mask].mean())
        gap = abs(avg_conf - avg_acc)
        ece += count * gap
        mce = max(mce, gap)
        bin_data.append({"bin": f"{lo:.1f}-{hi:.1f}", "count": int(count),
            "avg_confidence": round(avg_conf, 4), "avg_accuracy": round(avg_acc, 4),
            "gap": round(gap, 4)})
    ece /= len(y_true)
    return bin_data, round(ece, 6), round(mce, 6)


def save_plots(y_true, y_prob, bin_data, out_dir):
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt

        precisions, recalls, _ = precision_recall_curve(y_true, y_prob)
        fig, ax = plt.subplots(1, 1, figsize=(8, 6))
        ax.plot(recalls, precisions, 'b-', lw=2, label='Model')
        prev = y_true.mean()
        ax.axhline(y=prev, color='r', ls='--', label=f'Random ({prev:.3f})')
        ax.set_xlabel('Recall'); ax.set_ylabel('Precision')
        ax.set_title('PR Curve (v5)'); ax.legend(); ax.grid(True, alpha=0.3)
        fig.tight_layout(); fig.savefig(os.path.join(out_dir, "pr_curve_v5.png"), dpi=150)
        plt.close(fig)
        logger.info("  ✅ pr_curve_v5.png")

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
        bins_with_data = [b for b in bin_data if b['count'] > 0]
        x = [b['avg_confidence'] for b in bins_with_data]
        y = [b['avg_accuracy'] for b in bins_with_data]
        ax1.bar(x, y, width=0.08, alpha=0.7, color='steelblue', edgecolor='black')
        ax1.plot([0, 1], [0, 1], 'r--', lw=1.5, label='Perfect')
        ax1.set_xlabel('Mean Predicted Prob'); ax1.set_ylabel('Fraction of Positives')
        ax1.set_title('Reliability Diagram'); ax1.legend(); ax1.set_xlim(0,1); ax1.set_ylim(0,1)
        ax1.grid(True, alpha=0.3)
        ax2.hist(y_prob, bins=50, alpha=0.7, color='steelblue', edgecolor='black')
        ax2.set_xlabel('Predicted Prob'); ax2.set_ylabel('Count')
        ax2.set_title('Confidence Distribution'); ax2.grid(True, alpha=0.3)
        fig.tight_layout(); fig.savefig(os.path.join(out_dir, "reliability_plot_v5.png"), dpi=150)
        plt.close(fig)
        logger.info("  ✅ reliability_plot_v5.png")
    except ImportError:
        logger.warning("matplotlib unavailable")


def main():
    logger.info("=" * 60)
    logger.info("📊 Phase 6 Part 1+2 — Calibration Analysis")
    logger.info("=" * 60)

    model = joblib.load(os.path.join(MODEL_DIR, "model_v5.pkl"))
    df = load_and_fuse()
    logger.info(f"Fused: {len(df)} rows")

    _, X_test, _, y_test, _ = animal_time_split(
        df, ALL_FEATURES, "disease_binary",
        animal_train_ratio=0.8, time_train_ratio=0.7, seed=42)

    y_prob = model.predict_proba(X_test)[:, 1]
    y_true = y_test.values

    prevalence = float(y_true.mean())
    pr_auc = average_precision_score(y_true, y_prob)
    roc_auc = roc_auc_score(y_true, y_prob)
    pr_lift = pr_auc / prevalence if prevalence > 0 else 0

    logger.info(f"  Prevalence: {prevalence:.4f} ({prevalence*100:.2f}%)")
    logger.info(f"  PR-AUC: {pr_auc:.4f}")
    logger.info(f"  PR Lift: {pr_lift:.1f}x {'✅' if pr_lift >= 4 else '❌'}")

    bin_data, ece, mce = compute_calibration(y_true, y_prob)
    brier = brier_score_loss(y_true, y_prob)

    logger.info(f"  ECE: {ece:.4f} {'✅' if ece <= 0.05 else '❌'}")
    logger.info(f"  MCE: {mce:.4f}")
    logger.info(f"  Brier: {brier:.4f}")

    for b in bin_data:
        if b['count'] > 0:
            flag = '⚠️' if b['gap'] > 0.1 else ''
            logger.info(f"    {b['bin']}: conf={b['avg_confidence']:.3f} "
                       f"acc={b['avg_accuracy']:.3f} gap={b['gap']:.3f} n={b['count']} {flag}")

    high_bins = [b for b in bin_data if b['count'] > 0 and b['avg_confidence'] > 0.7]
    overconfident = any(b['gap'] > 0.15 for b in high_bins)

    save_plots(y_true, y_prob, bin_data, DATA_DIR)

    report = {
        "prevalence": round(prevalence, 6), "pr_auc": round(float(pr_auc), 4),
        "roc_auc": round(float(roc_auc), 4), "pr_lift": round(float(pr_lift), 2),
        "pr_lift_pass": pr_lift >= 4, "ece": ece, "mce": mce,
        "brier_score": round(float(brier), 6), "ece_pass": ece <= 0.05,
        "overconfident_high_bins": overconfident, "reliability_bins": bin_data,
        "test_size": len(y_true), "test_positive": int(y_true.sum()),
    }
    with open(os.path.join(DATA_DIR, "calibration_metrics_v5.json"), "w") as f:
        json.dump(report, f, indent=2)
    logger.info(f"  ✅ calibration_metrics_v5.json saved")


if __name__ == "__main__":
    main()
