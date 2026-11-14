"""Theta re-tuning analysis for the 4D ablation.

For every ablation variant: test macro F1 with theta locked at the config value vs. theta
re-tuned on out-of-fold dev probabilities, plus the dev and test F1 curve over the theta grid.
The test curve is a diagnostic only; theta is still selected on dev.

Run from the repo root:
    python development/theta_analysis.py          # GPU if available
    python development/theta_analysis.py --cpu    # force CPU (laptop without CUDA)

Reads hyperparameters from results/test_set_1_4D_ablation_results.json so the run matches the
ablation exactly, then checks the re-tuned test F1 of each variant against that file.
Writes results/test_set_1_4D_theta_analysis.json
"""
import argparse
import json
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "development"))

ap = argparse.ArgumentParser()
ap.add_argument("--cpu", action="store_true")
ap.add_argument("--ablation", default=os.path.join(ROOT, "results", "test_set_1_4D_ablation_results.json"))
args = ap.parse_args()

from framework import config  # noqa: E402

with open(args.ablation, encoding="utf-8") as f:
    ablation = json.load(f)
cfg = ablation["config"]
config.TAU_S = cfg["tau_s"]
config.TFIDF_THRESH = cfg["tfidf"]
config.C = cfg["C"]
config.THETA = cfg["locked_theta"]
config.RANDOM_STATE = cfg["random_state"]
config.THETA_GRID = cfg["theta_grid"]
if args.cpu:
    config.DEVICE = "cpu"

import ablation_study as A  # noqa: E402  (main() does not run on import)
from sklearn.model_selection import StratifiedKFold  # noqa: E402

A.THETA_VALUES = cfg["theta_grid"]
stored = {r["name"]: r for r in ablation["results"]}


def oof_probs(X, y):
    skf = StratifiedKFold(n_splits=A.N_FOLDS, shuffle=True, random_state=config.RANDOM_STATE)
    folds = list(skf.split(X, y))
    oof = np.zeros((len(y), len(A.LABELS)))
    for tr, va in folds:
        m = A.make_policy_model().fit(X[tr], y[tr])
        assert list(m.classes_) == A.LABELS
        oof[va] = m.predict_proba(X[va])
    return oof, folds


def main():
    print(f"tau_s={config.TAU_S} tfidf={config.TFIDF_THRESH} C={config.C} "
          f"locked theta={config.THETA} device={config.DEVICE}")
    dev_records, dev_raw = A.load_raw_features(A.DEV_PATH, A.DEV_RAW_CACHE, A.FeatureExtractor)
    test_records, test_raw = A.load_raw_features(A.TEST_PATH, A.TEST_RAW_CACHE, A.FeatureExtractor)

    def matrix(raws):
        return np.array([np.concatenate([A.features_from_raw(r), A.shortcut_features_from_raw(r)]) for r in raws])

    X_dev, X_test = matrix(dev_raw), matrix(test_raw)
    y_dev = np.array([r["gold_action"] for r in dev_records])
    y_test_idx = A.encode([r["gold_action"] for r in test_records])
    y_dev_idx = A.encode(y_dev)
    rng = np.random.default_rng(config.RANDOM_STATE)

    out = []
    for v in A.VARIANTS:
        if v["theta"] == "argmax":
            continue
        keep = v["use"]
        Xd, Xt = X_dev[:, keep], X_test[:, keep]
        oof, folds = oof_probs(Xd, y_dev)
        dev_curve = {}
        for th in A.THETA_VALUES:
            p = A.encode(A.gated_predict(oof, A.LABELS, th))
            dev_curve[th] = float(np.mean([A.macro_f1(y_dev_idx[va], p[va]) for _, va in folds]))
        tuned = max(A.THETA_VALUES, key=lambda th: (dev_curve[th], -th))  # ties -> smallest theta

        model = A.make_policy_model().fit(Xd, y_dev)
        probs = model.predict_proba(Xt)
        test_curve = {th: A.macro_f1(y_test_idx, A.encode(A.gated_predict(probs, model.classes_, th)))
                      for th in A.THETA_VALUES}
        p_locked = A.encode(A.gated_predict(probs, model.classes_, config.THETA))
        p_tuned = A.encode(A.gated_predict(probs, model.classes_, tuned))
        f1_locked, f1_tuned = A.macro_f1(y_test_idx, p_locked), A.macro_f1(y_test_idx, p_tuned)
        lo, hi, pval = A.paired_bootstrap(y_test_idx, p_tuned, p_locked, rng)

        ref = stored.get(v["name"], {})
        ref_f1 = ref.get("test_macro_f1")
        ref_theta = ref.get("theta")
        check_val = f1_locked if v["theta"] == "locked" else f1_tuned
        match = ref_f1 is not None and abs(check_val - ref_f1) < 5e-4
        out.append({
            "name": v["name"],
            "theta_locked": config.THETA,
            "theta_tuned": tuned,
            "theta_tuned_in_ablation": ref_theta,
            "dev_f1_at_tuned": dev_curve[tuned],
            "dev_f1_at_locked": dev_curve[config.THETA],
            "test_f1_locked": f1_locked,
            "test_f1_tuned": f1_tuned,
            "tuned_minus_locked": f1_tuned - f1_locked,
            "ci95": [lo, hi],
            "p": pval,
            "test_f1_in_ablation": ref_f1,
            "matches_ablation": match,
            "dev_curve": dev_curve,
            "test_curve": test_curve,
        })

    print(f"\n{'Variant':<34}{'theta*':>7}{'dev@0.40':>10}{'dev@theta*':>11}{'test@0.40':>10}"
          f"{'test@theta*':>12}{'diff':>9}{'95% CI':>18}{'p':>7}  check")
    for r in out:
        ci = f"[{r['ci95'][0]:+.3f},{r['ci95'][1]:+.3f}]"
        chk = "OK" if r["matches_ablation"] else f"MISMATCH (ablation {r['test_f1_in_ablation']})"
        print(f"{r['name']:<34}{r['theta_tuned']:>7.2f}{r['dev_f1_at_locked']:>10.4f}{r['dev_f1_at_tuned']:>11.4f}"
              f"{r['test_f1_locked']:>10.4f}{r['test_f1_tuned']:>12.4f}{r['tuned_minus_locked']:>+9.4f}{ci:>18}"
              f"{r['p']:>7.3f}  {chk}")

    print("\nTest macro F1 by theta (diagnostic only, theta is selected on dev):")
    print(f"{'Variant':<34}" + "".join(f"{th:>6.2f}" for th in A.THETA_VALUES))
    for r in out:
        print(f"{r['name']:<34}" + "".join(f"{r['test_curve'][th]:>6.3f}" for th in A.THETA_VALUES))

    path = os.path.join(ROOT, "results", f"{config.RUN_TAG}_theta_analysis.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump({"config": cfg, "variants": out}, f, indent=2)
    print(f"\nSaved to {path}")


if __name__ == "__main__":
    main()
