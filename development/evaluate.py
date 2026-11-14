"""
Evaluates saved per-example result files on the test set. No models are loaded and no GPU is
needed: it reads results/*.json only.

Reports, for the 4D policy, each baseline, and the shortcut controls:
  macro-F1 over the 4 actions, accuracy, per-class F1, accuracy by source dataset, and the
  paired-bootstrap difference against the 4D policy (B = 1000, seed 42).

Shortcut controls (no learning):
  k-only rule      1 passage -> ANSWER, 2 -> DISCLOSE, 3 -> ABSTAIN
  majority class   always ANSWER

Usage:  python development/evaluate.py            # print the table
        python development/evaluate.py --save     # also write results/<RUN_TAG>_summary.json
"""
import argparse
import json
import os
import re
import sys

import numpy as np
from sklearn.metrics import f1_score

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(ROOT_DIR)

from framework import config

LABELS = ["ABSTAIN", "ANSWER", "DISCLAIMER", "DISCLOSE"]
UNSEEN_SOURCES = {"whoqa", "nomiracl"}  # never appear in dev_set_natural
CRAG_MAP = {"INCORRECT": "ABSTAIN", "AMBIGUOUS": "DISCLAIMER", "CORRECT": "ANSWER"}
K_RULE = {1: "ANSWER", 2: "DISCLOSE", 3: "ABSTAIN"}
N_BOOTSTRAP = 1000

RESULTS = os.path.join(ROOT_DIR, "results")
SYSTEMS = {
    "4D policy": (os.path.join(RESULTS, f"{config.RUN_TAG}_results.json"), "predicted_action"),
    "6D policy (previous)": (os.path.join(RESULTS, "test_set_1_results_final.json"), "predicted_action"),
    "Vanilla LLM router": (os.path.join(RESULTS, "vanilla_test_set_1_results.json"), "predicted_action"),
    # Static-CRAG: map the CRAG evaluator state to an action (no gold information used)
    "Static-CRAG": (os.path.join(RESULTS, "crag_test_set_1_results.json"), "crag_internal_state"),
}


def macro_f1(y, p):
    return float(f1_score(y, p, labels=LABELS, average="macro", zero_division=0))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--save", action="store_true")
    args = ap.parse_args()

    with open(config.DATA_PATH, encoding="utf-8") as f:
        test = {r["id"]: r for r in (json.loads(l) for l in f)}
    ids = list(test)
    y = np.array([test[i]["gold_action"] for i in ids])
    k = np.array([len(test[i]["passages"]) for i in ids])
    src = np.array([re.split(r"[-_]", i)[0] for i in ids])

    preds = {}
    for name, (path, field) in SYSTEMS.items():
        with open(path, encoding="utf-8") as f:
            rows = {r["id"]: r for r in json.load(f)}
        raw = [rows[i][field] for i in ids]
        preds[name] = np.array([CRAG_MAP.get(v, v) for v in raw] if field == "crag_internal_state" else raw)
    preds["Control: k-only rule"] = np.array([K_RULE[x] for x in k])
    preds["Control: always ANSWER"] = np.array(["ANSWER"] * len(y))

    ref = preds["4D policy"]
    rng = np.random.default_rng(config.RANDOM_STATE)
    samples = [rng.integers(0, len(y), len(y)) for _ in range(N_BOOTSTRAP)]

    summary = {}
    for name, p in preds.items():
        cls = f1_score(y, p, labels=LABELS, average=None, zero_division=0)
        row = {
            "macro_f1": macro_f1(y, p),
            "accuracy": float(np.mean(y == p)),
            "class_f1": {lab: float(v) for lab, v in zip(LABELS, cls)},
            "accuracy_by_source": {s: float(np.mean(y[src == s] == p[src == s])) for s in sorted(set(src))},
            "accuracy_seen_sources": float(np.mean((y == p)[~np.isin(src, list(UNSEEN_SOURCES))])),
            "accuracy_unseen_sources": float(np.mean((y == p)[np.isin(src, list(UNSEEN_SOURCES))])),
        }
        if name != "4D policy":
            d = np.array([macro_f1(y[s], p[s]) - macro_f1(y[s], ref[s]) for s in samples])
            row["delta_vs_4D"] = row["macro_f1"] - macro_f1(y, ref)
            row["delta_ci95"] = [float(v) for v in np.percentile(d, [2.5, 97.5])]
        summary[name] = row

    print(f"Test set: {len(y)} examples. Delta = system minus 4D policy (paired bootstrap, B={N_BOOTSTRAP}).\n")
    print(f"{'System':<26}{'Macro-F1':>9}{'Acc':>7}{'ABST':>7}{'ANS':>7}{'DSCLM':>7}{'DSCLO':>7}"
          f"{'Seen':>7}{'Unseen':>8}   Delta [95% CI]")
    for name, r in summary.items():
        c = r["class_f1"]
        delta = (f"   {r['delta_vs_4D']:+.3f} [{r['delta_ci95'][0]:+.3f}, {r['delta_ci95'][1]:+.3f}]"
                 if "delta_vs_4D" in r else "   --")
        print(f"{name:<26}{r['macro_f1']:>9.3f}{r['accuracy']:>7.3f}{c['ABSTAIN']:>7.3f}{c['ANSWER']:>7.3f}"
              f"{c['DISCLAIMER']:>7.3f}{c['DISCLOSE']:>7.3f}{r['accuracy_seen_sources']:>7.3f}"
              f"{r['accuracy_unseen_sources']:>8.3f}{delta}")

    print("\nAccuracy by source:")
    sources = sorted(set(src))
    print(f"{'System':<26}" + "".join(f"{s:>11}" for s in sources))
    for name, r in summary.items():
        print(f"{name:<26}" + "".join(f"{r['accuracy_by_source'][s]:>11.3f}" for s in sources))

    if args.save:
        out = os.path.join(RESULTS, f"{config.RUN_TAG}_summary.json")
        with open(out, "w", encoding="utf-8") as f:
            json.dump({"n": len(y), "n_bootstrap": N_BOOTSTRAP, "systems": summary}, f, indent=2)
        print(f"\nSaved to {out}")


if __name__ == "__main__":
    main()
