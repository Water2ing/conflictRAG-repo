"""Evaluate a ConflictRAG results file (output of framework.main).

Usage:
    python development/evaluate_results.py results/test_set_1_4D_results.json \
        --data data/test_set_1.jsonl \
        --out results/test_set_1_4D_eval.json

--data     test jsonl, used to get k (number of passages) per example
--compare  another results file on the same examples (e.g. the 6D run)
--out      write all numbers to a json file
"""
import argparse
import json
import random
from collections import Counter, defaultdict

LABELS = ["ABSTAIN", "ANSWER", "DISCLAIMER", "DISCLOSE"]


def load_results(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def load_k(path):
    k = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            item = json.loads(line)
            k[item["id"]] = len(item["passages"])
    return k


def per_class(gold, pred, labels):
    out = {}
    for c in labels:
        tp = sum(g == c and p == c for g, p in zip(gold, pred))
        fp = sum(g != c and p == c for g, p in zip(gold, pred))
        fn = sum(g == c and p != c for g, p in zip(gold, pred))
        prec = tp / (tp + fp) if tp + fp else 0.0
        rec = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
        out[c] = {"precision": prec, "recall": rec, "f1": f1, "support": tp + fn}
    return out


def macro_f1(gold, pred, labels=LABELS):
    pc = per_class(gold, pred, labels)
    return sum(v["f1"] for v in pc.values()) / len(labels)


def accuracy(gold, pred):
    return sum(g == p for g, p in zip(gold, pred)) / len(gold) if gold else 0.0


def confusion(gold, pred):
    m = {g: {p: 0 for p in LABELS} for g in LABELS}
    for g, p in zip(gold, pred):
        m[g][p] += 1
    return m


def paired_bootstrap(gold, pred_a, pred_b, b=1000, seed=42):
    """Macro F1 difference a - b with 95% CI and two-sided p."""
    rng = random.Random(seed)
    n = len(gold)
    obs = macro_f1(gold, pred_a) - macro_f1(gold, pred_b)
    diffs = []
    for _ in range(b):
        idx = [rng.randrange(n) for _ in range(n)]
        g = [gold[i] for i in idx]
        diffs.append(macro_f1(g, [pred_a[i] for i in idx]) - macro_f1(g, [pred_b[i] for i in idx]))
    diffs.sort()
    lo, hi = diffs[int(0.025 * b)], diffs[int(0.975 * b) - 1]
    p = 2 * min(sum(d <= 0 for d in diffs), sum(d >= 0 for d in diffs)) / b
    return {"diff": obs, "ci95": [lo, hi], "p": min(p, 1.0)}


def source_of(example_id):
    return example_id.split("-")[0]


def print_table(rows, header):
    widths = [max(len(str(r[i])) for r in rows + [header]) for i in range(len(header))]
    line = "  ".join(str(h).ljust(w) for h, w in zip(header, widths))
    print(line)
    print("-" * len(line))
    for r in rows:
        print("  ".join(str(c).ljust(w) for c, w in zip(r, widths)))
    print()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("results")
    ap.add_argument("--data")
    ap.add_argument("--compare")
    ap.add_argument("--out")
    args = ap.parse_args()

    res = load_results(args.results)
    ids = [x["id"] for x in res]
    gold = [x["gold_action"] for x in res]
    pred = [x["predicted_action"] for x in res]
    summary = {"results_file": args.results, "n": len(res)}

    # overall
    pc = per_class(gold, pred, LABELS)
    summary["macro_f1"] = macro_f1(gold, pred)
    summary["accuracy"] = accuracy(gold, pred)
    summary["per_class"] = pc
    summary["confusion"] = confusion(gold, pred)
    print(f"== Overall (n={len(res)})")
    print(f"macro F1 = {summary['macro_f1']:.4f}   accuracy = {summary['accuracy']:.4f}\n")
    print_table([[c, f"{v['precision']:.3f}", f"{v['recall']:.3f}", f"{v['f1']:.3f}", v["support"]] for c, v in pc.items()],
                ["class", "precision", "recall", "f1", "support"])
    print("== Confusion matrix (rows = gold, cols = predicted)")
    print_table([[g] + [summary["confusion"][g][p] for p in LABELS] for g in LABELS], ["gold"] + LABELS)

    # by source
    by_src = defaultdict(list)
    for i, x in enumerate(res):
        by_src[source_of(x["id"])].append(i)
    summary["by_source"] = {}
    rows = []
    for s, idx in sorted(by_src.items()):
        g = [gold[i] for i in idx]
        p = [pred[i] for i in idx]
        dist = dict(Counter(g))
        summary["by_source"][s] = {"n": len(idx), "gold": dist, "accuracy": accuracy(g, p)}
        rows.append([s, len(idx), ", ".join(f"{k} {v}" for k, v in sorted(dist.items())), f"{accuracy(g, p):.3f}"])
    print("== By source")
    print_table(rows, ["source", "n", "gold labels", "accuracy"])

    # by k
    if args.data:
        kmap = load_k(args.data)
        missing = [i for i in ids if i not in kmap]
        if missing:
            raise SystemExit(f"{len(missing)} ids not found in {args.data}, e.g. {missing[:3]}")
        ks = [kmap[i] for i in ids]

        print("== Source x k (count)")
        all_k = sorted(set(ks))
        sk = Counter((source_of(i), k) for i, k in zip(ids, ks))
        print_table([[s] + [sk[(s, k)] for k in all_k] for s in sorted(by_src)], ["source"] + [f"k={k}" for k in all_k])

        print("== Within each k: model vs majority label of that group")
        summary["by_k"] = {}
        rows = []
        for k in all_k:
            idx = [j for j, kk in enumerate(ks) if kk == k]
            g = [gold[j] for j in idx]
            p = [pred[j] for j in idx]
            labels_k = sorted(set(g))
            maj = Counter(g).most_common(1)[0][0]
            entry = {
                "n": len(idx),
                "gold": dict(Counter(g)),
                "accuracy": accuracy(g, p),
                "macro_f1_within": macro_f1(g, p, labels_k) if len(labels_k) > 1 else None,
                "majority_label": maj,
                "majority_accuracy": g.count(maj) / len(g),
                "majority_macro_f1_within": macro_f1(g, [maj] * len(g), labels_k) if len(labels_k) > 1 else None,
            }
            summary["by_k"][k] = entry
            fmt = lambda v: "-" if v is None else f"{v:.3f}"
            rows.append([k, len(idx), ", ".join(f"{a} {b}" for a, b in sorted(entry["gold"].items())),
                         fmt(entry["macro_f1_within"]), fmt(entry["majority_macro_f1_within"]),
                         f"{entry['accuracy']:.3f}", f"{entry['majority_accuracy']:.3f}"])
        print_table(rows, ["k", "n", "gold labels", "model F1", "majority F1", "model acc", "majority acc"])

        # k-only rule: majority label per k (picked on this same file)
        rule = {k: summary["by_k"][k]["majority_label"] for k in all_k}
        rule_pred = [rule[k] for k in ks]
        summary["k_rule"] = {
            "mapping": rule,
            "macro_f1": macro_f1(gold, rule_pred),
            "accuracy": accuracy(gold, rule_pred),
            "per_class_f1": {c: v["f1"] for c, v in per_class(gold, rule_pred, LABELS).items()},
            "model_minus_rule": paired_bootstrap(gold, pred, rule_pred),
        }
        kr = summary["k_rule"]
        print("== k-only rule (majority label per k, chosen on this file)")
        print("rule:", ", ".join(f"k={k} -> {v}" for k, v in rule.items()))
        print(f"macro F1 = {kr['macro_f1']:.4f}   accuracy = {kr['accuracy']:.4f}")
        mb = kr["model_minus_rule"]
        print(f"model - rule: {mb['diff']:+.4f}  95% CI [{mb['ci95'][0]:+.3f}, {mb['ci95'][1]:+.3f}]  p = {mb['p']:.3f}\n")

        # do the policy features still carry k?
        if all(len(x["features"]) >= 2 for x in res):
            print("== Features 0 and 1 equal to 0, by k")
            summary["zero_features_by_k"] = {}
            rows = []
            for k in all_k:
                idx = [j for j, kk in enumerate(ks) if kk == k]
                z0 = sum(res[j]["features"][0] == 0 for j in idx)
                z1 = sum(res[j]["features"][1] == 0 for j in idx)
                summary["zero_features_by_k"][k] = {"n": len(idx), "f0_zero": z0, "f1_zero": z1}
                rows.append([k, len(idx), z0, z1])
            print_table(rows, ["k", "n", "feature0 == 0", "feature1 == 0"])

    # comparison with another run
    if args.compare:
        other = load_results(args.compare)
        omap = {x["id"]: x["predicted_action"] for x in other}
        common = [j for j, i in enumerate(ids) if i in omap]
        g = [gold[j] for j in common]
        pa = [pred[j] for j in common]
        pb = [omap[ids[j]] for j in common]
        pca, pcb = per_class(g, pa, LABELS), per_class(g, pb, LABELS)
        summary["compare"] = {
            "compare_file": args.compare,
            "n_common": len(common),
            "this": {"macro_f1": macro_f1(g, pa), "accuracy": accuracy(g, pa), "per_class_f1": {c: v["f1"] for c, v in pca.items()}},
            "other": {"macro_f1": macro_f1(g, pb), "accuracy": accuracy(g, pb), "per_class_f1": {c: v["f1"] for c, v in pcb.items()}},
            "agreement": accuracy(pa, pb),
            "this_minus_other": paired_bootstrap(g, pa, pb),
        }
        cmp_ = summary["compare"]
        print(f"== This run vs {args.compare} (n={len(common)})")
        rows = [["macro F1", f"{cmp_['this']['macro_f1']:.4f}", f"{cmp_['other']['macro_f1']:.4f}"],
                ["accuracy", f"{cmp_['this']['accuracy']:.4f}", f"{cmp_['other']['accuracy']:.4f}"]]
        rows += [[f"F1 {c}", f"{cmp_['this']['per_class_f1'][c]:.3f}", f"{cmp_['other']['per_class_f1'][c]:.3f}"] for c in LABELS]
        print_table(rows, ["metric", "this", "other"])
        tb = cmp_["this_minus_other"]
        print(f"this - other: {tb['diff']:+.4f}  95% CI [{tb['ci95'][0]:+.3f}, {tb['ci95'][1]:+.3f}]  p = {tb['p']:.3f}")
        print(f"same prediction on {cmp_['agreement']:.1%} of examples\n")

    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        print(f"Saved to {args.out}")


if __name__ == "__main__":
    main()
