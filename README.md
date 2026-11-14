# Response-Policy Layer for RAG (working name: ConflictRAG)

Given the passages a RAG retriever returns, decide how the system should respond:

| Action | Meaning |
|---|---|
| `ANSWER` | evidence is sufficient and consistent |
| `DISCLAIMER` | evidence is partial or uncertain |
| `DISCLOSE` | retrieved sources contradict each other |
| `ABSTAIN` | evidence is irrelevant |

The policy computes four signals from the retrieved passages and feeds them to a one-vs-rest
logistic regression with a confidence gate θ on `P(ANSWER)`:

| Feature | What it measures |
|---|---|
| dispersion | 1 − mean pairwise cosine similarity of the passages (MiniLM embeddings) |
| NLI contradiction | max P(contradiction) over passage pairs (`cross-encoder/nli-MiniLM2-L6-H768`) |
| claim support | share of passages similar to the extracted answer span, threshold τ_s (`deepset/minilm-uncased-squad2`) |
| mismatch | 1 if the question and passages differ in language or TF-IDF overlap < threshold |

**Scope:** the study evaluates the routing decision only. Response generation with Llama 3.1 is
deliberately switched off (`config.GENERATE = False`).

The name **ConflictRAG** collides with Wang et al., *ConflictRAG: Detecting and Resolving
Knowledge Conflicts in RAG*, arXiv:2605.17301 (May 2026). The method needs a new name before
submission.

## Current status (4D run, 31 Oct 2026)

Test set: 3,073 examples. Full table: `python development/evaluate.py`.

| System | Macro-F1 | Accuracy |
|---|---|---|
| **4D policy** | **0.602** | **0.735** |
| 6D policy (previous version, with passage count + query length) | 0.642 | 0.725 |
| Vanilla LLM router (Llama 3.1 8B, zero-shot) | 0.195 | 0.304 |
| Static-CRAG (fine-tuned T5 evaluator) | 0.180 | 0.403 |
| Control: rule using passage count only | 0.665 | 0.847 |

**Open problem:** the policy loses to the passage-count rule. In this benchmark each source
dataset contributes essentially one action and one passage count, so passage count predicts the
label. Details and the plan are in `docs/ConflictRAG_Progress_Report.pdf` (overview) and
`docs/4D_run_report.pdf` (technical).

## Repository layout

```
framework/      the method
  config.py           all paths and the locked hyperparameters (single source of truth)
  feature_extractor.py  4D features; extract_raw() is threshold-free, features_from_raw() applies τ_s / TF-IDF
  feature_cache.py    caches raw features with a fingerprint (data hash + extractor version); rebuilds if stale
  policy.py           model factory, θ-gated prediction, PolicyLayer (refuses a model with the wrong feature count)
  logger.py           appends every run to results/log.jsonl
  main.py             routes the test set with the trained policy -> results/test_set_1_4D_results.json
development/    experiments (run from the repo root)
  joint_grid_search.py  tunes τ_s, TF-IDF, C, θ on dev with 5-fold CV (--tfidf-ext for the extended TF-IDF grid)
  train_policy.py     trains the final policy on dev -> models/policy_dev_set_natural_4D.joblib
  ablation_study.py   leave-one-feature-out, θ re-tuned per variant, paired bootstrap vs. full model
  evaluate.py         compares saved result files: baselines, controls, per-class, per-source (CPU, seconds)
baselines/
  baseline_vanilla.py   zero-shot Llama 3.1 router (needs llama.cpp server on port 8080)
  baseline_crag_v5.py   Static-CRAG: fine-tuned T5 evaluator, web search disabled (needs local_models/checkpoint-7059 + server on 8080)
data/
  dev_set_natural.jsonl   1,000 examples, used for ALL tuning and training
  test_set_1.jsonl        3,073 examples, used only for final evaluation
  generate_dev_natural.py / test_set_1.py   rebuild the two sets from the public datasets (Hugging Face)
  cache/                  raw-feature caches written by the scripts (see "Reproducing")
models/         trained policy (policy_dev_set_natural_4D.joblib)
results/        result files, see below
logs/           console logs and pip freeze of the 31 Oct 2026 server run
docs/           progress deck and technical report (PDF + LaTeX source)
```

### Datasets

| Source | Dev | Test | Gold actions | Passages |
|---|---|---|---|---|
| ConflictQA | 240 | 500 | DISCLOSE | 2 |
| RGB | 140 | 100 | ABSTAIN | 2 |
| FaithDial | 360 (`valid`) | 1,272 (`test`) | ANSWER / DISCLAIMER | 1 |
| PubMedQA | 260 (indices < 300) | 559 (indices ≥ 300) | ANSWER / DISCLAIMER | 1 |
| WhoQA | – | 300 | ANSWER / DISCLOSE | 2 |
| NoMIRACL | – | 342 | ABSTAIN | 3 |

WhoQA and NoMIRACL never appear in dev (unseen sources). Dev and test share no example IDs and no
questions. DISCLAIMER labels come from FaithDial's "Generic/Uncooperative" response tags and
PubMedQA's "maybe" decisions.

### Results folder

| File | Contents |
|---|---|
| `test_set_1_4D_results.json` | **current method**: per-example prediction and features on the test set |
| `test_set_1_4D_summary.json` | output of `development/evaluate.py --save` (main comparison table) |
| `test_set_1_4D_eval.json` | confusion matrix, per-k breakdown, k-only rule (from the 31 Oct run) |
| `test_set_1_4D_ablation_results.json` | ablation: every variant with θ, ΔF1, 95% CI, per-source and per-k accuracy |
| `test_set_1_4D_grid_search_full.json` | all 18,480 grid configurations (dev, 5-fold mean ± std) |
| `test_set_1_4D_grid_search_tfidf_ext.json` | extended TF-IDF grid (15,120 configurations) |
| `test_set_1_results_final.json` | previous 6D policy, kept as the 6D reference |
| `vanilla_test_set_1_results.json` | Vanilla LLM router baseline |
| `crag_test_set_1_results.json` | Static-CRAG baseline |
| `log.jsonl` | one line per run (parameters + metrics) |

## Locked hyperparameters (`framework/config.py`)

τ_s = 0.3, TF-IDF = 0.20, C = 1.0, θ = 0.40, `random_state` = 42. Chosen by
`joint_grid_search.py` on dev only; the extended TF-IDF grid confirmed the optimum.

## Reproducing

```bash
python3.10 -m venv venv && source venv/bin/activate
pip install -r requirements.txt

python development/evaluate.py                 # CPU only: reproduces the results table from results/*.json
python development/joint_grid_search.py        # GPU recommended (feature extraction)
python development/train_policy.py
python development/ablation_study.py
python framework/main.py
```

- Set `DEVICE = "cpu"` in `framework/config.py` if there is no GPU (feature extraction is slower).
- The first run extracts raw features for dev and test (≈4,000 examples) and caches them in
  `data/cache/`; later runs reuse the cache. Features match to ~1e-6 between runs, except that
  language detection was unseeded before 14 Nov 2026 (now seeded) and claim support can differ
  on a few examples between CPU and GPU.
- scikit-learn must be < 1.8 (`multi_class="ovr"` was removed in 1.8).
- Rules followed throughout: the test set is never used for tuning; ablations retrain and re-tune
  θ per variant; all randomness uses seed 42; every run appends to `results/log.jsonl`.

## Key history

- **Contradiction bug fixed.** The NLI feature read P(neutral) instead of P(contradiction)
  because of a label-case mismatch; the grid search also never extracted the answer span. All
  results before the fix were discarded.
- **6D → 4D.** Passage count and query length were removed as inputs because they predicted the
  label on their own. Two of the remaining features are always zero when only one passage is
  retrieved, so the policy can still partly infer passage count.
- **dev_set_1 retired.** It shared 59 questions with the test set; only `dev_set_natural` is used.
