# CHANGE_LOG

## 2026-11-12 (b) — Professor overview + generation wording fix

- New: `Progress_overview.tex` / `.pdf` (2 pages), a brief progress summary for the professor.
- Edited `4D_run_report.tex` / `.pdf` (backup: `backups/2026-11-12_report_fix/4D_run_report.tex`):
  `GENERATION_ERROR` is now described as intentional (Llama 3.1 deliberately not connected; the
  study evaluates routing only), instead of as a server failure. The "fail loudly when the LLM
  server is unreachable" step became "add a routing-only mode", and end-to-end generation was
  removed from the analysis list.
- No code, data or results changed.

## 2026-11-12 — Added `4D_run_report.tex` / `.pdf` (3 pages)

New files in `/home/khangiguess/conflictrag/`. No code, data, config or result file was changed.
The report summarises the 31 Oct 4D server run from `logs/` and `results/test_set_1_4D_*`.
Baseline, k-only rule, AUROC, feature-mean and seen/unseen figures were recomputed in memory
from the saved per-example predictions; nothing was written to `results/` or `log.jsonl`.
Open item recorded in the report: two 6D reference files disagree
(`test_set_1_results_final.json` = 0.642 macro-F1, root `results/test_set_1_results.json` = 0.633).

## 2026-10-25 (c) — 6D → 4D policy (remove k and query length). Run tag `test_set_1_4D`

**Why:** in the 6D ablation (`results/log.jsonl`, 2026-09-27), "k + len only" scored
test macro-F1 0.641 vs. 0.632 for the full model. The policy was identifying the source
dataset from passage count and query length rather than using the retrieval signals.
Requested by the user (methodology change approved).

**Backups:** `backups/2026-10-25_4D/` (`framework/{config,feature_extractor,policy,main}.py`,
`development/{joint_grid_search,ablation_study,train_policy}.py`).
6D artifacts (`models/policy_dev_set_natural.joblib`, `results/ablation_results.json`,
`results/test_set_1_results*.json`) are left untouched.

| File | Change |
|---|---|
| `framework/config.py` | `RUN_TAG = "test_set_1_4D"`. `RESULTS_PATH` → `results/test_set_1_4D_results.json`. New `POLICY_MODEL_PATH` → `models/policy_dev_set_natural_4D.joblib`. New shared `THETA_GRID` (0.20–0.90, step 0.05). |
| `framework/feature_extractor.py` | `FEATURE_NAMES` / `features_from_raw()` → 4D `[dispersion, nli_contradiction, claim_support, mismatch]`. New `N_FEATURES`. New `shortcut_features_from_raw()` (k/10, len/20), used only for the ablation's shortcut control. `extract_raw()` unchanged, so `EXTRACTOR_VERSION` unchanged and existing raw caches stay valid (no re-extraction). |
| `framework/policy.py` | Loads `config.POLICY_MODEL_PATH`. Raises if the model's `n_features_in_` != 4, so a 6D model can't be loaded silently. |
| `framework/main.py` | Comments only (4D, config theta). Output goes to `test_set_1_4D_results.json` via config. |
| `development/joint_grid_search.py` | Expanded grid: tau_s 0.05–0.80 (16 values), TF-IDF 0.02–0.20 (11), C 0.001–1000 (7), theta `THETA_GRID` (15); 18,480 configs. Warns if the optimum is on a grid edge. Prints top 10 with fold std. Saves every config to `results/test_set_1_4D_grid_search.json`. |
| `development/train_policy.py` | Already on `dev_set_natural.jsonl` + shared `FeatureExtractor` (2026-10-25 entry). Now saves to `models/policy_dev_set_natural_4D.joblib`; logs feature list and run tag. |
| `development/ablation_study.py` | 4D variants: full (locked θ), full (re-tuned θ, reference), w/o D, w/o C-hat, w/o S, w/o L, w/o θ gate, k+len only (shortcut control). Uses `THETA_GRID`. Output → `results/test_set_1_4D_ablation_results.json`. Log run names tagged `test_set_1_4D`. |

**Hyperparameters:** `config.py` still holds the 6D-tuned values (tau_s 0.2, TF-IDF 0.10, C 10,
θ 0.50) as placeholders. They must be re-tuned for 4D with the expanded grid.

**Verified:** all files compile. End-to-end run on **synthetic** features (output in /tmp):
- grid search → train_policy → PolicyLayer, and the 6D-model guard raises correctly.
- ablation: all 8 variants, table, bootstrap and JSON output work.

Repo `results/log.jsonl` unchanged (checksum verified). Not run on real data.

**Known residual shortcut (for the paper):** D and C-hat are exactly 0 when k=1 (as is
fractional S at k=1, which can only be 0 or 1), so a 4D model can still partly infer k.
The by-k / by-source breakdown in the ablation shows how much this matters.

## 2026-10-25 (b) — Hyperparameters re-tuned, `config.py` updated

**Backup:** `backups/2026-10-25_config_retune/framework/config.py`

Source: `results/log.jsonl` run "Joint Grid Search (shared extractor, fixed QA span + NLI
index)", 2026-10-23 07:48:59, mean 5-fold macro F1 = 0.6956. Confirmed against
`results/grid_search_run.log` (same run). `C` and `theta` unchanged.

| | old (buggy features) | new (fixed features) |
|---|---|---|
| tau_s | 0.6 | **0.2** |
| tfidf | 0.09 | **0.10** |

Caveat: tau_s=0.2 is the lower edge of the searched grid (0.20–0.80 in the committed script).
The true optimum may be below 0.2; consider extending the grid down to ~0.05 and re-running.

Not yet done: retrain `models/policy_dev_set_natural.joblib` with these values
(`development/train_policy.py`) and re-run `development/ablation_study.py` — both must be
re-run on the server (needs `scikit-learn==1.5.1` for `multi_class='ovr'`).

## 2026-10-25 — `train_policy.py` → dev_set_natural + shared extractor

**Backup:** `/home/khangiguess/conflictrag/backups/2026-09-27_train_policy/development/train_policy.py`

- Trains on `data/dev_set_natural.jsonl` (was `dev_set_1.jsonl`, which shares 59 questions with the test set).
- Saves to `models/policy_dev_set_natural.joblib`, the path `PolicyLayer` loads (was `policy_dev_set_1.joblib`).
- Uses the shared `FeatureExtractor.extract_raw()` + `features_from_raw()` and the same raw-feature cache
  (`data/cache/raw_dev_set_natural.json`) as `joint_grid_search.py` and `ablation_study.py`.
- Hyperparameters come from `framework/config.py` (`make_policy_model()`, `config.THETA`) instead of hardcoded `C=10.0` / `0.50`.
- Reports 5-fold out-of-fold macro-F1 (same folds as the grid search) as well as the in-sample score, which is labeled optimistic.
- Logs to `results/log.jsonl` via `log_run`.
- No longer writes `data/X_dev.npy` / `y_dev.npy`. Nothing in the current pipeline reads them;
  `abstention_threshold.py` used them but it is deleted in git.
- Verified: compiles; ran end-to-end on synthetic features with output redirected to /tmp.
  Not run on real data. `config.py` NOT changed (hyperparameter adoption pending, see below).

## 2026-09-26 (b) — Revert to `multi_class='ovr'`, pin scikit-learn

**Backups:** `/home/khangiguess/conflictrag/backups/2026-09-26_revert_ovr/` (`framework/policy.py`, `requirements.txt`).

- `framework/policy.py`: `make_policy_model()` again returns
  `LogisticRegression(multi_class='ovr', ...)` rather than `OneVsRestClassifier(...)`, as requested.
  This supersedes fix 5 in the entry below.
- `requirements.txt`: `scikit-learn` → `scikit-learn==1.5.1`. `multi_class` was removed in 1.8.
  1.5.1 is the version the existing `.joblib` policies were pickled with.
  The local venv (1.9.0) cannot run the ablation or grid search; they must run on the server
  (or in a venv with the pinned version).

## 2026-09-26 — Ablation study fixes and extractor bug fixes

**Backups (pre-change copies):** `/home/khangiguess/conflictrag/backups/2026-09-26_ablation_fix/`
(`development/ablation_study.py`, `development/joint_grid_search.py`, `framework/config.py`,
`framework/feature_extractor.py`, `framework/policy.py`, `framework/generator.py`).
Backups are outside the git repo so they are not pushed.

**Status:** code changed and smoke-tested only. The real ablation and grid search have NOT
been run. No results or `log.jsonl` entries were produced.

### Bugs fixed (affect previously reported numbers)

1. **NLI contradiction feature used the wrong class** — `framework/feature_extractor.py`.
   The model's labels are lowercase (`0: contradiction, 1: entailment, 2: neutral`). The
   exact match on `"CONTRADICTION"` failed and silently fell back to index 2 = **neutral**.
   Feature 2 has been P(neutral), not P(contradiction), in every previous run (dev and test).
   Fix: case-insensitive lookup that raises if the label is not found.
   Verified: old index = 2 ("neutral"); new index = 0. On "Eiffel Tower is in Paris" vs.
   "...in Berlin", P(contradiction) = 0.998 and P(neutral) = 0.002.
2. **Grid search never extracted the answer span** — `development/joint_grid_search.py`.
   It used `config.DEVICE` / `torch` without importing them. A bare `except:` swallowed the
   NameError, so claim support was computed from the question text. Fix: the grid search now
   uses the same `FeatureExtractor.extract_raw()` as test-time inference.
3. **`generator.py` import** — `import config` changed to `from framework import config`.
   Before, `import framework.generator` raised ModuleNotFoundError.
4. **Silent rule-based fallback** — `framework/policy.py`. A missing policy file used to fall
   back to hand-written rules. It now raises `FileNotFoundError`.
5. **scikit-learn 1.8+ incompatibility** — `LogisticRegression(multi_class='ovr')` raises
   TypeError in the local venv (1.9.0). Replaced with
   `OneVsRestClassifier(LogisticRegression(...))` (same OvR model). `requirements.txt`
   does not pin scikit-learn, so the server version may differ.

### Hyperparameter sync (repo ← log.jsonl best run on dev_set_natural)

`framework/config.py` now holds all locked values in one place:
`TAU_S=0.6`, `TFIDF_THRESH=0.09` (was hardcoded 0.05 in the extractor), `C=10.0`,
`THETA=0.50` (was hardcoded 0.45 in policy.py), `RANDOM_STATE=42`.
`policy.py` now loads `models/policy_dev_set_natural.joblib` (was `policy_dev_set_1.joblib`,
which is trained on the leaked dev_set_1).

**Caveat:** these locked values were tuned with bugs 1 and 2 present. They should be
re-derived by re-running `joint_grid_search.py`. That is a methodology decision; the script
logs the new optimum but does not modify `config.py`.
`models/policy_dev_set_natural.joblib` was also trained on buggy features and must be
retrained before any main-results run.

### Refactors

- `feature_extractor.py`: split into `extract_raw()` (threshold-independent) and
  `features_from_raw(raw, tau_s, tfidf_thresh)`. The grid search and ablation can sweep
  thresholds without re-running the models. `extract()` behaves as before.
  NLI pairs are now scored in one batched `predict` call.
  Bare `except:` replaced with narrow exceptions (`LangDetectException`, `ValueError`).
  QA-span failures now raise instead of silently using the question.
- `policy.py`: added `make_policy_model()` and `gated_predict()`, used by inference,
  grid search and ablation so the θ-gate logic exists in exactly one place.
- New `framework/feature_cache.py`: raw-feature cache keyed by a fingerprint
  (extractor version, data SHA-256, model names). It rebuilds automatically when stale.
  Cache files: `data/cache/raw_<dataset>.json`.

### `development/ablation_study.py` (rewritten)

- θ is tuned on **5-fold out-of-fold** dev probabilities (was: in-sample on training data),
  using the same θ grid as the joint grid search.
- Reference row = "Full Model (θ re-tuned)", produced by the same procedure as every
  ablated variant. "Full Model (locked θ=0.50)" is also reported.
- New variants: w/o k, w/o query length, retrieval signals only (w/o k+len),
  k+len only (shortcut control).
- ΔF1 against the reference, with paired-bootstrap 95% CI and p-value (B=1000, seed 42).
- Test accuracy broken down by source dataset and by passage count k.
- Macro-F1 is computed over all 4 labels (a never-predicted class counts as 0).
- Results saved to `results/ablation_results.json`; each variant appended to
  `results/log.jsonl` via `log_run`.
- The test set is used only for final evaluation. No selection is made on it.

### Verification performed

- All changed files compile. `import framework.generator` succeeds.
- Extractor run on one example per test source on CPU. The local GPU returned
  "No CUDA GPUs are available", so DEVICE was overridden in-process only; config is unchanged.
- End-to-end ablation run on **synthetic** features, with output redirected to /tmp:
  all 11 variants, the table, the bootstrap, the JSON output and 11 log entries work.
  No real results were produced, and the repo's `log.jsonl` is unchanged.

### Not changed / open

- `data/dev_set_1.jsonl` still shares 59 questions with the test set. Scripts that use it
  (`train_policy.py`, `tune_tau_s.py`, `tune_tfidf.py`) were not touched.
- `train_policy.py`, `tune_tau_s.py` and `tune_tfidf.py` still use `multi_class='ovr'`
  and will crash on scikit-learn ≥1.8.
- `development/abstention_threshold.py` shows as deleted in git (a pre-existing change,
  not made here).
- Root `/framework` copy has not been synced.
