# NLI Contradiction Bug + Grid Search Fix

## What was wrong

1. **NLI contradiction feature read the wrong class label.**  
   `framework/feature_extractor.py` looked for the label `"CONTRADICTION"`. The NLI model's labels are actually lowercase (`contradiction`, `entailment`, `neutral`), so the lookup failed and the code silently fell back to index 2 — which is **neutral**, not contradiction. Every previous run, dev and test, fed the policy P(neutral). On a clearly contradictory pair, contradiction is ~0.998 while neutral is ~0.002, so the feature was close to the opposite of what the paper claims.

2. **Grid search never extracted the answer span.**  
   `development/joint_grid_search.py` used `config.DEVICE` and `torch` without importing them. A bare `except:` swallowed the `NameError`, so claim support was computed from similarity between the **question** and passages instead of between the **extracted answer span** and passages. The test-time extractor did extract the span, so τ_s was tuned on the wrong quantity.

3. **Consequences.**  
   - The locked hyperparameters (τ_s = 0.6, TF-IDF = 0.09, C = 10, θ = 0.50) were tuned on the wrong features.  
   - `models/policy_dev_set_natural.joblib` learned weights for feature 2 as if it meant "neutral", so it must be retrained.  
   - The ablation script's results are only valid after the grid search is rerun.

## What was fixed

- `framework/feature_extractor.py`: contradiction index is now resolved case-insensitively against the model's own `id2label`, and raises if not found. Extraction is split into `extract_raw()` (threshold-free) and `features_from_raw()` (applies τ_s / TF-IDF after), so tuning and test inference compute identical features.
- `development/joint_grid_search.py`: rewritten to use the same shared `FeatureExtractor.extract_raw()` as test-time inference; answer span is actually extracted during tuning.
- `framework/policy.py`: `make_policy_model()` uses `LogisticRegression(multi_class='ovr', ...)`; scikit-learn pinned to `==1.5.1` in `requirements.txt`.
- `framework/config.py`: TF-IDF threshold corrected to 0.09 (code had 0.05), θ set to 0.50.

## Current state

The fixes are uploaded to the server at `103.73.232.201:62161` (`~/conflictRAG_server/`), but **nothing has been run yet**. `log.jsonl` is unchanged.

## Your job

1. **Run the grid search** (it caches dev features, then logs the best values to `log.jsonl`; it does not change `config.py`):
   ```bash
   cd ~/conflictRAG_server && nohup python3 development/joint_grid_search.py > results/grid_search_run.log 2>&1 &
   ```

2. **Decide whether to adopt the new hyperparameters.** This is a methodology change — you approve it before any values are written into `config.py` or results.

3. **Retrain the policy** on `dev_set_natural` with the chosen hyperparameters (note: `train_policy.py` still trains on `dev_set_1`, which overlaps the test set; fix that before retraining).

4. **Run the ablation:**
   ```bash
   nohup python3 development/ablation_study.py > results/ablation_run.log 2>&1 &
   ```

5. **Download results:** `ablation_results.json`, `log.jsonl`, `grid_search_run.log`, `ablation_run.log`.

Old files are backed up locally in `backups/2026-09-26_ablation_fix/`; changes are recorded in `CHANGE_LOG.md`.
