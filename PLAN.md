```markdown
## 6. Immediate Objective: The Ablation Study
The current task is to design and execute a rigorous ablation study to prove the necessity of each feature in the 6D vector.

### Ablation Design (Leave-One-Out)
For each variant, we must:
1. Physically drop the feature column from the dev set (`X_dev`).
2. Retrain the Logistic Regression model on the 5D vector.
3. Re-tune the $\theta$ threshold on the 5D dev set.
4. Drop the corresponding feature from the test set (`X_test`).
5. Evaluate on the reduced test set.
6. Compare the Macro F1 and Per-Class F1 against the Full Model (6D).

### Expected Variants
* `w/o Dispersion` (Feature 1)
* `w/o NLI Contradiction` (Feature 2) -> Expected to catastrophically drop `DISCLOSE` F1.
* `w/o Claim Support` (Feature 3) -> Expected to drop `ANSWER` F1.
* `w/o Mismatch` (Feature 4) -> Expected to drop `ABSTAIN` F1.
* `w/o Theta Threshold` -> Use standard `argmax` instead of the 0.50 gate. Expected to drop minority class recall.

### Script Requirements
The AI agent needs to write a script (`development/run_ablation.py`) that automates this entire pipeline. The script should output a clean comparison table showing the Full Model vs. the Ablated Models, including the $\Delta$ F1.
```
