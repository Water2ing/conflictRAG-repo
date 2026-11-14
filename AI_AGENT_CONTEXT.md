```markdown
# AI Agent Context: ConflictRAG Research Project

## 1. Project Overview
**ConflictRAG** is a training-free, CPU-deployable policy layer for Retrieval-Augmented Generation (RAG) systems. It sits between the retriever and the generator. Instead of blindly generating answers over flawed retrieval contexts, it extracts a 6-dimensional mathematical feature vector from the retrieved passages and uses a Logistic Regression policy layer to route the interaction into one of four structured actions: `ANSWER`, `DISCLOSE`, `DISCLAIMER`, or `ABSTAIN`.

This research is being prepared for submission to an A/A* NLP venue (e.g., NAACL). Rigor, reproducibility, and prevention of data leakage are the absolute highest priorities.

## 2. System Architecture & Locked Hyperparameters
The framework uses a 6D feature vector $\phi \in \mathbb{R}^6$. The hyperparameters have been rigorously tuned via a Joint Grid Search with 5-Fold Cross-Validation on a natural-distribution dev set.

### The 6D Feature Extractor (`framework/feature_extractor.py`)
1. **Retrieval Dispersion ($D$):** $1 - \text{mean}(\text{cosine\_sim}(passages))$. Measures semantic disagreement.
2. **Lexical Contradiction ($\hat{C}$):** $\max(\text{NLI\_Contradiction\_Prob}(pairs))$. Detects inter-document conflicts using `cross-encoder/nli-MiniLM2-L6-H768`.
3. **Semantic Claim Support ($S$):** Fraction of passages whose embedding similarity to the extracted answer span $> \tau_s$. Uses `deepset/minilm-uncased-squad2` for span extraction.
4. **Language/Domain Mismatch ($L$):** Binary flag (1 if language differs or TF-IDF overlap $< \text{thresh}$).
5. **Passage Count ($k$):** Normalized by 10.0.
6. **Query Length:** Normalized by 20.0.

### The Policy Layer (`framework/policy.py`)
* A Logistic Regression model (`scikit-learn`), trained with One-vs-Rest (`multi_class='ovr'`).
* **Confidence Gate ($\theta$):** If $P(\text{ANSWER}) < \theta$, the system escalates to the safest alternative action.

### Locked Optimal Hyperparameters (DO NOT CHANGE)
* **$\tau_s$ (Claim Support):** `0.60`
* **TF-IDF (Domain Mismatch):** `0.09`
* **$C$ (Regularization):** `10.0`
* **$\theta$ (Abstention Threshold):** `0.50`

## 3. Data Pipeline & Leakage Prevention Firewalls
The data strategy is strictly partitioned to guarantee zero overlap between dev (training/tuning) and test (evaluation).

### Dev Set: `data/dev_set_natural.jsonl`
* **Size:** ~1,000 examples.
* **Distribution:** Aligned with test set (~52% ANSWER, 24% DISCLOSE, 14% ABSTAIN, 10% DISCLAIMER).
* **Sources:** RGB (ABSTAIN), ConflictQA (DISCLOSE), FaithDial `valid` split (ANSWER/DISCLAIMER), PubMedQA `train` split (ANSWER/DISCLAIMER).
* **Usage:** Used to train the Logistic Regression model and tune $\theta, \tau_s, \text{TF-IDF}, C$.

### Test Set: `data/test_set_1.jsonl`
* **Size:** 3,073 examples.
* **Distribution:** Natural distribution.
* **Sources:** RGB, NoMIRACL (ABSTAIN), ConflictQA, WhoQA (DISCLOSE), FaithDial `test` split, PubMedQA `train` split (indices 300+).
* **Usage:** Strictly held out for final evaluation. No tuning allowed on this data.

## 4. Baselines
1. **Vanilla LLM (Zero-Shot Routing):** Prompts Llama 3.1 8B to classify the context into one of the 4 actions without any mathematical features.
2. **Static-CRAG:** Uses the fine-tuned T5-large evaluator to score passage relevance (-1.0 to 1.0). Maps CRAG states (CORRECT, INCORRECT, AMBIGUOUS) to the 4 actions. Web search is ablated (disabled).

## 5. Strict Rules for AI Agent
When writing code or modifying the pipeline, the AI agent must adhere to the following:
1. **No Data Leakage:** Never read `test_set_1.jsonl` during any training, feature extraction, or hyperparameter tuning step. Only the dev set is used for optimization.
2. **Rigorous Ablation:** If an ablation study is requested, the AI must retrain the Logistic Regression model on the reduced feature set. Do not simply mask features in the pre-trained model.
3. **Device Compatibility:** The framework is designed to be "CPU-deployable" for the paper's narrative, but runs on an H100 GPU during server execution. Ensure PyTorch tensors are properly moved to `config.DEVICE`.
4. **Reproducibility:** Always set `random_state=42` for all `scikit-learn` operations.
```
