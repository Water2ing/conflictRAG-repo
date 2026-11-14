"""
Runs the trained 4D policy over the test set and saves one record per example to
config.RESULTS_PATH (results/test_set_1_4D_results.json).

Generation is OFF by default (config.GENERATE = False): the study evaluates the routing
decision only. With generation off, non-ABSTAIN examples get generated_text = SKIPPED_TEXT.
"""
import os
import json
import sys

import numpy as np
import requests
from sklearn.metrics import f1_score

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(ROOT_DIR)

from framework.feature_extractor import FeatureExtractor
from framework.policy import PolicyLayer
from framework.logger import log_run
from framework import config

LABELS = ["ABSTAIN", "ANSWER", "DISCLAIMER", "DISCLOSE"]
ABSTAIN_TEXT = "I cannot answer based on the provided passages."
SKIPPED_TEXT = "SKIPPED (generation disabled: routing-only run)"

PROMPTS = {
    "ANSWER": (
        "You are a factual question-answering assistant. Use ONLY the following passages to answer the question.",
        "Passages: {context}\n\nQuestion: {question}\n\nAnswer concisely:",
    ),
    "DISCLAIMER": (
        "You are a cautious assistant. The evidence is ambiguous. Provide a hedged answer based on the passages.",
        "Based on the available evidence, which may be incomplete:\nPassages: {context}\n\nQuestion: {question}\n\nAnswer:",
    ),
    "DISCLOSE": (
        "You are an analytical assistant. The retrieved sources disagree. Disclose the conflict to the user based on the passages.",
        "The retrieved sources disagree on this point.\nPassages: {context}\n\nQuestion: {question}\n\nAnswer:",
    ),
}


class ResponseGenerator:
    """Llama 3.1 via an OpenAI-compatible llama.cpp server. Used only when config.GENERATE."""

    def __init__(self):
        self.api_url = config.LLM_API_URL

    def generate(self, system_prompt, user_prompt):
        payload = {
            "model": "llama3",  # Ignored by local server, but required by schema
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "temperature": 0.0,
            "max_tokens": 256,
        }
        response = requests.post(self.api_url, json=payload, timeout=60)
        response.raise_for_status()
        return response.json()["choices"][0]["message"]["content"].strip()


def main():
    print("Initializing pipeline...")
    extractor = FeatureExtractor()
    policy = PolicyLayer()  # Loads config.POLICY_MODEL_PATH
    generator = ResponseGenerator() if config.GENERATE else None
    print(f"Generation: {'ON (' + config.LLM_API_URL + ')' if config.GENERATE else 'OFF (routing only)'}")

    with open(config.DATA_PATH, "r", encoding="utf-8") as f:
        data = [json.loads(line) for line in f]
    print(f"\nRouting {len(data)} examples...\n")

    results = []
    for i, item in enumerate(data):
        question, passages = item["question"], item["passages"]
        features = extractor.extract(question, passages)   # 4D [D, C-hat, S, L]
        predicted_action = policy.decide(features)          # logistic regression + theta gate

        if predicted_action == "ABSTAIN":
            generated_text = ABSTAIN_TEXT
        elif generator is None:
            generated_text = SKIPPED_TEXT
        else:
            sys_prompt, user_tmpl = PROMPTS[predicted_action]
            user_prompt = user_tmpl.format(context="\n\n".join(passages), question=question)
            generated_text = generator.generate(sys_prompt, user_prompt)

        results.append({
            "id": item["id"],
            "question": question,
            "gold_action": item["gold_action"],
            "predicted_action": predicted_action,
            "features": features.tolist(),
            "generated_text": generated_text,
            "gold_answer": item.get("gold_answer", ""),
        })
        if (i + 1) % 250 == 0:
            print(f"  {i + 1}/{len(data)}")

    os.makedirs(os.path.dirname(config.RESULTS_PATH), exist_ok=True)
    with open(config.RESULTS_PATH, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    y = np.array([r["gold_action"] for r in results])
    p = np.array([r["predicted_action"] for r in results])
    macro_f1 = float(f1_score(y, p, labels=LABELS, average="macro", zero_division=0))
    accuracy = float(np.mean(y == p))
    print(f"\nTest macro-F1 = {macro_f1:.4f}, accuracy = {accuracy:.4f}")
    print(f"Results saved to {config.RESULTS_PATH}")

    log_run(
        run_name=f"Test Set Evaluation ({config.RUN_TAG})",
        data_path=config.DATA_PATH,
        parameters={"tau_s": config.TAU_S, "tfidf": config.TFIDF_THRESH, "C": config.C,
                    "theta": config.THETA, "generate": config.GENERATE},
        metrics={"macro_f1": macro_f1, "accuracy": accuracy},
    )


if __name__ == "__main__":
    main()
