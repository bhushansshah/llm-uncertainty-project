import os
import json

from utils import compute_auroc
RESULTS_DIR = "experiments/DiagnosisArena/qwen_qwen3-32b"


def load_results(results_dir: str):
    """Load all result_*.json files from the given directory."""
    results = []
    for filename in sorted(os.listdir(results_dir)):
        if not (filename.startswith("result_") and filename.endswith(".json")):
            continue
        path = os.path.join(results_dir, filename)
        with open(path, "r") as f:
            data = json.load(f)
        results.append(data)
    return results

def extract_step_confidences(text):
    """Extract step confidences from the text."""
    if not text:
        return []

    lowered = text.lower()
    needle = "confidence: "
    confidence_ids = []
    start = 0

    # Find all occurrences of "confidence:"
    while True:
        idx = lowered.find(needle, start)
        if idx == -1:
            break
        confidence_ids.append(idx)
        start = idx + len(needle)

    confidence_texts = []
    for idx in confidence_ids:
        t1 = text[idx + len(needle):].split("\n")[0].strip()
        t2 = text[idx + len(needle):].split(".")[0].strip()
        if len(t1) > len(t2):
            confidence_texts.append(t2)
        else:
            confidence_texts.append(t1)
    print(confidence_texts)
    confidence_classes = [get_confidence_class(conf_text) for conf_text in confidence_texts]
    print(confidence_classes)
    return confidence_classes

def get_confidence_class(confidence_text):
    """
    Get the confidence class from the confidence text.
    """
    confidence_classes_map = {
        "Almost no chance": 0.05,
        "Highly unlikely": 0.15,
        "Chances are slight": 0.25,
        "Unlikely": 0.35,
        "Less than even": 0.45,
        "Better than even": 0.55,
        "Likely": 0.65,
        "Very good chance": 0.75,
        "Highly likely": 0.85,
        "Almost certain": 0.95,
    }
    return confidence_classes_map.get(confidence_text, None)

if __name__ == "__main__":
    # Assume this script is in `scripts/` and project root is its parent
    base_dir = os.path.dirname(os.path.abspath(__file__))
    full_results_dir = os.path.join(base_dir, RESULTS_DIR)
    all_results = load_results(full_results_dir)
    uncertianty_scores = []
    is_correct = []
    for result in all_results:
        reasoning = result.get("response").get("message").get("reasoning")
        extracted_step_confidences = extract_step_confidences(reasoning)
        confidence = result.get("extracted_confidence")
        # count = 1
        # for conf in extracted_step_confidences:
        #     if confidence is None:
        #         continue
        #     confidence += conf
        #     count += 1
        # confidence /= count
        uncertianty_scores.append(1 - confidence)
        is_correct.append(1 if result.get("is_correct") else 0)

    auroc = compute_auroc(uncertianty_scores, is_correct)
    print(f"AUROC = {auroc:.4f}")

    print(f"Loaded {len(all_results)} results from {full_results_dir}")
