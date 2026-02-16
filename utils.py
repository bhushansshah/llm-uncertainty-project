import os
import json
import glob

import numpy as np
from sklearn.metrics import roc_auc_score


def load_results(outputs_dir, dataset, model):
    """
    Load all result JSON files for a given dataset and model.

    Args:
        outputs_dir (str): Path to the top-level outputs directory.
        dataset (str): Dataset name (subfolder of outputs_dir).
        model (str): Model name (subfolder of dataset folder).

    Returns:
        list[dict]: List of parsed result dictionaries, sorted by index.
    """
    model_dir = os.path.join(outputs_dir, dataset, model)
    pattern = os.path.join(model_dir, "result_*.json")
    files = sorted(glob.glob(pattern), key=lambda f: int(os.path.basename(f).split("_")[1].split(".")[0]))

    decoder = json.JSONDecoder()
    results = []
    for fp in files:
        with open(fp, "r") as f:
            raw = f.read()
        obj, _ = decoder.raw_decode(raw)
        results.append(obj)
    return results

def get_data_config(outputs_dir, dataset, model):
    """
    Get the data configuration for a given dataset and model.
    Args:
        outputs_dir (str): Path to the top-level outputs directory.
        dataset (str): Dataset name (subfolder of outputs_dir).
        model (str): Model name (subfolder of dataset folder).
    Returns:
        dict: The data configuration.
    """
    model_dir = os.path.join(outputs_dir, dataset, model)
    config_path = os.path.join(model_dir, "config.json")
    with open(config_path, "r") as f:
        config = json.load(f)
    return config

def normalize(logits):
    exp_logits = np.exp(logits - np.max(logits))
    return exp_logits / exp_logits.sum()

def compute_entropy(probs):
    return -np.sum(probs * np.log(probs + 1e-10))

def token_entropy(token_logprobs_dict):
    token_logprobs = list(token_logprobs_dict.values())
    # logprobs: top-10 log probabilities (natural log)
    ps = np.exp(token_logprobs)
    m = ps.sum()
    r = max(0.0, 1.0 - m)

    H = compute_entropy(ps)
    if r > 0:
        H -= r * np.log(r)

    return H


def get_reasoning_steps(text):
    if '</think>' in text:
        reasoning_content = text.split('</think')[0]
    else:
        reasoning_content = text

    steps = reasoning_content.split('\n\n')
    steps = [step for step in steps if step.strip() != '']
    return steps

def get_step_start_end_indices(response, step_number):
    step_markers = [0]
    for i in range(len(response['logprobs']['tokens'])):
        token = response['logprobs']['tokens'][i]
        if '\n\n' in token:
            step_markers.append(i)
            if len(step_markers) > step_number:
                break
    else:
        return step_markers[-1], len(response['logprobs']['tokens']) - 1
    
    step_start, step_end = step_markers[-2], step_markers[-1]
    return step_start, step_end

def compute_auroc(scores, labels):
    """
    Compute AUROC using negative average log-probability as the
    uncertainty score and correctness as the binary label.

    Higher negative avg log-prob → more uncertain → more likely wrong.
    AUROC measures how well this score separates correct from incorrect.

    Args:
        scores (list[float]): Negative average log-probabilities.
        labels (list[int]): Correctness labels (0 or 1).

    Returns:
        float: AUROC score.
    """
    # AUROC: score predicts *incorrectness*, so we flip labels
    # (1 = incorrect) or equivalently use 1 - labels as positive class.
    # Convention: higher score → more likely incorrect.
    scores = np.array(scores)
    labels = np.array(labels)
    # flip labels so that higher score → more likely incorrect
    labels = 1 - labels
    auroc = roc_auc_score(labels, scores)
    return auroc

def correct_space_tokens(tokens):
    """
    Correct the space tokens in the tokens list.
    """
    result = []
    for token in tokens:
        if len(token) >= 1 and token[0] == "Ġ":
            result.append(" " + token[1:])
        else:
            result.append(token)
    return result
