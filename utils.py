import numpy as np


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
