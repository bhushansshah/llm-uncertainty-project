import argparse
import json
import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from box import Box
from tqdm import tqdm

def normalize(logits):
    exp_logits = np.exp(logits - np.max(logits))
    return exp_logits / exp_logits.sum()

def compute_entropy(probs):
    return -np.sum(probs * np.log(probs + 1e-10))

def entropy_with_tail_bucket(logprobs):
    # logprobs: top-10 log probabilities (natural log)
    ps = np.exp(logprobs)
    m = ps.sum()
    r = max(0.0, 1.0 - m)

    H = compute_entropy(ps)
    if r > 0:
        H -= r * np.log(r)

    return H

def get_reasoning_steps(choice):
    if '</think>' in choice.text:
        reasoning_content = choice.text.split('</think')[0]
    else:
        reasoning_content = choice.text

    steps = reasoning_content.split('\n\n')
    steps = [step for step in steps if step.strip() != '']
    return steps

def get_step_start_end_indices(response, step_number):
    step_markers = [0]
    for i in range(len(response.logprobs.tokens)):
        token = response.logprobs.tokens[i]
        if '\n\n' in token:
            step_markers.append(i)
            if len(step_markers) > step_number:
                break
    else:
        return step_markers[-1], len(response.logprobs.tokens) - 1
    
    step_start, step_end = step_markers[-2], step_markers[-1]
    return step_start, step_end

def get_volatility(token_entropies):
    changes = np.diff(token_entropies)
    volatility = np.std(changes)
    return volatility

def get_token_entropy(token_logprobs):
    probs = normalize(list(token_logprobs.values()))
    return compute_entropy(probs)

def get_token_entropy_with_tail_bucket(token_logprobs):
    logprobs = list(token_logprobs.values())
    return entropy_with_tail_bucket(np.array(logprobs))

def get_all_token_entropies(response, use_tail_bucket=False):
    token_entropies = []
    for token_info in response.logprobs.top_logprobs:
        if use_tail_bucket:
            token_uncertainty = get_token_entropy_with_tail_bucket(token_info)
        else:
            token_uncertainty = get_token_entropy(token_info)
        token_entropies.append(token_uncertainty)
    return token_entropies

def extract_answer(resp_choice):
    text = resp_choice.text
    if '</think>' in text:
        text = text.split('</think>')[-1]
    
    answer_id = text.lower().find('answer:')
    if answer_id == -1:
        return ""
    else:
        answer = text[answer_id + len('answer:'):].strip()
        return answer
    
import json
import re

def extract_two_key_json(text: str):
    candidates = re.finditer(
        r'\{.*?\}',
        text,
        flags=re.DOTALL
    )

    for m in candidates:
        try:
            obj = json.loads(m.group(0))
        except json.JSONDecodeError:
            continue

        if set(obj.keys()) == {"claim_true_or_false", "explanation"}:
            return obj

    raise ValueError("JSON with expected keys not found")

def extract_answer_json(resp_choice):
    text = resp_choice.text
    if '</think>' in text:
        text = text.split('</think>')[-1]
    try:
        answer_json = extract_two_key_json(text)
        return answer_json['claim_true_or_false']
    except ValueError:
        return ""

def is_correct_answer(answer, gold_option, gold_answer):
    gd = gold_option.lower()
    ans = answer.lower().strip()
    if ans == gd:
        return True
    elif ans == gold_answer.lower().strip():
        return True
    elif '*' in ans:
        cleaned_ans = ans.replace('*', '').strip().lower()
        if cleaned_ans == gd or cleaned_ans == gold_answer.lower().strip():
            return True
        if ':' in cleaned_ans:
            cleaned_ans = cleaned_ans.split(':')[-1].strip()
            if cleaned_ans == gd or cleaned_ans == gold_answer.lower().strip():
                return True
    return False

def get_accuracy(response, dataset_name):
    if dataset_name in ['gpqa', 'mmlupro']:
        answer = extract_answer(response.response)
        is_correct = is_correct_answer(answer, response.gold_option, response.gold_answer)
        return is_correct
    elif dataset_name == 'matteroffact':
        answer = extract_answer_json(response.response)
        is_correct = answer == response.gold_answer
        return is_correct
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")
    
def is_special_token(t: str) -> bool:
    t = t.strip()
    return (t.startswith("<|") and t.endswith("|>")) or (t.startswith("<") and t.endswith(">"))
import re

WORD_RE = re.compile(r"[A-Za-z0-9]+(?:[’']+[A-Za-z0-9]+)*")

def group_openai_logprob_tokens_into_words(tokens):
    """
    tokens: OpenAI-style logprobs.tokens (strings), in order.
    returns: [{"word": str, "token_indices": [i,...]}, ...]
    """
    # Build token spans in concatenated text
    spans = []
    pos = 0
    for t in tokens:
        start = pos
        pos += len(t)
        spans.append((start, pos))

    text = "".join(tokens)

    # Find word spans in that SAME text
    word_spans = [(m.start(), m.end()) for m in WORD_RE.finditer(text)]

    # Overlap tokens with word spans
    out = []
    ti = 0
    n = len(spans)

    for ws, we in word_spans:
        while ti < n and spans[ti][1] <= ws:
            ti += 1

        tj = ti
        group = []
        while tj < n and spans[tj][0] < we:
            ts, te = spans[tj]
            if te > ws and ts < we:
                group.append(tj)
            tj += 1

        out.append({"word": text[ws:we], "token_indices": group})

    return out

def group_tokens_into_words_ignore_special(tokens):
    kept_tokens = []
    kept_to_orig = []
    for i, t in enumerate(tokens):
        if not is_special_token(t):
            kept_to_orig.append(i)
            kept_tokens.append(t)

    groups = group_openai_logprob_tokens_into_words(kept_tokens)

    # Map indices back to original token list
    for g in groups:
        g["token_indices"] = [kept_to_orig[j] for j in g["token_indices"]]
    return groups

def get_word_uncertainties(response, use_tail_bucket=False):
    words = group_tokens_into_words_ignore_special(response.logprobs.tokens)
    words_uncertainties = []
    token_entropies = get_all_token_entropies(response, use_tail_bucket)
    for word in words:
        token_indices = word["token_indices"]
        word_uncertainty = np.mean([token_entropies[i] for i in token_indices])
        words_uncertainties.append(word_uncertainty)
    words_selves = [word["word"] for word in words]
    return words_uncertainties, words_selves

def get_word_uncertainties_first_token(response, use_tail_bucket=False):
    words = group_tokens_into_words_ignore_special(response.logprobs.tokens)
    words_uncertainties = []
    token_entropies = get_all_token_entropies(response,use_tail_bucket )
    for word in words:
        token_indices = word["token_indices"]
        word_uncertainty = token_entropies[token_indices[0]]
        words_uncertainties.append(word_uncertainty)
    words_selves = [word["word"] for word in words]
    return words_uncertainties, words_selves

def get_step_token_entropies(response, step_number, use_tail_bucket=False):
    step_start, step_end = get_step_start_end_indices(response, step_number)
    token_entropies = get_all_token_entropies(response, use_tail_bucket)
    step_token_entropies = token_entropies[step_start:step_end]
    return step_token_entropies

def get_step_word_uncertainties(response, step_number, use_tail_bucket=False):
    step_start, step_end = get_step_start_end_indices(response, step_number)
    step_words = group_tokens_into_words_ignore_special(response.logprobs.tokens[step_start:step_end])
    token_entropies = get_step_token_entropies(response, step_number, use_tail_bucket)

    step_word_uncertainties = []
    for word in step_words:
        token_indices = word["token_indices"]
        word_uncertainty = np.mean([token_entropies[i] for i in token_indices])
        step_word_uncertainties.append(word_uncertainty)
    
    step_words_selves = [word["word"] for word in step_words]
    return step_word_uncertainties, step_words_selves

def get_step_word_uncertainties_first_token(response, step_number, use_tail_bucket=False):
    step_start, step_end = get_step_start_end_indices(response, step_number)
    step_words = group_tokens_into_words_ignore_special(response.logprobs.tokens[step_start:step_end])
    token_entropies = get_step_token_entropies(response, step_number, use_tail_bucket)

    step_word_uncertainties = []
    for word in step_words:
        token_indices = word["token_indices"]
        word_uncertainty = token_entropies[token_indices[0]]
        step_word_uncertainties.append(word_uncertainty)
    
    step_words_selves = [word["word"] for word in step_words]
    return step_word_uncertainties, step_words_selves

def get_average_logprobs(choice, step_number):
    step_start, step_end = get_step_start_end_indices(choice, step_number)
    avg =0
    avg = sum(choice.logprobs.token_logprobs[step_start: step_end + 1])
    avg = avg / (step_end - step_start + 1) 

    return avg

def get_processed_data(data, dataset_name, use_tail_bucket=False):
    processed_data = []
    for resp in tqdm(data):
        acc = get_accuracy(resp, dataset_name)
        num_steps = len(get_reasoning_steps(resp.response))
        num_tokens = len(resp.response.logprobs.tokens)
        p = {
            'accuracy': acc,
            'num_steps': num_steps,
            'num_tokens': num_tokens,
            'step_confidences': [],
            'total_confidences': []
        }
        for step in range(1, num_steps + 1):
            step_token_entropies = get_step_token_entropies(resp.response, step, use_tail_bucket)
            step_word_uncertainties, _ = get_step_word_uncertainties(resp.response, step, use_tail_bucket)
            step_word_uncertainties_first, _ = get_step_word_uncertainties_first_token(resp.response, step, use_tail_bucket)
            step_confidences = {
                'avg_token_entropy': np.mean(step_token_entropies),
                'max_token_entropy': np.max(step_token_entropies),
                'std_token_entropy': np.std(step_token_entropies),
                # 'volatility_token_entropy': get_volatility(step_token_entropies)
            }
            if len(step_word_uncertainties) > 0:
                step_confidences.update({
                    'avg_word_entropy': np.mean(step_word_uncertainties),
                    'max_word_entropy': np.max(step_word_uncertainties),
                    'std_word_entropy': np.std(step_word_uncertainties),
                    # 'volatility_word_entropy': get_volatility(step_word_uncertainties)
                })
            else:
                step_confidences.update({
                    'avg_word_entropy': None,
                    'max_word_entropy': None,
                    'std_word_entropy': None,
                    # 'volatility_word_entropy': None
                })
            if len(step_word_uncertainties_first) > 0:
                step_confidences.update({
                    'avg_word_entropy_first': np.mean(step_word_uncertainties_first),
                    'max_word_entropy_first': np.max(step_word_uncertainties_first),
                    'std_word_entropy_first': np.std(step_word_uncertainties_first),
                    # 'volatility_word_entropy_first': get_volatility(step_word_uncertainties_first)
                })
            else:
                step_confidences.update({
                    'avg_word_entropy_first': None,
                    'max_word_entropy_first': None,
                    'std_word_entropy_first': None,
                    # 'volatility_word_entropy_first': None
                })
            p['step_confidences'].append(step_confidences)
        total_token_entropies = get_all_token_entropies(resp.response, use_tail_bucket)
        total_word_uncertainties, _ = get_word_uncertainties(resp.response, use_tail_bucket)
        total_word_uncertainties_first, _ = get_word_uncertainties_first_token(resp.response, use_tail_bucket)

        total_confidences = {
            'avg_token_entropy': np.mean(total_token_entropies),
            'max_token_entropy': np.max(total_token_entropies),
            'std_token_entropy': np.std(total_token_entropies),
            # 'volatility_token_entropy': get_volatility(total_token_entropies)
        }
        if len(total_word_uncertainties) > 0:
            total_confidences.update({
                'avg_word_entropy': np.mean(total_word_uncertainties),
                'max_word_entropy': np.max(total_word_uncertainties),
                'std_word_entropy': np.std(total_word_uncertainties)
            })
        else:
            total_confidences.update({
                'avg_word_entropy': None,
                'max_word_entropy': None,
                'std_word_entropy': None,
                # 'volatility_word_entropy': None
            })
        if len(total_word_uncertainties_first) > 0:
            total_confidences.update({
                'avg_word_entropy_first': np.mean(total_word_uncertainties_first),
                'max_word_entropy_first': np.max(total_word_uncertainties_first),
                'std_word_entropy_first': np.std(total_word_uncertainties_first),
                # 'volatility_word_entropy_first': get_volatility(total_word_uncertainties_first)
            })
        else:
            total_confidences.update({
                'avg_word_entropy_first': None,
                'max_word_entropy_first': None,
                'std_word_entropy_first': None,
                # 'volatility_word_entropy_first': None
            })
        p['total_confidences'] = total_confidences
        p['all_token_entropies'] = total_token_entropies
        processed_data.append(p)
    return processed_data

def main(args):
    dataset_name = args.dataset_name
    model_name = args.model_name
    use_tail_bucket = args.use_tail_bucket    
    results = []
    results_path = os.path.join(args.input_path, f"{dataset_name}_{model_name.replace('/', '_')}")
    if args.subdir:
        results_path = os.path.join(results_path, args.subdir)
    if not os.path.exists(results_path):
        raise ValueError(f"Results path does not exist: {results_path}")
    for file in os.listdir(results_path):
        if file.startswith('result_'):
            with open(os.path.join(results_path, file), 'r') as f:
                data = json.load(f)
            data = Box(data)
            results.append(data)
    results = get_processed_data(results, dataset_name, use_tail_bucket)
    output_path = args.output_path
    if args.subdir:
        output_path = os.path.join(output_path, args.subdir)
    os.makedirs(output_path, exist_ok=True)
    if use_tail_bucket:
        tb_flag = 'with_tail_bucket'
    else:
        tb_flag = 'without_tail_bucket'
    file_name = os.path.join(output_path, f'{dataset_name}_{model_name.replace("/", "_")}_{tb_flag}.json')
    with open(file_name, 'w') as f:
        json.dump(results, f, indent=4)


parser = argparse.ArgumentParser()
parser.add_argument('--dataset_name', type=str, required=True, help='Dataset name (e.g., gpqa, mmlupro, matteroffact)')
parser.add_argument('--model_name', type=str, required=True, help='Model name (e.g., Qwen/Qwen3-32B)')
parser.add_argument('--use_tail_bucket', action='store_true', help='Whether to use tail bucket in entropy calculation')
parser.add_argument('--output_path', type=str, required=True, help='Path to save processed data')
parser.add_argument('--input_path', type=str, required=True, help='Path to input results data')
parser.add_argument('--subdir', type=str, default='', help='optional subdirectory for input results data')
args = parser.parse_args()
if __name__ == "__main__":
    main(args)