#!/usr/bin/env python3
"""
Generate model responses for the DiagnosisArena dataset via a vLLM server (OpenAI-compatible API).
Writes config.json and result_<idx>.json under outputs/DiagnosisArena/<model>/.
"""

import argparse
import json
import os

from datasets import load_dataset
from openai import OpenAI

# Prompt must end with <think> so Qwen3 and other thinking models produce reasoning before Answer:
SYSTEM_PROMPT = """
You will receive a medical case file, including Case Information,
Physical Examination and Diagnostic Tests. First, reason through the medical case file step by step to arrive at an answer and select the most appropriate diagnosis from the following four options.
Then, assess your confidence in that answer by evaluating your thinking process.
Finally, classify your confidence into one of the following classes based on how likely your answer is to be
correct:

- "Almost no chance" (0.0-0.1)
- "Highly unlikely" (0.1-0.2)
- "Chances are slight" (0.2-0.3)
- "Unlikely" (0.3-0.4)
- "Less than even" (0.4-0.5)
- "Better than even" (0.5-0.6)
- "Likely" (0.6-0.7)
- "Very good chance" (0.7-0.8)
- "Highly likely" (0.8-0.9)
- "Almost certain" (0.9-1.0)

Each category reflects the probability that your answer is correct.
At the very end of your output, format your answer and confidence as
Answer: $ANSWER
Confidence: $CLASS
where ANSWER is the choice of the correct diagnosis (Eg. A) <diagnosis>) and CLASS is one of the names (only the names without the probability ranges) of the classes above."""

USER_PROMPT_TEMPLATE = """Here is the medical case file:
Case Information:
{case_info}
Physical Examination:
{physical}
Diagnostic Tests:
{diagnostic}
Here are the four options:
{options}

<think>
"""


def format_options(options_dict):
    """Format Options dict as 'A: ...\\nB: ...\\nC: ...\\nD: ...'."""
    return "\n".join(f"{k}) {options_dict[k]}" for k in sorted(options_dict.keys()))


def extract_answer(text):
    """Extract answer text after 'Answer:' (and after </think> if present). Matches process.py logic."""
    if "</think>" in text:
        text = text.split("</think>")[-1]
    answer_id = text.lower().find("answer: ")
    if answer_id == -1:
        return ""
    # Slice to end of string, then strip (e.g. "Answer: A" -> "A")
    answer_text = text[answer_id + len("answer: "):].strip()
    return answer_text.split("\n")[0].strip()


def extract_confidence(text):
    """Extract confidence text after 'Confidence:' (and after </think> if present) and return the confidence class."""
    if "</think>" in text:
        text = text.split("</think>")[-1]
    confidence_id = text.lower().find("confidence: ")
    if confidence_id == -1:
        return ""
    # take all the text after "Confidence: " and before the next newline and return the confidence class
    confidence_text = text[confidence_id + len("confidence: "):].strip()
    confidence_class = confidence_text.split("\n")[0].strip()
    return confidence_class

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

def is_correct_answer(answer, gold_option, gold_answer):
    """Check if extracted answer matches gold option or gold answer text. Matches process.py logic."""
    gd = gold_option.lower()
    ans = answer.lower().strip()[0]
    if ans == gd:
        return True
    if "*" in ans:
        cleaned_ans = ans.replace("*", "").strip().lower()
        if cleaned_ans == gd or cleaned_ans == gold_answer.lower().strip():
            return True
        if ":" in cleaned_ans:
            cleaned_ans = cleaned_ans.split(":")[-1].strip()
            if cleaned_ans == gd or cleaned_ans == gold_answer.lower().strip():
                return True
    return False


def main():
    parser = argparse.ArgumentParser(description="Generate DiagnosisArena responses via vLLM (OpenAI API).")
    parser.add_argument("--model", type=str, required=True, help="Model name (e.g. Qwen/Qwen3-32B).")
    parser.add_argument("--output_dir", type=str, default="outputs", help="Top-level output directory.")
    parser.add_argument("--dataset_name", type=str, default="DiagnosisArena", help="Output subfolder name for this dataset.")
    parser.add_argument("--port", type=int, default=8000, help="vLLM server port.")
    parser.add_argument("--num_samples", type=int, default=-1, help="Max number of test samples (-1 = all).")
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top_p", type=float, default=0.95)
    parser.add_argument("--top_k", type=int, default=20)
    parser.add_argument("--no_do_sample", action="store_true", help="Disable sampling (default: do_sample=True).")
    args = parser.parse_args()
    args.do_sample = not args.no_do_sample

    model_slug = args.model.replace("/", "_")
    out_dir = os.path.join(args.output_dir, args.dataset_name, model_slug)
    os.makedirs(out_dir, exist_ok=True)

    sampling_params = {
        "temperature": args.temperature,
        "top_p": args.top_p,
        "top_k": args.top_k,
        "do_sample": args.do_sample,
    }
    config = {
        "arguments": vars(args),
        "system_prompt": SYSTEM_PROMPT,
        "user_prompt_template": USER_PROMPT_TEMPLATE,
        "sampling_params": sampling_params,
    }
    with open(os.path.join(out_dir, "config.json"), "w") as f:
        json.dump(config, f, indent=4)

    base_url = f"http://localhost:{args.port}/v1"
    client = OpenAI(api_key="EMPTY", base_url=base_url)

    ds = load_dataset("shzyk/DiagnosisArena", split="test")
    n_logprobs = 10
    max_tokens = 8192

    for idx in range(len(ds)):
        if args.num_samples >= 0 and idx >= args.num_samples:
            break
        row = ds[idx]
        case_info = row["Case Information"]
        physical = row["Physical Examination"]
        diagnostic = row["Diagnostic Tests"]
        options = row["Options"]
        right_option = row["Right Option"]

        options_str = format_options(options)
        user_prompt = USER_PROMPT_TEMPLATE.format(case_info=case_info, physical=physical, diagnostic=diagnostic, options=options_str)
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ]

        gold_answer = options.get(right_option, row.get("Final Diagnosis", ""))

        extra_body = {
            "top_k": args.top_k,
            "do_sample": args.do_sample,
            "reasoning": {"enabled": True},
            "top_logprobs": n_logprobs,
        }

        response = client.chat.completions.create(
            model=args.model,
            messages=messages,
            temperature=args.temperature,
            top_p=args.top_p,
            extra_body=extra_body,
            logprobs=True,
            n=1,
            max_tokens=max_tokens,
        )

        resp_choice = response.choices[0]
        resp_dict = resp_choice.model_dump()
        if resp_dict.get("logprobs"):
            resp_dict["logprobs"].pop("text_offset", None)
        cleaned_resp_dict = {
            "finish_reason": resp_dict.get("finish_reason"),
            "logprobs": resp_dict.get("logprobs"),
            "message": resp_dict.get("message"),
        }
        extracted_answer = extract_answer(resp_dict.get("message").get("content"))
        extracted_confidence = extract_confidence(resp_dict.get("message").get("content"))
        extracted_confidence_class = get_confidence_class(extracted_confidence)
        if extracted_confidence_class is None:
            print(f"Warning: Confidence class not found for {extracted_confidence}")
            continue
        correct = is_correct_answer(extracted_answer, right_option, gold_answer)

        res = {
            "question": messages[0]["content"] + messages[1]["content"],
            "gold answer": gold_answer,
            "gold option": right_option,
            "response": cleaned_resp_dict,
            "extracted_answer": extracted_answer,
            "extracted_confidence": extracted_confidence_class,
            "is_correct": correct,
        }

        result_path = os.path.join(out_dir, f"result_{idx}.json")
        with open(result_path, "w") as f:
            json.dump(res, f, indent=4)

        print(f"Wrote {result_path}")

    print(f"Done. Output directory: {out_dir}")


if __name__ == "__main__":
    main()
