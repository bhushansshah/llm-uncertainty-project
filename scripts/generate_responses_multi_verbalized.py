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
from dotenv import load_dotenv

load_dotenv()

# Prompt must end with <think> so Qwen3 and other thinking models produce reasoning before Answer:
PROMPT_TEMPLATE = """You will receive a medical case file, including Case Information,
Physical Examination, and Diagnostic Tests.

Your task is to reason through the case in a natural, human-like way, as a clinician thinking aloud.

IMPORTANT INSTRUCTIONS ABOUT CONFIDENCE REPORTING:

• As you reason, you MUST periodically pause to assess how confident you are so far.
• A pause should occur naturally after analyzing a meaningful portion of the case 
  (for example: after reviewing case information, after physical exam interpretation,
  after diagnostic tests, or after narrowing the differential).
• You MUST include AT LEAST THREE confidence pauses during your reasoning.
• Each pause MUST appear on a new line in EXACTLY this format:

Confidence: <CLASS>

• After providing the confidence, you MUST continue your reasoning by SEPERATING it by DOUBLE NEWLINES.
• NOTE that the confidence should be solely based on the reasoning and should accurately reflect your confidence in the reasoning so far.
• If you find out that your confidence is not high maybe you need to go back and review your reasoning and find out what you missed.
• NOTE that if you think perfectly fine to have low confidence in your reasoning and you should be UNBIASED while reporting it.
• After completing your full reasoning, provide the final answer in EXACTLY this format at the end:

Answer: <LETTER>
Confidence: <CLASS>

Where <CLASS> is one of the following (write ONLY the name):

- Almost no chance
- Highly unlikely
- Chances are slight
- Unlikely
- Less than even
- Better than even
- Likely
- Very good chance
- Highly likely
- Almost certain

Rules:
- Do NOT include probability numbers.
- Do NOT modify the class names.
- Do NOT skip confidence reporting.
- Confidence lines must appear naturally embedded in your reasoning.
- Each confidence statement should reflect your confidence at that specific point in reasoning — NOT final confidence unless it is the end.

Here is the medical case file:

Case Information:
%s

Physical Examination:
%s

Diagnostic Tests:
%s

Here are the four options:
%s
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
    # find the last occurrence of "Confidence: "
    confidence_id = text.lower().rfind("confidence: ")
    if confidence_id == -1:
        return ""
    # take all the text after "Confidence: " and before the next newline and return the confidence class
    confidence_text = text[confidence_id + len("confidence: "):].strip()
    confidence_class = confidence_text.split("\n")[0].strip()
    return confidence_class

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

    confidence_texts = [
        text[i + len(needle):].split("\n")[0].strip()
        for i in confidence_ids
    ]
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
        "=prompt": PROMPT_TEMPLATE,
        "sampling_params": sampling_params,
    }
    with open(os.path.join(out_dir, "config.json"), "w") as f:
        json.dump(config, f, indent=4)

    base_url = f"http://localhost:{args.port}/v1"
    openrouter_url = f"https://openrouter.ai/api/v1"
    openrouter_api_key = os.getenv("OPENROUTER_API_KEY")
    client = OpenAI(api_key=openrouter_api_key, base_url=openrouter_url)

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
        user_prompt = PROMPT_TEMPLATE.format(case_info=case_info, physical=physical, diagnostic=diagnostic, options=options_str)
        messages = [
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
        print(resp_dict.get("message").get("reasoning"))
        extracted_step_confidences = extract_step_confidences(resp_dict.get("message").get("reasoning"))
        extracted_confidence_class = get_confidence_class(extracted_confidence)
        if extracted_confidence_class is None:
            print(f"Warning: Confidence class not found for {extracted_confidence}")
            continue
        correct = is_correct_answer(extracted_answer, right_option, gold_answer)

        res = {
            "question": user_prompt,
            "gold answer": gold_answer,
            "gold option": right_option,
            "response": cleaned_resp_dict,
            "extracted_answer": extracted_answer,
            "extracted_confidence": extracted_confidence_class,
            "extracted_step_confidences": extracted_step_confidences,
            "is_correct": correct,
        }

        result_path = os.path.join(out_dir, f"result_{idx}.json")
        with open(result_path, "w") as f:
            json.dump(res, f, indent=4)

        print(f"Wrote {result_path}")

    print(f"Done. Output directory: {out_dir}")


if __name__ == "__main__":
    main()
