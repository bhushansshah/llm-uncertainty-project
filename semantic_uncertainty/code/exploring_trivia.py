import argparse
import random
import textwrap

import datasets
from transformers import AutoTokenizer
import config


def _wrap(label, value, width=100):
    wrapped = textwrap.fill(str(value), width=width, subsequent_indent="  ")
    return f"{label}\n  {wrapped}"


def _format_value(value):
    try:
        import torch

        if isinstance(value, torch.Tensor):
            return value.tolist()
    except Exception:
        pass

    return value


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=10)
    parser.add_argument("--index", type=int, default=None)
    args = parser.parse_args()

    data_path = f"{config.data_dir}/trivia_qa"
    dataset = datasets.load_from_disk(data_path)

    if len(dataset) == 0:
        raise ValueError(f"No examples found at {data_path}")

    if args.index is None:
        rng = random.Random(args.seed)
        idx = rng.randrange(len(dataset))
    else:
        if args.index < 0 or args.index >= len(dataset):
            raise IndexError(f"Index {args.index} out of range (0..{len(dataset) - 1})")
        idx = args.index

    sample = dataset[idx]
    tokenizer = AutoTokenizer.from_pretrained("facebook/opt-350m", use_fast=False)

    input_ids = sample.get("input_ids")
    decoder_input_ids = sample.get("decoder_input_ids")

    decoded_prompt = None
    if input_ids is not None:
        decoded_prompt = tokenizer.decode(input_ids.tolist(), skip_special_tokens=True)

    decoded_answer = None
    if decoder_input_ids is not None:
        decoded_answer = tokenizer.decode(decoder_input_ids.tolist(), skip_special_tokens=True)

    print(f"Example index: {idx} / {len(dataset) - 1}")
    print("Fields and values:")
    for key in sorted(sample.keys()):
        print(_wrap(f"- {key}:", _format_value(sample.get(key))))
    print(_wrap("Question:", sample.get("question", "N/A")))
    print(_wrap("Answer:", sample.get("answer", "N/A")))
    if decoded_prompt is not None:
        print(_wrap("Prompt (decoded input_ids):", decoded_prompt))
    if decoded_answer is not None:
        print(_wrap("Decoded decoder_input_ids:", decoded_answer))


if __name__ == "__main__":
    main()

