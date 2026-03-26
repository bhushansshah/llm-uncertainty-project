import os
import random
from datasets import load_dataset
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

PROMPT_TEMPLATE = (
    " Answer the question by choosing one of the provided options.\n"
    "Provide the final answer in the last line, prefixed with \"Answer:\". "
    "Do not answer with a full sentence. Just provide the letter of the correct choice, like: Answer: A\n\n"
    "Here is the question:\n{question}\n{choices}\n"
)

data = load_dataset("Idavidrein/gpqa", "gpqa_diamond", split="train")
row = data[random.randint(0, len(data) - 1)]

options = ["Correct Answer", "Incorrect Answer 1", "Incorrect Answer 2", "Incorrect Answer 3"]
random.shuffle(options)
symbols = ["A)", "B)", "C)", "D)"]
choices_str = "\n".join(symbols[i] + row[options[i]] for i in range(4))
gold_option = symbols[options.index("Correct Answer")].replace(")", "")

prompt = PROMPT_TEMPLATE.format(question=row["Question"], choices=choices_str)

print("=== Question ===")
print(prompt)
print(f"\n(Gold answer: {gold_option})\n")

client = OpenAI(
    base_url="https://router.huggingface.co/v1",
    api_key=os.environ["HF_TOKEN"],
)

stream = client.responses.create(
    model="openai/gpt-oss-120b",
    input=[{"role": "user", "content": prompt}],
    temperature=0.7,
    top_p=0.95,
    reasoning={"effort": "high", "summary": "auto"},
    stream=True,
)

in_reasoning = False
in_output = False
count = 0
thinking_text = ""
current_ind = -1
for event in stream:
    count += 1

    if event.type == "response.reasoning_text.delta":
        if not in_reasoning:
            print("=== Reasoning ===")
            in_reasoning = True
        thinking_text += event.delta
        if len(thinking_text) - 1 - current_ind > 200:
            print('--------------------------------')
            print(thinking_text[current_ind:])
            current_ind = len(thinking_text)
        
    elif event.type == "response.reasoning_text.done":
        if len(thinking_text) > current_ind:
            print("--------------------------------")
            print(thinking_text[current_ind:])
            print("Reasoning done")
            current_ind = len(thinking_text)
    elif event.type == "response.content_part.done":
        if not in_output:
            if in_reasoning:
                print("\n")
            print("=== Answer ===")
            in_output = True
        print(event.delta, end="", flush=True)
print()
print("Total chunks:", count)