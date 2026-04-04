"""Minimal HF Router stream demo; set HF_TOKEN in .env."""

from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()
import os

client = OpenAI(
    base_url="https://router.huggingface.co/v1",
    api_key=os.getenv("HF_TOKEN"),
)

stream = client.chat.completions.create(
    model="Qwen/Qwen3-32B:groq",
    messages=[
        {
            "role": "user",
            "content": "What is the capital of India? Answer in one sentence only.",
        }
    ],
    max_tokens=10000,
    temperature=0,
    stream=True,
    logprobs=True,
    top_logprobs=5,
)

for chunk in stream:
    print(chunk)
