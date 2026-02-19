# LLM Uncertainty

This repository contains resources and notes for the **LLM Uncertainty** project, focused on understanding, measuring, and analyzing uncertainty in Large Language Models (LLMs).

## Research Paper Reading List

The primary reading list for this project are maintained in the following Google Drive document:

**[LLM Uncertainty – Research Paper Reading List](https://docs.google.com/document/d/1aac8Eib-_C1iswuSVIThSGRYn4nHSL20nMcPFUq24pw/edit?usp=sharing)**

## Setup

Install dependencies:

```bash
uv pip install -r requirements.txt
```

## Data

The scripts expect an `outputs/` directory with the following structure:

```
outputs/
├── <dataset>/
│   ├── <model>/
│   │   ├── config.json
│   │   ├── result_0.json
│   │   ├── result_1.json
│   │   └── ...
│   └── <model>/
│       └── ...
└── <dataset>/
    └── ...
```

### Running Baselines

`computing_baselines.py` is a unified script that can run any combination of baselines in a single command. Available baselines: `neg_avg_logprobs`, `avg_token_entropy`, `trace_length`, `num_forking_tokens`, `answer_prob`.

```bash
python3 computing_baselines.py \
  --datasets gpqa mmlupro scifact_without_evidence scifact_with_evidence \
  --models openai_gpt-oss-120b Qwen_Qwen3-32B openai_gpt-oss-20b deepseek-ai_DeepSeek-R1-Distill-Llama-70B \
  --baselines avg_logprobs avg_token_entropy trace_length forking_tokens answer_prob \
  --results_dir results
```

Each baseline's results are saved as a separate CSV file in the results directory (e.g., `results/neg_avg_logprobs_baselines.csv`), with columns: `dataset`, `model`, `auroc`, `accuracy`.

