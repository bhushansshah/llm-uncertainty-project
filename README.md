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

Each `result_*.json` file contains one data example with the model's response and log-probabilities.

## Reproducing Baseline Metrics

### Average Log-Probability Baseline

Computes the negative average log-probability per question as an uncertainty score and reports AUROC.

```bash
python3 avg_logprobs_baselines.py \
  --datasets gpqa mmlupro scifact_without_evidence scifact_with_evidence\
  --models openai_gpt-oss-120b Qwen_Qwen3-32B \
  --results_filepath results/avg_logprobs_baselines.csv
```

Results are saved to the specified CSV path with columns: `dataset`, `model`, `auroc`.

### Average Token Entropy Baseline

Computes the average token entropy per question using top-k log-probabilities as an uncertainty score and reports AUROC.

```bash
python3 avg_token_entropy_baselines.py \
  --datasets gpqa mmlupro scifact_without_evidence scifact_with_evidence\
  --models openai_gpt-oss-120b Qwen_Qwen3-32B \
  --results_filepath results/avg_token_entropy_baselines.csv
```

Results are saved to the specified CSV path with columns: `dataset`, `model`, `auroc`.

### Trace Length Baseline

Uses the length of the reasoning trace (number of tokens) as an uncertainty score and reports AUROC.

```bash
python3 trace_length_baselines.py \
  --datasets gpqa mmlupro scifact_without_evidence scifact_with_evidence\
  --models openai_gpt-oss-120b Qwen_Qwen3-32B \
  --results_filepath results/trace_length_baselines.csv
```

Results are saved to the specified CSV path with columns: `dataset`, `model`, `auroc`.

### Normalized Trace Length Baseline

Uses the z-score normalized trace length (number of tokens, standardized by mean and std across the dataset/model combination) as an uncertainty score and reports AUROC.

```bash
python3 normalized_trace_length_baselines.py \
  --datasets gpqa mmlupro scifact_without_evidence scifact_with_evidence\
  --models openai_gpt-oss-120b Qwen_Qwen3-32B \
  --results_filepath results/normalized_trace_length_baselines.csv
```

Results are saved to the specified CSV path with columns: `dataset`, `model`, `auroc`.

