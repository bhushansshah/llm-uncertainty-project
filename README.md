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
  --baselines avg_logprobs avg_token_entropy trace_length forking_tokens normalized_forking_tokens answer_prob \
  --results_dir results
```

Each baseline's results are saved as a separate CSV file in the results directory (e.g., `results/neg_avg_logprobs_baselines.csv`), with columns: `dataset`, `model`, `auroc`, `accuracy`.

### Getting Freeform Baselines

Run `computing_baselines.py` with the is a unified script that can run any combination of baselines in a single command. Available baselines: `neg_avg_logprobs`, `avg_token_entropy`, `trace_length`, `num_forking_tokens`.

```bash
python3 computing_baselines.py \
  --datasets gpqa_free_answer mmlupro_free_answer \
  --models openai_gpt-oss-120b Qwen_Qwen3-32B \
  --baselines avg_logprobs avg_token_entropy trace_length forking_tokens normalized_forking_tokens \
  --results_dir results \
  --results_file free_answer_baselines.csv
```

### Running Verbalized Baselines

`verbalized_comparison.py` is the corresponding script for **verbalized** outputs, where result files store token-level logprobs in `response.logprobs.content` (a different JSON shape than `computing_baselines.py` expects).

By default, this script reads from `outputs_feb21/verbalized/`.

```bash
python3 verbalized_comparison.py \
  --datasets gpqa mmlupro scifact_without_evidence scifact_with_evidence \
  --models openai_gpt-oss-120b Qwen_Qwen3-32B openai_gpt-oss-20b deepseek-ai_DeepSeek-R1-Distill-Llama-70B \
  --baselines avg_logprobs avg_token_entropy trace_length forking_tokens normalized_forking_tokens answer_prob verbalized\
  --results_dir results
```

Arguments are the same as `computing_baselines.py`, with one key default:

- `--outputs_dir` defaults to `outputs_feb21/verbalized` (override if needed).

Each baseline is saved as a separate CSV in `--results_dir` (e.g., `results/verbalized_baselines/avg_logprobs_baselines.csv`), with columns including `dataset`, `model`, `auroc`, and `accuracy` (and `num_selected_examples` for `answer_prob`).

### Abstention experiments (step entropy, −log p, cumulative entropy)

**Inputs:** JSON files named `result_<idx>.json` with `is_correct` and logprob fields as required by [`abstain_step_entropy.py`](abstain_step_entropy.py) (entropy methods need `response.logprobs` with `tokens` / `top_logprobs`; −log p needs `token_logprobs` or `logprobs`). Thinking-token boundaries follow the same rules as in that module.

**Output layout** (batch mode — recommended):

- `abstaining_results/<dataset>/<method>/avg_entropy.csv` — one row per model (all three methods)  
- `abstaining_results/<dataset>/step_entropy/grid.csv` — combined validation grid with a `model_name` column (**step-entropy batch only**, unchanged from before; neg_logprob and agg_entropy batch runs never wrote this file)  
- `abstaining_plots/<dataset>/<method>/<ModelSafeName>_val_step_<kind>.png`

Where `<method>` is one of `step_entropy`, `neg_logprob`, `agg_entropy`.

**Orchestrator** — runs every selected method for each dataset, optionally filtered to specific models:

```bash
python abstain_experiment.py \
  --outputs_dir outputs \
  --datasets gpqa \
  --models deepseek-ai_DeepSeek-V3.1 Qwen_Qwen3-32B openai_gpt-oss-120b Qwen_Qwen2.5-32B Qwen_Qwen3-235B-A22B-Thinking-2507 RedHatAI_DeepSeek-R1-Distill-Llama-70B-FP8-dynamic RedHatAI_Meta-Llama-3.1-70B-Instruct-FP8 \
  --methods step_entropy neg_logprob agg_entropy \
  --abstaining_results_dir abstaining_results \
  --abstaining_plots_dir abstaining_plots \
  --val_size 60 --seed 42 --min_support_per_class 3
```

Omit `--models` to include every model folder under each dataset that contains `result_*.json`. You need **more than `--val_size`** examples per model (e.g. 61+ files for `--val_size 60`).

**Single dataset + method** (same layout as the orchestrator):

```bash
python scripts/abstain_step_entropy_experiment.py \
  --outputs_dir outputs --dataset gpqa \
  --abstaining_results_dir abstaining_results --abstaining_plots_dir abstaining_plots \
  --val_size 60 --seed 42 --min_support_per_class 3
```

Swap the script for [`scripts/abstain_step_neg_logprob_experiment.py`](scripts/abstain_step_neg_logprob_experiment.py) or [`scripts/abstain_step_agg_entropy_experiment.py`](scripts/abstain_step_agg_entropy_experiment.py). The cumulative-entropy script uses a larger validation grid (chunk 50–700, wider δ/noise ranges); it can be slow.

**Single model directory** (one flat folder of `result_*.json`):

```bash
python scripts/abstain_step_entropy_experiment.py \
  --results_dir outputs/gpqa/MyModel \
  --val_size 60 --seed 42 --min_support_per_class 3 \
  --output_csv grid_results.csv
```

Use `--plot_path` / `--no_plot` as needed. Same pattern for the other two scripts (`--results_dir` + optional `--output_csv`).

**Grid visualization:** [`scripts/plot_step_entropy_hyperparam_grid.py`](scripts/plot_step_entropy_hyperparam_grid.py) reads a grid CSV (`--csv`) and writes heatmaps/scatters; for multi-model CSVs with `model_name`, point `--output_dir` at a folder and pass e.g. `abstaining_results/gpqa/step_entropy/grid.csv`.

### Validation: mean negative log probability vs step

[`scripts/val_step_neg_logprob_plot.py`](scripts/val_step_neg_logprob_plot.py) plots **validation-only** curves of **mean −log p** per chunk (step) for correct vs incorrect responses, using `response.logprobs.token_logprobs` or `response.logprobs.logprobs` (natural log of the sampled token) on the **thinking** slice (same boundaries as abstention). Uses the same train/validation split idea as the abstention script (`--val_size`, `--seed`).

Reads `outputs/<dataset>/<model>/result_*.json` (override root with `--outputs_dir`). Writes one combined PNG to `abstaining_validation_plot/<dataset>/<model>_val_step_neg_logprob.png` (model name sanitized for the filename).

```bash
python scripts/val_step_neg_logprob_plot.py \
  --dataset gpqa \
  --model deepseek-ai_DeepSeek-V3.1 \
  --val_size 60 \
  --chunk_size 50 \
  --seed 42
```

Requires per-token log probs (`token_logprobs` or `logprobs` under `response.logprobs`) in each JSON. Optional `--title` overrides the default figure title line (default is the model folder name).

### Validation: per-chunk mean entropy (step entropy) vs step

[`scripts/val_step_entropy_plot.py`](scripts/val_step_entropy_plot.py) uses the same CLI and output layout as the plots above. At each step it plots the **mean token entropy within that chunk only** (`chunk_step_means` in [`abstain_step_entropy.py`](abstain_step_entropy.py)), matching the step-entropy abstention experiment — **not** cumulative (see the next section for cumulative / aggregated entropy).

Requires `response.logprobs` with `tokens` and `top_logprobs`. Output: `abstaining_validation_plot/<dataset>/<model>_val_step_entropy.png`.

```bash
python scripts/val_step_entropy_plot.py \
  --dataset gpqa \
  --model Qwen_Qwen3-32B \
  --val_size 60 \
  --chunk_size 50 \
  --seed 42
```

### Validation: cumulative (aggregated) entropy vs step

[`scripts/val_step_agg_entropy_plot.py`](scripts/val_step_agg_entropy_plot.py) matches the layout above, but at each step the value is the **sum of per-chunk mean entropies** from chunk 0 through the current step (thinking-only token entropies, same `chunk_size` as abstention). Requires `top_logprobs` in `response.logprobs` like the entropy abstention pipeline.

```bash
python scripts/val_step_agg_entropy_plot.py \
  --dataset gpqa \
  --model Qwen_Qwen3-32B \
  --val_size 60 \
  --chunk_size 50 \
  --seed 42
```

Output: `abstaining_validation_plot/gpqa/Qwen_Qwen3-32B_val_step_agg_entropy.png` (path reflects `--dataset` and sanitized `--model`).

### Validation: cumulative positive jumps between chunk mean entropies vs step

[`scripts/val_step_pos_entropy_diff_plot.py`](scripts/val_step_pos_entropy_diff_plot.py) uses the same three-panel layout, but at step ``j`` the per-response value is the **cumulative sum of positive differences** between consecutive **chunk mean** entropies: ``sum_{i=1}^{j} max(0, m[i] - m[i-1])`` with ``m[0]`` contributing ``0`` (no prior chunk). This matches the positive-variation construction in [`test.ipynb`](test.ipynb) (Experiment 5), with chunking aligned to ``chunk_size`` and thinking-only entropies. See ``cumulative_positive_chunk_increments`` in [`abstain_step_entropy.py`](abstain_step_entropy.py).

```bash
python scripts/val_step_pos_entropy_diff_plot.py \
  --dataset gpqa \
  --model Qwen_Qwen3-32B \
  --val_size 60 \
  --chunk_size 50 \
  --seed 42
```

Output: `abstaining_validation_plot/gpqa/<model>_val_step_pos_entropy_diff.png`.
