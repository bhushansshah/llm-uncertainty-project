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

### Abstention experiments (step entropy, −log p, step KL, cumulative entropy)

**Inputs:** JSON files named `result_<idx>.json` with `is_correct` and logprob fields as required by [`abstain_step_entropy.py`](abstain_step_entropy.py) (entropy methods need `response.logprobs` with `tokens` / `top_logprobs`; −log p needs `token_logprobs` or `logprobs`). Thinking-token boundaries follow the same rules as in that module.

**step_kl** ([`abstain_kl.py`](abstain_kl.py), [`scripts/abstain_step_kl_experiment.py`](scripts/abstain_step_kl_experiment.py)) uses the same `tokens` / `top_logprobs` structure as step entropy. It needs the **full vocabulary size V** per model: in batch mode pass `--vocab_map` JSON `{"<model_folder_name>": V, ...}` to the orchestrator or to the KL script; for a single `--results_dir`, pass `--vocab_size V`. Token-level **KL(U‖p)** is computed from the top‑k masses with the remainder spread **uniformly** over the other `V−k` types (documented in `abstain_kl.py`); it is exact KL for that completed distribution, not necessarily the true model KL if the tail differs.

**Rows with missing/invalid logprob structure** (e.g. `tokens` is `null`) are **dropped before** the train/validation split. For **step entropy**, **neg_logprob**, and **step_kl**, `--val_fraction` (e.g. `0.3` → 30% of the usable pool) sets the validation size; for **agg_entropy**, `--val_size` is still a fixed count on the usable pool. Grid search, τ, and active-step logic are unchanged; they apply only to examples that can be scored.

**Output layout** (batch mode — recommended):

- `abstaining_results/<dataset>/<method>/avg_entropy.csv` — one row per model  
- `abstaining_results/<dataset>/step_entropy/grid.csv`, `.../neg_logprob/grid.csv`, and `.../step_kl/grid.csv` — combined validation grids with a `model_name` column (**step_entropy**, **neg_logprob**, **step_kl** batch; agg_entropy batch does not write this file)  
- `abstaining_plots/<dataset>/<method>/<ModelSafeName>_val_step_<kind>.png`  
- `abstaining_threshold_marks/<dataset>/<method>/<ModelSafeName>_threshold_marks.png` — step entropy, −log p, and step_kl batch (optional `--abstaining_threshold_marks_dir`)

Where `<method>` is one of `step_entropy`, `neg_logprob`, `step_kl`, `agg_entropy`.

**Code layout:** [`abstain_experiment_common.py`](abstain_experiment_common.py) centralizes loading `result_*.json` directories, stratified validation/test splits (by fraction or fixed count), filename sanitization, and the **shared data-driven grid** used by step entropy, −log p, and step KL. Core abstention math (chunking, τ, active steps, F1) remains in [`abstain_step_entropy.py`](abstain_step_entropy.py); KL-to-uniform helpers in [`abstain_kl.py`](abstain_kl.py). Scripts under `scripts/` mostly compose these modules (agg_entropy keeps its own wider grid).

**Orchestrator** — runs every selected method for each dataset, optionally filtered to specific models:

```bash
python abstain_experiment.py \
  --outputs_dir outputs \
  --datasets gpqa \
  --models deepseek-ai_DeepSeek-V3.1 Qwen_Qwen3-32B openai_gpt-oss-120b Qwen_Qwen2.5-32B Qwen_Qwen3-235B-A22B-Thinking-2507 RedHatAI_DeepSeek-R1-Distill-Llama-70B-FP8-dynamic RedHatAI_Meta-Llama-3.1-70B-Instruct-FP8 \
  --methods step_entropy neg_logprob agg_entropy \
  --abstaining_results_dir abstaining_results \
  --abstaining_plots_dir abstaining_plots \
  --val_fraction 0.3 --val_size 60 --seed 42 --min_support_per_class 3
```

`--val_fraction` applies to `step_entropy`, `neg_logprob`, and `step_kl`; `--val_size` applies to `agg_entropy` only. Include **`--vocab_map vocab.json`** when `--methods` includes `step_kl` (JSON maps each model folder name to integer `V`). Omit `--models` to include every model folder under each dataset that contains `result_*.json`. You need **at least 2** usable examples per model for step entropy, neg_logprob, and step_kl, and **more than `--val_size`** for agg_entropy when using a fixed count.

**Single dataset + method** (same layout as the orchestrator):

```bash
python scripts/abstain_step_entropy_experiment.py \
  --outputs_dir outputs --dataset gpqa \
  --abstaining_results_dir abstaining_results --abstaining_plots_dir abstaining_plots \
  --val_fraction 0.3 --seed 42 --min_support_per_class 3
```

Swap the script for [`scripts/abstain_step_neg_logprob_experiment.py`](scripts/abstain_step_neg_logprob_experiment.py), [`scripts/abstain_step_kl_experiment.py`](scripts/abstain_step_kl_experiment.py) (requires `--vocab_map` in batch or `--vocab_size` for a single `--results_dir`), or [`scripts/abstain_step_agg_entropy_experiment.py`](scripts/abstain_step_agg_entropy_experiment.py). The cumulative-entropy script uses a larger validation grid (chunk 50–700, wider δ/noise ranges); it can be slow.

**Single model directory** (one flat folder of `result_*.json`):

```bash
python scripts/abstain_step_entropy_experiment.py \
  --results_dir outputs/gpqa/MyModel \
  --val_fraction 0.3 --seed 42 --min_support_per_class 3 \
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

### Validation: per-chunk mean KL(U‖p) (step KL) vs step

[`scripts/val_step_kl_plot.py`](scripts/val_step_kl_plot.py) matches the three-panel layout above: **mean chunk KL(U‖p)** (correct vs incorrect), variance of chunk means, and per-step counts on the validation split. Uses [`abstain_kl.py`](abstain_kl.py) (top‑k + uniform-tail completion) and the same thinking slice as step entropy. Requires **`--vocab_size`** (full vocabulary size `V`, same as [`scripts/abstain_step_kl_experiment.py`](scripts/abstain_step_kl_experiment.py)). Examples without aligned `tokens` / `top_logprobs` are dropped before the split.

```bash
python scripts/val_step_kl_plot.py \
  --dataset gpqa \
  --model Qwen_Qwen3-32B \
  --vocab_size 151936 \
  --val_size 60 \
  --chunk_size 50 \
  --seed 42
```

Output: `abstaining_validation_plot/<dataset>/<model>_val_step_kl.png`.

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
