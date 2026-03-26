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

### Step-entropy abstention experiment

Use [`scripts/abstain_step_entropy_experiment.py`](scripts/abstain_step_entropy_experiment.py) on a **flat** directory of `result_*.json` files (each must include `response.logprobs.tokens`, `response.logprobs.top_logprobs`, and `is_correct`).

**Thinking tokens:** Entropy is computed only on the chain-of-thought region. If `</think>` / `</think>` markers appear in token strings, the slice between them is used; if only a closing marker appears, tokens before it are used; if neither appears (e.g. plain instruct models), the full completion is used.

**Validation grid** (maximize F1 of abstention targeting incorrect answers):

| Hyperparameter | Search range |
|----------------|--------------|
| `chunk_size` | 50–600 step 50 |
| `delta` | 0.02–0.08 step 0.01 (min gap between incorrect/correct mean step entropy on val) |
| `noise` | 0.01–0.10 step 0.01 (added to midpoint threshold τⱼ) |
| `ground_threshold` | 1–10 (min number of “bad” active steps to abstain) |

**Additional rule (same as step −log p / cumulative entropy):** a step is only **active** if the validation set has at least **`min_support_per_class` correct and at least `min_support_per_class` incorrect** responses that reach that step (default **`--min_support_per_class 3`**).

Per-step τⱼ and **active** steps are fit on the validation split only (incorrect mean > correct mean, gap ≥ `delta`). **Test** metrics use the same τ / active from validation.

**Metrics:**

- **Baseline test accuracy:** fraction of test examples with `is_correct` (no abstention).
- **Test counts:** number of correct vs incorrect responses on the test split (printed).
- **Validation counts:** correct vs incorrect on the validation split (printed).
- **Test accuracy (with abstention):** `(# non-abstained ∧ is_correct) / N` — abstentions count as failures.
- **Abstention precision** = (abstained ∧ incorrect) / (abstained); **recall** = (abstained ∧ incorrect) / (incorrect).

**Plot:** After selecting the best hyperparameters, the script saves a figure of validation **mean chunk entropy vs step index** for correct vs incorrect examples, with **τ** and **τ + noise** at **active** steps (default path: `<results_dir>/abstain_val_step_entropy.png`). Use `--no_plot` to skip, or `--plot_path` to set the file path.

```bash
python scripts/abstain_step_entropy_experiment.py \
  --results_dir path/to/result_jsons \
  --val_size 60 \
  --seed 42 \
  --min_support_per_class 3 \
  --output_csv grid_results.csv
```

**Grid search visualization:** [`scripts/plot_step_entropy_hyperparam_grid.py`](scripts/plot_step_entropy_hyperparam_grid.py) reads a grid CSV and writes two PNGs per run: a **heatmap** of the best validation F1 over (noise, ground_threshold) for each (chunk_size, δ), and a **faceted scatter** (one row per δ) where every point is a full hyperparameter tuple (color = noise, marker size = ground_threshold). **Single-model CSV** (no `model_name` column): `--output_prefix` sets paths (`<prefix>_summary.png`, `<prefix>_detail.png`). **GPQA combined** [`abstaining_results/gpqa/grid.csv`](abstaining_results/gpqa/grid.csv) (has `model_name`): one pair of PNGs **per model** under `--output_dir` (default [`abstaining_plots/gpqa/`](abstaining_plots/gpqa/), files `<SafeModelName>_summary.png` / `_detail.png`). Examples: `python scripts/plot_step_entropy_hyperparam_grid.py --csv grid_results.csv --output_prefix runs/my_model_grid` and `python scripts/plot_step_entropy_hyperparam_grid.py --csv abstaining_results/gpqa/grid.csv`.

**GPQA multi-model batch (step entropy):** point `--gpqa_root` at a directory whose **subfolders** each contain one model’s `result_*.json` files. The script runs the grid + test **per subfolder**, saves validation plots into `--image_dir`, and writes results under **`--gpqa_results_dir`** (default [`abstaining_results/gpqa/`](abstaining_results/gpqa/)): **`avg_entropy.csv`** (per-model summary) and **`grid.csv`** (all models’ validation grid rows, with a `model_name` column). **Concrete commands** are in [Final GPQA commands](#final-gpqa-commands) below.

### Cumulative step-entropy abstention experiment

[`scripts/abstain_step_agg_entropy_experiment.py`](scripts/abstain_step_agg_entropy_experiment.py) matches the step-entropy abstention pipeline, but **per-step uncertainty** is the **cumulative sum** of per-chunk mean entropies from chunk 0 through the current step (same chunking as [`val_step_agg_entropy_plot.py`](scripts/val_step_agg_entropy_plot.py)). JSON requirements match step-entropy (`tokens`, `top_logprobs`, `is_correct`).

**Validation grid** (same objective: maximize abstention F1 on validation):

| Hyperparameter | Search range |
|----------------|--------------|
| `chunk_size` | 50–700 step 50 |
| `delta` | 0.1–10.0 step 0.3 (min gap between incorrect/correct **mean cumulative entropy** at a step on val) |
| `noise` | 1–30 step 0.5 (added to τⱼ when comparing cumulative entropy) |
| `ground_threshold` | 1–50 (min number of active steps with cumulative entropy > τⱼ + noise) |

**Additional rule (same as step −log p):** a step is only **active** if the validation set has at least **`min_support_per_class` correct and at least `min_support_per_class` incorrect** responses that reach that step (default **`--min_support_per_class 3`**).

The grid is large; use `--output_csv` in single-model mode only if you want the full search table. Default validation plot: `<results_dir>/abstain_val_step_agg_entropy.png`.

### Step −log p abstention experiment

[`scripts/abstain_step_neg_logprob_experiment.py`](scripts/abstain_step_neg_logprob_experiment.py) mirrors the entropy abstention pipeline but uses **mean −log p** per chunk (thinking tokens only) as step uncertainty. Per-token log probabilities must appear as **`response.logprobs.token_logprobs`** (older files) or **`response.logprobs.logprobs`** (newer format; same meaning).

**Same grid** as step-entropy (`chunk_size`, `delta`, `noise`, `ground_threshold`). **Additional rule:** a step is only eligible to be **active** (and get τ) if the validation set has **at least `min_support_per_class` correct and at least `min_support_per_class` incorrect** responses that reach that step (default **`--min_support_per_class 3`**, i.e. ≥ 3 each).

Default validation plot: `<results_dir>/abstain_val_step_neg_logprob.png`. GPQA batch saves `ModelName_val_step_neg_logprob.png` per model.

```bash
python scripts/abstain_step_neg_logprob_experiment.py \
  --results_dir path/to/result_jsons \
  --val_size 60 \
  --seed 42 \
  --min_support_per_class 3 \
  --output_csv grid_neg_logprob.csv
```

### Final GPQA commands

Use these from the repo root when GPQA model outputs live under [`outputs/gpqa/`](outputs/gpqa/) (one subfolder per model, each with `result_*.json`). You need **more than `val_size` examples per model** (e.g. 61+ files for `--val_size 60`).

**Step entropy** — writes [`abstaining_results/gpqa/avg_entropy.csv`](abstaining_results/gpqa/avg_entropy.csv) (summary) and [`abstaining_results/gpqa/grid.csv`](abstaining_results/gpqa/grid.csv) (combined validation grids for all models), and `*_val_step_entropy.png` under [`abstaining_plots/`](abstaining_plots/). Override the results directory with `--gpqa_results_dir` (default: `abstaining_results/gpqa`).

```bash
python scripts/abstain_step_entropy_experiment.py \
  --gpqa_root outputs/gpqa \
  --image_dir abstaining_plots \
  --gpqa_results_dir abstaining_results/gpqa \
  --val_size 60 \
  --seed 42 \
  --min_support_per_class 3
```

**Step −log p** — requires per-token log probs (`token_logprobs` or `logprobs`); writes [`abstaining_results/gpqa_neg_logprob.csv`](abstaining_results/gpqa_neg_logprob.csv) and `*_val_step_neg_logprob.png` under [`abstaining_plots/neg_logprob/`](abstaining_plots/neg_logprob/):

```bash
python scripts/abstain_step_neg_logprob_experiment.py \
  --gpqa_root outputs/gpqa \
  --image_dir abstaining_plots/neg_logprob \
  --summary_csv abstaining_results/gpqa_neg_logprob.csv \
  --val_size 60 \
  --seed 42 \
  --min_support_per_class 3
```

**Cumulative step entropy (aggregated uncertainty)** — same `top_logprobs` requirement as step entropy; writes [`abstaining_results/gpqa_agg_entropy.csv`](abstaining_results/gpqa_agg_entropy.csv) and `*_val_step_agg_entropy.png` under [`abstaining_plots/agg_entropy/`](abstaining_plots/agg_entropy/):

```bash
python scripts/abstain_step_agg_entropy_experiment.py \
  --gpqa_root outputs/gpqa \
  --image_dir abstaining_plots/agg_entropy \
  --summary_csv abstaining_results/gpqa_agg_entropy.csv \
  --val_size 60 \
  --seed 42 \
  --min_support_per_class 3
```

### Validation: mean negative log probability vs step

[`scripts/val_step_neg_logprob_plot.py`](scripts/val_step_neg_logprob_plot.py) plots **validation-only** curves of **mean −log p** per chunk (step) for correct vs incorrect responses, using `response.logprobs.token_logprobs` or `response.logprobs.logprobs` (natural log of the sampled token) on the **thinking** slice (same boundaries as abstention). Uses the same train/validation split idea as the abstention script (`--val_size`, `--seed`).

```bash
python scripts/val_step_neg_logprob_plot.py \
  --results_dir path/to/result_jsons \
  --val_size 60 \
  --chunk_size 50 \
  --seed 42 \
  --output_path neg_logprob_val.png \
  --title "MyModel"
```

Requires per-token log probs (`token_logprobs` or `logprobs` under `response.logprobs`) in each JSON. By default writes **one combined PNG** with three stacked panels (mean, variance, counts) and a **shared x-axis**. Use `--separate_plots` to also write standalone `*_mean_only.png`, `*_variance.png`, and `*_counts.png` beside the same stem.

### Validation: cumulative (aggregated) entropy vs step

[`scripts/val_step_agg_entropy_plot.py`](scripts/val_step_agg_entropy_plot.py) matches the layout above, but at each step the value is the **sum of per-chunk mean entropies** from chunk 0 through the current step (thinking-only token entropies, same `chunk_size` as abstention). Requires `top_logprobs` in `response.logprobs` like the entropy abstention pipeline.

```bash
python scripts/val_step_agg_entropy_plot.py \
  --results_dir /Users/bhushanshah/Documents/llm-uncertainty-project/outputs/gpqa/Qwen_Qwen3-32B \
  --val_size 60 \
  --chunk_size 50 \
  --seed 42 \
  --output_path abstaining_plots/val_agg_entropy_vs_step.png \
  --title "MyModel"
```

### Validation: cumulative positive jumps between chunk mean entropies vs step

[`scripts/val_step_pos_entropy_diff_plot.py`](scripts/val_step_pos_entropy_diff_plot.py) uses the same three-panel layout, but at step ``j`` the per-response value is the **cumulative sum of positive differences** between consecutive **chunk mean** entropies: ``sum_{i=1}^{j} max(0, m[i] - m[i-1])`` with ``m[0]`` contributing ``0`` (no prior chunk). This matches the positive-variation construction in [`test.ipynb`](test.ipynb) (Experiment 5), with chunking aligned to ``chunk_size`` and thinking-only entropies. See ``cumulative_positive_chunk_increments`` in [`abstain_step_entropy.py`](abstain_step_entropy.py).

```bash
python scripts/val_step_pos_entropy_diff_plot.py \
  --results_dir path/to/result_jsons \
  --val_size 60 \
  --chunk_size 50 \
  --seed 42 \
  --output_path abstaining_plots/val_pos_entropy_diff_vs_step.png \
  --title "MyModel"
```
