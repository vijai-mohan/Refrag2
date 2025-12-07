# Evaluation Guide

This guide shows how to evaluate any Hugging Face decoder-only (causal LM) model on CPU using:
- lm-evaluation-harness (task-style benchmarks)
- RAGAS (RAG quality metrics on your own dataset)

The evaluation is Hydra-driven and integrates with the existing `run.py` driver.

## Prerequisites

- Python environment with the repository `requirements.txt` installed (it pins `datasets<3.0` to avoid breaking changes affecting some loaders in lm-eval). If you already have `datasets>=3.0`, reinstall with the pin.
- Optional GPU is supported, but this guide focuses on CPU-friendly settings.

## TL;DR

- lm-eval baseline on HellaSwag with any HF model id:

```cmd
python run.py task=eval model.name=meta-llama/Llama-3.2-1B eval.device=cpu eval.max_samples=100 eval.lm_eval.tasks=[hellaswag]
```

- RAGAS on your dataset (JSON/JSONL):

```cmd
python run.py task=eval \
  model.name=meta-llama/Llama-3.2-1B \
  eval.device=cpu \
  eval.ragas.enabled=true \
  eval.ragas.dataset_path=./path/to/ragas_dataset.jsonl \
  eval.ragas.limit=50 \
  eval.ragas.generate_responses=true
```

- Refrag paper baselines (accuracy + perplexity sweep using the built-in list from `conf/app/eval.yaml`):

```cmd
python run.py task=eval app=eval eval.baselines.enabled=true eval.baselines.accuracy_tasks=[hellaswag] eval.baselines.perplexity_tasks=[wikitext] eval.baselines.save_dir=./outputs/baselines
```

Outputs are written under `./outputs/eval/<run_name>/` by default.

## Configuration Overview

- Refrag paper baselines: edit `eval.baselines.models` in `conf/app/eval.yaml` to point `model_id` to your local checkpoints for each baseline listed in the paper (LLaMA-No-Context, LLaMA-Full-Context, CEPE, CEPED, LLaMA-32K, REPLUG, LLaMAK, LLaMA256, REFRAG variants). Set `eval.baselines.enabled=true` and the runner will iterate over the suite, saving summaries to `eval.baselines.save_dir/<suite_name>/summary.json`.

Edit `conf/config.yaml` or override via CLI flags.

Top-level:
- `task`: set to `eval` to run evaluation.
- `model.name`: any HF repo id for a decoder model (e.g., `meta-llama/Llama-3.2-1B`, `gpt2`).

`eval` section highlights:
- `save_dir`: root folder for results.
- `device`: `cpu` or `cuda`.
- `max_samples`: global cap used by helpers when supported.
- `trust_remote_code`: pass through to Hugging Face loaders and lm-eval (ignored by `datasets>=3.0`).

`eval.lm_eval`:
- `enabled`: turn lm-eval on/off.
- `tasks`: list of harness task names (e.g., `hellaswag`, `wikitext`, `piqa`, `arc_easy`, `openbookqa`).
- `batch_size`: small values recommended for CPU.
- `limit`: per-task sample limit when accepted by the harness (fallbacks handled internally).
- Results are sanitized to JSON and saved at `outputs/eval/<model>/lm_eval/lm_eval_results.json`.

`eval.ragas`:
- `enabled`: turn RAGAS on/off.
- `dataset_path`: JSON or JSONL path. Records should have:
  - `question` (string)
  - `contexts` (list[string] or string)
  - `ground_truth` (string or list[string])
  - `response` (optional; if missing, set `generate_responses=true` to have the model answer)
- A sample dataset is provided at `./data/ragas_samples.jsonl` and is wired into the default config.
- `limit`: cap number of rows (CPU-friendly).
- `generate_responses`: generate answers when `response` is missing.
- `fields`: rename if your keys differ.
- `generation`: decoding parameters used when generating responses.

Baseline comparison: `conf/app/eval.yaml` includes both `llama_full_context` and `llama_no_context` entries so you can directly compare metrics with and without context to gauge the value of context conditioning.

## Running lm-eval

Example: Evaluate `gpt2` on a couple of tasks with 50 samples each, CPU mode.

```cmd
python run.py task=eval model.name=gpt2 eval.device=cpu eval.lm_eval.tasks=[hellaswag,piqa] eval.lm_eval.limit=50
```

Results:
- Per-run JSON at `outputs/eval/<model>/lm_eval/lm_eval_results.json`
- Summary merged in `outputs/eval/<model>/summary.json`

## Running RAGAS

Prepare your dataset as JSONL with lines that include at least `question`, `contexts`, and `ground_truth`. If you don’t have model responses yet, set `generate_responses=true` and they’ll be generated with your chosen model.

Minimal JSONL example (3 lines shown):

```text
{"question":"What is the capital of France?","contexts":["Paris is the capital of France."],"ground_truth":"Paris"}
{"question":"Who wrote 1984?","contexts":["1984 is a novel by George Orwell."],"ground_truth":"George Orwell"}
{"question":"E=mc^2 relates mass and?","contexts":"Energy","ground_truth":"energy"}
```

Run RAGAS:

```cmd
python run.py task=eval model.name=gpt2 eval.ragas.enabled=true eval.ragas.dataset_path=./data/ragas_samples.jsonl eval.ragas.limit=20
```

Outputs:
- Per-row metrics at `outputs/eval/<model>/ragas/ragas_per_row.json`
- Summary metrics at `outputs/eval/<model>/ragas/ragas_results.json`
- Overall summary at `outputs/eval/<model>/summary.json`

## Tips for CPU

- Lower `eval.max_samples`, `eval.lm_eval.limit`, and `eval.ragas.limit` to speed up.
- Use small models first (e.g., `gpt2`, `ibm-granite/granite-4.0-350M`).
- Keep `eval.lm_eval.batch_size=1` on CPU.
- For RAGAS generation, reduce `generation.max_new_tokens`.

## Troubleshooting

- lm-eval import error: ensure `pip install -r requirements.txt`.
- Datasets trust_remote_code error:
  - Some versions of `datasets` (>=3.0) removed the `trust_remote_code` argument and can cause errors when lm-eval loads tasks. This repo pins `datasets<3.0`. If you still see errors in the logs, reinstall with the pin.
- Tokenizer without pad token: the evaluation runner sets pad to EOS/UNK to enable batching on CPU.
- RAGAS expects specific columns; adjust `eval.ragas.fields` if your keys differ.

## Where to look next

- `eval.py`: evaluation runner wrapping lm-eval and RAGAS.
- `conf/config.yaml`: central place to configure tasks and limits.
- `run.py`: Hydra driver; use `task=eval` to trigger this pipeline.
