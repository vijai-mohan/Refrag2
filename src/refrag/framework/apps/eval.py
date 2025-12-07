import importlib
import inspect
import json
import logging
import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

# Ensure Hugging Face never prompts for remote code confirmation.
os.environ.setdefault("HF_ALLOW_CODE", "1")
os.environ.setdefault("TRANSFORMERS_NO_ADVISORY_WARNINGS", "1")

from tqdm.auto import tqdm

from omegaconf import DictConfig, OmegaConf

log = logging.getLogger(__name__)


def _optional_import(module_name: str) -> Optional[Any]:

    """Return a module if available without raising on missing modules."""


    spec = importlib.util.find_spec(module_name)
    return importlib.import_module(module_name) if spec else None


def _sanitize_for_json(obj: Any) -> Any:
    """Convert common python/numpy/torch objects into JSON-friendly primitives."""
    np = _optional_import("numpy")
    torch = _optional_import("torch")

    if obj is None or isinstance(obj, (bool, int, float, str)):
        return obj

    if np is not None:
        if isinstance(obj, np.generic):
            return obj.item()
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, np.dtype):
            return str(obj)

    if torch is not None:
        if isinstance(obj, torch.Tensor):
            return obj.detach().cpu().tolist()
        if isinstance(obj, torch.dtype):
            return str(obj)
        torch_device_cls = getattr(torch, "device", None)
        if torch_device_cls is not None and isinstance(obj, torch_device_cls):
            return str(obj)

    if isinstance(obj, dict):
        return {str(_sanitize_for_json(k)): _sanitize_for_json(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_sanitize_for_json(x) for x in obj]
    if isinstance(obj, set):
        return [_sanitize_for_json(x) for x in sorted(list(obj), key=lambda x: str(x))]

    return str(obj)


def _collect_accuracy_and_perplexity(lm_eval_summary: Dict[str, Any]) -> Dict[str, Any]:
    """Extract accuracy/perplexity metrics from an lm-eval summary payload."""
    import numbers

    tasks = lm_eval_summary.get("tasks", {})
    accuracy: Dict[str, float] = {}
    perplexity: Dict[str, float] = {}
    for task, metrics in tasks.items():
        for metric_name, value in metrics.items():
            if not isinstance(value, numbers.Number):
                continue
            lower_name = metric_name.lower()
            key = f"{task}.{metric_name}"
            if "acc" in lower_name:
                accuracy[key] = float(value)
            if "ppl" in lower_name or "perplexity" in lower_name:
                perplexity[key] = float(value)

    metrics: Dict[str, Any] = {}
    if accuracy:
        metrics["accuracy"] = accuracy
        metrics["accuracy_avg"] = sum(accuracy.values()) / len(accuracy)
    if perplexity:
        metrics["perplexity"] = perplexity
        metrics["perplexity_avg"] = sum(perplexity.values()) / len(perplexity)
    return metrics


def _load_hf_causal_model(
    model_id: str,
    device: str = "cpu",
    trust_remote_code: bool = True,
) -> Tuple[Any, Any]:
    """Load a Hugging Face decoder-only (causal LM) model and tokenizer."""
    from transformers import AutoModelForCausalLM, AutoTokenizer
    import torch

    log.info("Loading HF causal model %s on %s", model_id, device)
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=trust_remote_code)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token if tokenizer.eos_token else tokenizer.unk_token
        tokenizer.pad_token_id = tokenizer.convert_tokens_to_ids(tokenizer.pad_token)

    model = AutoModelForCausalLM.from_pretrained(model_id, trust_remote_code=trust_remote_code)
    torch_device = torch.device("cuda" if (device.startswith("cuda") and torch.cuda.is_available()) else "cpu")
    model.to(torch_device)
    model.eval()
    return tokenizer, model


def _generate_responses(
    prompts: List[str],
    tokenizer: Any,
    model: Any,
    device: str = "cpu",
    max_new_tokens: int = 256,
    temperature: float = 0.0,
    top_p: float = 1.0,
    do_sample: bool = False,
) -> List[str]:
    """Generate responses for prompts with simple micro-batching to stay CPU-friendly."""
    import torch

    torch_device = torch.device("cuda" if (device.startswith("cuda") and torch.cuda.is_available()) else "cpu")
    outputs: List[str] = []
    batch_size = 1 if len(prompts) <= 1 else 4
    for i in range(0, len(prompts), batch_size):
        batch = prompts[i : i + batch_size]
        inputs = tokenizer(batch, padding=True, truncation=True, return_tensors="pt").to(torch_device)
        with torch.no_grad():
            gen_ids = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=do_sample,
                temperature=temperature,
                top_p=top_p,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
        outputs.extend(tokenizer.batch_decode(gen_ids, skip_special_tokens=True))
    return outputs


def _build_generation_prompt(question: str, contexts: List[str], disable_context: bool) -> str:
    """Create a prompt for response generation, optionally ignoring context."""
    if disable_context:
        return f"Summarize the question concisely and provide an answer.\n\nQuestion: {question}\nAnswer:"
    if contexts:
        context_block = "\n\n".join(contexts)
        return (
            "Use the following context to answer the question.\n\n"
            + context_block
            + "\n\nQuestion: "
            + question
            + "\nAnswer:"
        )
    return f"Question: {question}\nAnswer:"


def _is_no_context_run(run_name: str) -> bool:
    """Determine whether this run should avoid feeding context to the model."""
    return run_name.lower() == "llama_no_context"


def _run_lm_eval(
    cfg: DictConfig,
    save_dir: str,
    model_id: str,
    run_name: Optional[str] = None,
    tasks_override: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """Run lm-evaluation-harness using the HF driver and persist results."""
    from lm_eval import evaluator
    import datasets as _datasets
    from packaging import version as _version
    import numbers

    dataset_version = getattr(_datasets, "__version__", "0")
    if _version.parse(dataset_version) >= _version.parse("3.0.0"):
        log.warning(
            "datasets>=3.0 detected. If lm-eval logs trust_remote_code errors during dataset loading, "
            "pin datasets<3.0 (e.g., pip install 'datasets<3.0')."
        )

    tasks = list(tasks_override) if tasks_override is not None else list(cfg.eval.lm_eval.get("tasks", []))
    if not tasks:
        log.warning("No lm-eval tasks specified. Skipping lm-eval.")
        return {"ok": True, "skipped": True}

    device = str(cfg.eval.get("device", "cpu"))
    limit = cfg.eval.lm_eval.get("limit", None) or cfg.eval.get("max_samples", None)
    batch_size = int(cfg.eval.lm_eval.get("batch_size", 1))
    # Force trusting remote code for Hugging Face models during lm-eval.
    trust_remote_code = True

    model_args_parts = [
        f"pretrained={model_id}",
        f"tokenizer={model_id}",
        "dtype=float32",
        f"trust_remote_code={'True' if trust_remote_code else 'False'}",
    ]
    model_args = ",".join(model_args_parts)

    context_label = "no-context" if _is_no_context_run(run_name or "") else "with-context"
    log.info("Starting lm-eval: tasks=%s, limit=%s, batch_size=%s", tasks, limit, batch_size)
    supports_device = "device" in inspect.signature(evaluator.simple_evaluate).parameters
    eval_kwargs = {
        "model": "hf",
        "model_args": model_args,
        "batch_size": batch_size,
        "limit": limit,
    }
    if supports_device:
        eval_kwargs["device"] = device

    aggregate_results: Dict[str, Any] = {}
    progress_desc = f"lm-eval | {run_name or model_id} | {context_label}"
    with tqdm(total=len(tasks), desc=progress_desc, unit="task") as pbar:
        for task in tasks:
            pbar.set_postfix_str(f"task={task}")
            eval_kwargs["tasks"] = [task]
            results = evaluator.simple_evaluate(**eval_kwargs)
            if not aggregate_results:
                aggregate_results = {k: v for k, v in results.items() if k != "results"}
                aggregate_results["results"] = {}
                if "versions" in aggregate_results and isinstance(aggregate_results["versions"], dict):
                    aggregate_results["versions"] = dict(aggregate_results["versions"])
            aggregate_results["results"].update(results.get("results", {}))
            if "versions" in aggregate_results and isinstance(aggregate_results["versions"], dict):
                aggregate_results["versions"].update(results.get("versions", {}) or {})
            pbar.update(1)

    os.makedirs(save_dir, exist_ok=True)
    sanitized = _sanitize_for_json(aggregate_results)
    out_path = os.path.join(save_dir, "lm_eval_results.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(sanitized, f, indent=2)
    log.info("Saved lm-eval results to %s", out_path)

    task_results = aggregate_results.get("results", {})
    compact: Dict[str, Dict[str, float]] = {}
    for task, metrics in task_results.items():
        compact[str(task)] = {}
        for metric_name, value in metrics.items():
            if isinstance(value, numbers.Number):
                compact[str(task)][str(metric_name)] = float(value)
    summary: Dict[str, Any] = {
        "ok": True,
        "tasks": compact,
        "versions": _sanitize_for_json(aggregate_results.get("versions", {})),
    }
    return summary


def _load_jsonl(path: str) -> List[Dict[str, Any]]:
    """Load a JSONL file into a list of dictionaries."""
    data: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line))
    return data


def _extract_fields_from_row(
    row: Dict[str, Any],
    fields: Dict[str, str],
) -> Tuple[str, List[str], List[str], Optional[str]]:
    """Normalize raw dataset rows into question/contexts/ground_truth/response."""
    q_keys = [fields["question"], "query", "question", "question_text", "question_body", "query_text", "prompt"]
    question = next((row[k] for k in q_keys if k in row and row[k]), "")

    ctx_keys = [fields["contexts"], "passages", "psgs", "contexts", "context", "positive_ctxs", "ctxs", "passage", "paragraphs"]
    contexts: List[str] = []
    for key in ctx_keys:
        if key in row and row[key]:
            value = row[key]
            if isinstance(value, list):
                if value and isinstance(value[0], str):
                    contexts = [str(x) for x in value]
                else:
                    contexts = [
                        str(item[sub])
                        for item in value
                        if isinstance(item, dict)
                        for sub in ["text", "passage", "title", "body"]
                        if sub in item and item[sub]
                    ]
                break
            if isinstance(value, str):
                contexts = [value]
                break

    gt_keys = [fields["ground_truth"], "answers", "ground_truth", "answer", "target", "label", "labels"]
    ground_truth_values: List[str] = []
    for key in gt_keys:
        if key in row and row[key]:
            value = row[key]
            ground_truth_values = [str(x) for x in value] if isinstance(value, list) else [str(value)]
            break

    resp_keys = [fields["response"], "response", "pred", "prediction", "model_response", "answer"]
    response = next((row[k] for k in resp_keys if k in row and row[k]), None)

    question_text = str(question[0] if isinstance(question, list) and question else question or "")
    contexts_text = [str(context) for context in contexts]
    ground_truth_text = [str(value) for value in ground_truth_values]
    response_text = str(response) if response is not None else None
    return question_text, contexts_text, ground_truth_text, response_text


def _run_ragas(cfg: DictConfig, save_dir: str, model_id: str, run_name: str) -> Dict[str, Any]:
    """Run RAGAS metrics, optionally generating responses when missing."""
    from datasets import Dataset as HFDataset
    from ragas import evaluate as ragas_evaluate
    from ragas.metrics import answer_relevancy, context_precision, context_recall, faithfulness

    rconf = cfg.eval.get("ragas", {}) if cfg.eval else {}
    if not bool(rconf.get("enabled", False)):
        log.info("RAGAS disabled; skipping.")
        return {"ok": True, "skipped": True}

    dataset_path = rconf.get("dataset_path", None)
    dataset_name = rconf.get("dataset_name", None)
    split = rconf.get("split", None)

    fields = {
        "question": rconf.get("fields", {}).get("question", "question"),
        "contexts": rconf.get("fields", {}).get("contexts", "contexts"),
        "ground_truth": rconf.get("fields", {}).get("ground_truth", "ground_truth"),
        "response": rconf.get("fields", {}).get("response", "response"),
    }

    records: List[Dict[str, Any]] = []
    if dataset_path:
        if dataset_path.endswith(".jsonl"):
            records = _load_jsonl(dataset_path)
        else:
            with open(dataset_path, "r", encoding="utf-8") as f:
                payload = json.load(f)
            if isinstance(payload, dict) and "data" in payload:
                records = list(payload["data"])
            elif isinstance(payload, list):
                records = payload
            else:
                raise ValueError("Unsupported JSON structure for ragas dataset")
    elif dataset_name:
        import datasets as _datasets

        dataset = _datasets.load_dataset(dataset_name, split=split) if split else _datasets.load_dataset(dataset_name)
        records = dataset.to_list()  # type: ignore[assignment]
    else:
        raise ValueError("RAGAS dataset_path or dataset_name is required")

    limit = rconf.get("limit", None) or cfg.eval.get("max_samples", None)
    if isinstance(limit, int) and limit > 0:
        records = records[:limit]

    # Force trusting remote code for Hugging Face models during generation.
    trust_remote_code = True
    need_generate = bool(rconf.get("generate_responses", False))
    generation_cfg = rconf.get("generation", {})
    max_new_tokens = int(generation_cfg.get("max_new_tokens", 256))
    temperature = float(generation_cfg.get("temperature", 0.0))
    top_p = float(generation_cfg.get("top_p", 1.0))
    do_sample = bool(generation_cfg.get("do_sample", False))
    disable_context = _is_no_context_run(run_name)

    tokenizer = None
    model = None
    if need_generate:
        tokenizer, model = _load_hf_causal_model(
            model_id=model_id,
            device=str(cfg.eval.get("device", "cpu")),
            trust_remote_code=trust_remote_code,
        )

    prepared: List[Dict[str, Any]] = []
    for record in records:
        question, contexts, ground_truth_values, response = _extract_fields_from_row(record, fields)
        if (response is None or response == "") and need_generate:
            prompt = _build_generation_prompt(question, contexts, disable_context)
            gen_out = _generate_responses(
                [prompt],
                tokenizer,
                model,
                device=str(cfg.eval.get("device", "cpu")),
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                do_sample=do_sample,
            )
            response = gen_out[0] if gen_out else ""

        prepared.append(
            {
                "question": question,
                "contexts": [str(context) for context in (contexts or [])],
                "answer": str(response) if response is not None else "",
                "ground_truth": [str(value) for value in ground_truth_values],
            }
        )

    dataset = HFDataset.from_list(prepared)

    metric_names = [name for name in rconf.get("metrics", ["faithfulness", "answer_relevancy", "context_precision", "context_recall"])]
    metric_map = {
        "faithfulness": faithfulness,
        "answer_relevancy": answer_relevancy,
        "context_precision": context_precision,
        "context_recall": context_recall,
    }
    metrics = [metric_map[name] for name in metric_names if name in metric_map] or [faithfulness, answer_relevancy]

    def _metric_display_name(metric: Any) -> str:
        return getattr(metric, "name", None) or getattr(metric, "__name__", str(metric))

    log.info("Running RAGAS with metrics: %s on %d rows", [_metric_display_name(m) for m in metrics], len(prepared))
    ragas_result = ragas_evaluate(dataset, metrics=metrics)

    result_dict: Dict[str, Any] = {"ok": True, "num_rows": len(prepared)}
    per_row_records: List[Dict[str, Any]] = []

    if hasattr(ragas_result, "column_names") and hasattr(ragas_result, "to_list"):
        per_row_records = ragas_result.to_list()  # type: ignore[attr-defined]
        for metric in metrics:
            metric_name = _metric_display_name(metric)
            if metric_name in ragas_result.column_names:
                values = ragas_result[metric_name]
                result_dict[metric_name] = float(sum(values) / max(1, len(values)))
    elif hasattr(ragas_result, "to_dict"):
        per_row_records = ragas_result.to_dict("records")  # type: ignore[attr-defined]
        for metric in metrics:
            metric_name = _metric_display_name(metric)
            values = [row.get(metric_name) for row in per_row_records if isinstance(row.get(metric_name), (int, float))]
            if values:
                result_dict[metric_name] = float(sum(values) / len(values))
    elif hasattr(ragas_result, "results") or hasattr(ragas_result, "scores"):
        seq = getattr(ragas_result, "results", None) or getattr(ragas_result, "scores", None)
        per_row_records = list(seq)
        for metric in metrics:
            metric_name = _metric_display_name(metric)
            values = [row.get(metric_name) for row in per_row_records if isinstance(row, dict) and isinstance(row.get(metric_name), (int, float))]
            if values:
                result_dict[metric_name] = float(sum(values) / len(values))

    if per_row_records:
        os.makedirs(save_dir, exist_ok=True)
        per_row_json = os.path.join(save_dir, "ragas_per_row.json")
        with open(per_row_json, "w", encoding="utf-8") as f:
            for row in per_row_records:
                f.write(json.dumps(_sanitize_for_json(row)) + "\n")
        log.info("Saved RAGAS per-row results to %s", per_row_json)

    out_path = os.path.join(save_dir, "ragas_results.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(_sanitize_for_json(result_dict), f, indent=2)
    log.info("Saved RAGAS summary to %s", out_path)

    return result_dict


def _save_summary(all_results: Dict[str, Any], save_dir: str) -> None:
    """Persist a summary JSON for the evaluation run."""
    os.makedirs(save_dir, exist_ok=True)
    out_path = os.path.join(save_dir, "summary.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(_sanitize_for_json(all_results), f, indent=2)
    log.info("Saved evaluation summary to %s", out_path)


def evaluate_model(
    cfg: DictConfig,
    model_id: str,
    run_name: str,
    eval_root: Optional[str] = None,
    tasks_override: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """Evaluate a single model across lm-eval and optional RAGAS metrics."""
    eval_root_dir = eval_root or os.path.join(os.getcwd(), "eval")
    save_dir = os.path.join(eval_root_dir, run_name)
    os.makedirs(save_dir, exist_ok=True)

    results: Dict[str, Any] = {"model": model_id, "run_name": run_name}

    if cfg.get("eval") and cfg.eval.get("lm_eval", {}).get("enabled", True):
        lm_dir = os.path.join(save_dir, "lm_eval")
        lm_res = _run_lm_eval(cfg, lm_dir, model_id, run_name=run_name, tasks_override=tasks_override)
        results["lm_eval"] = lm_res
        metrics = _collect_accuracy_and_perplexity(lm_res)
        if metrics:
            results["metrics"] = metrics
    else:
        log.info("lm-eval disabled.")
        results["lm_eval"] = {"skipped": True}

    if cfg.get("eval") and cfg.eval.get("ragas", {}).get("enabled", False):
        ragas_dir = os.path.join(save_dir, "ragas")
        ragas_res = _run_ragas(cfg, ragas_dir, model_id, run_name)
        results["ragas"] = ragas_res
    else:
        log.info("RAGAS disabled or not configured.")
        results["ragas"] = {"skipped": True}

    _save_summary(results, save_dir)
    return results


def _print_single_summary(results: Dict[str, Any]) -> None:
    """Print a human-friendly summary for a single evaluation run."""
    model_id = results.get("model", "")
    print("\n=== Evaluation Summary ===")
    print(f"Model: {model_id}")

    lm_eval_res = results.get("lm_eval", {})
    if lm_eval_res.get("ok", False) and not lm_eval_res.get("skipped", False):
        print("- lm-eval: done (see lm_eval_results.json)")
    elif lm_eval_res.get("skipped", False):
        print("- lm-eval: skipped")
    else:
        print(f"- lm-eval: error: {lm_eval_res.get('error')}")

    ragas_res = results.get("ragas", {})
    if ragas_res.get("ok", False) and not ragas_res.get("skipped", False):
        ragas_metrics = {k: v for k, v in ragas_res.items() if isinstance(v, (int, float))}
        if ragas_metrics:
            print("- RAGAS:")
            for key, value in ragas_metrics.items():
                print(f"  {key}: {value:.4f}")
        else:
            print("- RAGAS: done (see ragas_results.json)")
    elif ragas_res.get("skipped", False):
        print("- RAGAS: skipped")
    else:
        print(f"- RAGAS: error: {ragas_res.get('error')}")

    metrics = results.get("metrics", {})
    if metrics.get("accuracy_avg") is not None:
        print(f"- accuracy_avg: {metrics['accuracy_avg']:.4f}")
    if metrics.get("perplexity_avg") is not None:
        print(f"- perplexity_avg: {metrics['perplexity_avg']:.4f}")

    def _render_table(headers: List[str], rows: List[List[Any]]) -> str:
        widths = [len(h) for h in headers]
        for row in rows:
            for i, cell in enumerate(row):
                widths[i] = max(widths[i], len(str(cell)))
        fmt = " | ".join("{:" + str(w) + "}" for w in widths)
        header_line = fmt.format(*headers)
        sep_line = "-+-".join("-" * w for w in widths)
        body = "\n".join(fmt.format(*[str(c) for c in row]) for row in rows)
        return "\n".join([header_line, sep_line, body]) if rows else header_line

    lm_tasks = results.get("lm_eval", {}).get("tasks", {})
    lm_rows: List[List[Any]] = []
    for task_name, task_metrics in lm_tasks.items():
        for metric_name, value in task_metrics.items():
            lm_rows.append([task_name, metric_name, f"{value:.4f}"])
    if lm_rows:
        print("\nLM-Eval Metrics:")
        print(_render_table(["task", "metric", "value"], lm_rows))

    ragas_res = results.get("ragas", {})
    ragas_rows: List[List[Any]] = []
    for key, value in ragas_res.items():
        if isinstance(value, (int, float)):
            ragas_rows.append(["ragas", key, f"{value:.4f}"])
    if ragas_rows:
        print("\nRAGAS Metrics:")
        print(_render_table(["component", "metric", "value"], ragas_rows))


def _summarize_baseline_runs(runs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Compact the baseline suite into a table of accuracy/perplexity averages."""
    rows: List[Dict[str, Any]] = []
    for res in runs:
        metrics = res.get("metrics", {})
        rows.append(
            {
                "baseline": res.get("baseline", res.get("run_name", res.get("model", ""))),
                "accuracy_avg": metrics.get("accuracy_avg"),
                "perplexity_avg": metrics.get("perplexity_avg"),
            }
        )
    return rows


def evaluate_baseline_suite(cfg: DictConfig) -> Dict[str, Any]:
    """Evaluate a configured suite of baseline models."""
    baseline_cfg = cfg.eval.get("baselines", {})
    baselines = baseline_cfg.get("models", [])
    if not baselines:
        raise ValueError("eval.baselines.enabled is true but no baselines are configured.")

    tasks = list(dict.fromkeys(list(baseline_cfg.get("accuracy_tasks", [])) + list(baseline_cfg.get("perplexity_tasks", []))))
    eval_root = os.path.join(os.getcwd(), "eval")
    suite_name = baseline_cfg.get("suite_name", "baselines")
    suite_dir = os.path.join(eval_root, suite_name)
    os.makedirs(suite_dir, exist_ok=True)

    runs: List[Dict[str, Any]] = []
    for entry in baselines:
        model_id = entry["model_id"]
        run_name = entry.get("name", model_id).replace("/", "-")
        result = evaluate_model(cfg, model_id, run_name, eval_root=suite_dir, tasks_override=tasks)
        result["baseline"] = entry.get("display_name", run_name)
        runs.append(result)

    rows = _summarize_baseline_runs(runs)
    suite_summary = {
        "suite": suite_name,
        "tasks": tasks,
        "runs": runs,
        "metrics": rows,
    }
    _save_summary(suite_summary, suite_dir)

    print("\n=== Baseline Suite Summary ===")
    for row in rows:
        line = f"- {row['baseline']}"
        if row.get("accuracy_avg") is not None:
            line += f" | acc: {row['accuracy_avg']:.4f}"
        if row.get("perplexity_avg") is not None:
            line += f" | ppl: {row['perplexity_avg']:.4f}"
        print(line)

    return suite_summary


def evaluate(cfg: DictConfig) -> Dict[str, Any]:
    """Entry point used by Hydra to launch evaluation flows."""
    log.info("Running evaluation with config: %s", OmegaConf.to_container(cfg, resolve=True))

    if cfg.get("eval") and cfg.eval.get("baselines", {}).get("enabled", False):
        result = evaluate_baseline_suite(cfg)
        log.info("Baseline evaluation finished.")
        return result

    model_id = cfg.model.name
    if not model_id:
        raise ValueError("cfg.model.name is required for evaluation")

    run_name = cfg.get("run_name", None) or model_id.replace("/", "-")
    results = evaluate_model(cfg, model_id, run_name)
    _print_single_summary(results)
    log.info("Evaluation finished.")
    return results


@dataclass
class EvalApp:
    """Hydra application wrapper for evaluation."""

    def __init__(self, **_: Any) -> None:
        pass

    def build_command(self, cfg: DictConfig) -> List[str]:
        """No external command required for in-process execution."""
        return []

    def run(self, cfg: DictConfig) -> Dict[str, Any]:
        """Main entry point; only place that catches errors."""
        try:
            return evaluate(cfg.app)
        except Exception as exc:
            log.exception("Evaluation failed")
            return {"ok": False, "error": str(exc)}
