import logging
import math
import os
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Tuple
from collections import defaultdict

import torch
from hydra.core.hydra_config import HydraConfig
from datasets import interleave_datasets, load_dataset
from omegaconf import DictConfig, OmegaConf
from torch.utils.tensorboard import SummaryWriter
from transformers import AutoTokenizer, get_cosine_schedule_with_warmup, get_linear_schedule_with_warmup

from refrag.model.config_refrag import RefragConfig
from refrag.model.modeling_refrag import RefragModel
from refrag.model.tokenization_refrag import RefragTokenizer

log = logging.getLogger(__name__)


def _bf16_supported(device: torch.device) -> bool:
    if device.type == "cuda":
        return torch.cuda.is_available() and getattr(torch.cuda, "is_bf16_supported", lambda: False)()
    cpu_backend = getattr(torch.backends, "cpu", None)
    if device.type == "cpu" and cpu_backend:
        return getattr(cpu_backend, "is_bf16_supported", lambda: False)()
    return False


def _build_stage_mix(stage_cfg: Dict) -> Tuple[Dict[int, int], List[int], List[int]]:
    mix = {int(k): int(v) for k, v in stage_cfg.mix.items()}
    factors = sorted(mix.keys())
    weights = [mix[f] for f in factors]
    return mix, factors, weights


def _cycle_iterable(ds: Iterable[Dict]) -> Iterator[Dict]:
    """Yield forever from a (re-iterable) dataset without rebuilding it."""
    iterator = iter(ds)
    while True:
        try:
            yield next(iterator)
        except StopIteration:
            iterator = iter(ds)


def _pack_tokens(token_ids: List[int], compression: int, pad_id: int) -> Tuple[List[List[int]], List[List[int]]]:
    rows: List[List[int]] = []
    masks: List[List[int]] = []
    for i in range(0, len(token_ids), compression):
        chunk = token_ids[i : i + compression]
        mask = [1] * len(chunk)
        if len(chunk) < compression:
            pad_len = compression - len(chunk)
            chunk = chunk + [pad_id] * pad_len
            mask = mask + [0] * pad_len
        rows.append(chunk)
        masks.append(mask)
    if not rows:
        rows = [[pad_id] * compression]
        masks = [[0] * compression]
    return rows, masks


def _escape_md_cell(text: str) -> str:
    return text.replace("|", "\\|").replace("\n", " ").replace("\r", " ").strip()


def _format_seconds(secs: float) -> str:
    secs_int = max(0, int(secs))
    mins, sec = divmod(secs_int, 60)
    hrs, mins = divmod(mins, 60)
    return f"{hrs:02d}:{mins:02d}:{sec:02d}"


@dataclass
class ReconstructionCurriculumApp:
    """Hydra app to train the reconstruction task with curriculum learning (Table 8/12 in paper)."""

    def __init__(self, **cfg: DictConfig):
        # Store raw app config; we resolve against the Hydra root cfg in run().
        self.init_app_cfg = cfg if isinstance(cfg, DictConfig) else OmegaConf.create(cfg)
        self._base_dataset = None  # cache to avoid re-resolving SlimPajama shards

    def _resolve_app_cfg(self, cfg: DictConfig) -> DictConfig:
        app_cfg = getattr(cfg, "app", None) or self.init_app_cfg
        return app_cfg if isinstance(app_cfg, DictConfig) else OmegaConf.create(app_cfg)

    # Dataset utilities -------------------------------------------------
    def _get_base_dataset(self, ds_cfg: DictConfig):
        """Load the streaming dataset once; reuse across domain streams to skip repeated resolution."""
        if self._base_dataset is None:
            hf_token = ds_cfg.get("hf_token") or os.environ.get("HF_TOKEN")
            self._base_dataset = load_dataset(
                ds_cfg.name,
                split=ds_cfg.split,
                streaming=bool(ds_cfg.get("streaming", True)),
                trust_remote_code=bool(ds_cfg.get("trust_remote_code", False)),
                token=hf_token,
            )
        return self._base_dataset

    def _build_domain_stream(self, base_cfg: DictConfig, domain_cfg: DictConfig) -> Iterable[Dict]:
        ds = self._get_base_dataset(base_cfg)
        filter_field = base_cfg.get("filter_field")
        allowed = set(domain_cfg.get("values", []))
        if filter_field and allowed:
            ds = ds.filter(lambda ex: ex.get(filter_field) in allowed)
        shuffle_buf = int(base_cfg.get("shuffle_buffer", 0))
        if shuffle_buf > 0:
            ds = ds.shuffle(seed=int(base_cfg.get("seed", 0)), buffer_size=shuffle_buf)
        if domain_cfg.get("max_samples"):
            ds = ds.take(int(domain_cfg.max_samples))
        ds = ds.map(lambda ex: {**ex, "__domain": domain_cfg.get("name", "unknown")})
        return ds

    def _mixed_stream(self, app_cfg: DictConfig) -> Iterable[Dict]:
        ds_cfg = app_cfg.dataset
        domains = ds_cfg.get("domains", [])
        if not domains:
            raise ValueError("dataset.domains must be configured with at least one domain")
        streams = []
        probs = []
        for dom in domains:
            streams.append(self._build_domain_stream(ds_cfg, dom))
            probs.append(float(dom.get("weight", 1.0)))
        if len(streams) == 1:
            return streams[0]
        return interleave_datasets(streams, probabilities=probs, seed=int(ds_cfg.get("seed", 0)))

    def _run_eval(
        self,
        app_cfg: DictConfig,
        encoder_tok,
        decoder_tok,
        refrag_tok,
        model: RefragModel,
        device: torch.device,
    ):
        """Lightweight eval loop to track reconstruction loss during training."""
        eval_batches = int(app_cfg.train.get("eval_batches", 0) or 0)
        if eval_batches <= 0:
            return None

        stage_cfg = app_cfg.curriculum.stages[0]
        stage_mix, factors, weights = _build_stage_mix(stage_cfg)
        eval_stream = self._mixed_stream(app_cfg)
        eval_iter = iter(eval_stream)
        eval_losses: List[float] = []
        domain_losses: Dict[str, List[float]] = {}

        was_training = model.training
        model.eval()
        with torch.no_grad():
            for _ in range(eval_batches):
                try:
                    example = next(eval_iter)
                except StopIteration:
                    break
                text = example.get(app_cfg.dataset.text_key, "")
                if not isinstance(text, str) or not text.strip():
                    continue
                domain = example.get("__domain", "unknown")
                chunk_factor = random.choices(factors, weights=weights, k=1)[0]
                target_tokens = int(chunk_factor * app_cfg.curriculum.base_tokens_per_chunk)
                max_tokens = int(app_cfg.dataset.get("max_tokens", target_tokens))
                enc = encoder_tok(text, add_special_tokens=False, truncation=True, max_length=max_tokens)
                token_ids = enc.get("input_ids") or []
                if len(token_ids) < int(app_cfg.dataset.get("min_tokens", app_cfg.curriculum.base_tokens_per_chunk)):
                    continue
                token_ids = token_ids[:target_tokens]
                if len(token_ids) < target_tokens:
                    pad_id = encoder_tok.pad_token_id or 0
                    token_ids = token_ids + [pad_id] * (target_tokens - len(token_ids))

                packed_tokens, attention_mask = _pack_tokens(token_ids, model.config.compression, encoder_tok.pad_token_id or 0)
                seg = {
                    "type": "encoder",
                    "raw_text": text,
                    "tokens": token_ids,
                    "packed_tokens": packed_tokens,
                    "attention_mask": attention_mask,
                    "meta": {"domain": domain, "chunk_factor": chunk_factor},
                }

                logits_list = model.encoder_projected_logits([seg])
                targets_list = refrag_tok.build_encoder_targets([seg], decoder_tok, getattr(model.decoder.config, "vocab_size", 50257))
                if not logits_list or not targets_list:
                    continue

                logits = logits_list[0][1].to(device)
                targets = targets_list[0][1].to(device)
                logp = torch.nn.functional.log_softmax(logits, dim=-1)
                loss = torch.nn.functional.kl_div(logp, targets, reduction="batchmean")
                loss_value = float(loss.item())
                eval_losses.append(loss_value)
                domain_losses.setdefault(domain, []).append(loss_value)

        if was_training:
            model.train()

        if not eval_losses:
            return None
        avg_loss = sum(eval_losses) / len(eval_losses)
        domain_stats: Dict[str, Tuple[float, float]] = {}
        for dom, vals in domain_losses.items():
            dom_avg = sum(vals) / len(vals)
            domain_stats[dom] = (dom_avg, math.exp(min(20.0, dom_avg)))
        return avg_loss, math.exp(min(20.0, avg_loss)), domain_stats

    # Training loop ----------------------------------------------------
    def run(self, cfg: DictConfig) -> int:
        app_cfg = self._resolve_app_cfg(cfg)
        device = torch.device(cfg.device if "device" in cfg else "cpu")
        use_bf16 = _bf16_supported(device)
        torch_dtype = torch.bfloat16 if use_bf16 else torch.float32

        # Model setup
        rcfg = RefragConfig(
            encoder_name_or_path=app_cfg.model.encoder,
            decoder_name_or_path=app_cfg.model.decoder,
            projector_hidden_dim=app_cfg.model.get("projector_hidden_dim", 768),
            compression=int(app_cfg.model.get("compression", 8)),
            pad_token_id=app_cfg.model.get("pad_token_id", 0),
            training_model=app_cfg.model.get("training_model", "pe"),
        )
        model = RefragModel.from_config(rcfg)
        model = model.to(device=device, dtype=torch_dtype)
        # Freeze decoder for reconstruction
        for p in model.decoder.parameters():
            p.requires_grad = False
        model.decoder.eval()

        encoder_tok = AutoTokenizer.from_pretrained(rcfg.encoder_name_or_path)
        decoder_tok = AutoTokenizer.from_pretrained(rcfg.decoder_name_or_path)
        if encoder_tok.pad_token is None:
            encoder_tok.pad_token = encoder_tok.eos_token
            encoder_tok.pad_token_id = encoder_tok.eos_token_id
        refrag_tok = RefragTokenizer(encoder_tok, decoder_tok, compression=rcfg.compression, encoder_pad_token_id=encoder_tok.pad_token_id or 0)

        trainable_params = [p for p in model.parameters() if p.requires_grad]
        optimizer = torch.optim.AdamW(trainable_params, lr=float(app_cfg.train.lr))

        curriculum = app_cfg.curriculum
        stage_scale = float(app_cfg.train.get("stage_scale", 1.0))
        est_total_samples = sum(max(1, int(stage.get("total_samples", sum(stage.mix.values())) * stage_scale)) for stage in curriculum.stages)
        total_steps = max(1, est_total_samples // int(app_cfg.train.batch_size))
        if app_cfg.train.get("max_steps") is not None:
            total_steps = min(total_steps, int(app_cfg.train.max_steps))

        warmup_steps = int(app_cfg.train.get("warmup_steps", 0))
        if warmup_steps <= 0 and app_cfg.train.get("warmup_ratio") is not None:
            warmup_steps = max(1, int(total_steps * float(app_cfg.train.warmup_ratio)))

        scheduler_type = app_cfg.train.get("scheduler", "cosine")
        if scheduler_type == "linear":
            scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps)
        else:
            scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps)
        eval_every = int(app_cfg.train.get("eval_every", 0) or 0)
        text_log_every = int(app_cfg.train.get("log_text_every", 0) or 0)

        hydra_run_dir = None
        try:
            hydra_run_dir = Path(HydraConfig.get().runtime.output_dir)
        except Exception:
            try:
                hydra_run_dir = Path(HydraConfig.get().run.dir)
            except Exception:
                hydra_run_dir = None

        output_dir_cfg = app_cfg.train.get("output_dir")
        if output_dir_cfg is not None:
            output_dir = Path(output_dir_cfg)
        else:
            output_dir = (hydra_run_dir or Path.cwd()) / "reconstruction"
        output_dir.mkdir(parents=True, exist_ok=True)

        log_dir_cfg = app_cfg.train.get("log_dir")
        log_dir = Path(log_dir_cfg) if log_dir_cfg is not None else output_dir / "tf"
        log_dir.mkdir(parents=True, exist_ok=True)
        writer = SummaryWriter(log_dir=str(log_dir))

        mixed_stream = self._mixed_stream(app_cfg)
        mixed_iter = _cycle_iterable(mixed_stream)
        global_step = 0
        losses: List[float] = []
        token_budget = {dom.get("name", f"domain_{idx}"): dom.get("token_budget") for idx, dom in enumerate(app_cfg.dataset.domains)}
        domain_tokens: Dict[str, int] = {dom: 0 for dom in token_budget}
        total_tokens_processed = 0
        start_time = time.time()
        text_examples: List[str] = []
        text_example_idx = 0
        chunk_loss_history: Dict[int, List[float]] = defaultdict(list)
        chunk_ppl_history: Dict[int, List[float]] = defaultdict(list)

        for stage_idx, stage_cfg in enumerate(curriculum.stages):
            stage_mix, factors, weights = _build_stage_mix(stage_cfg)
            stage_total = max(1, int(stage_cfg.get("total_samples", sum(stage_mix.values())) * stage_scale))
            stage_steps = max(1, stage_total // int(app_cfg.train.batch_size))
            log.info("Stage %s: %d steps (scale=%.2f) mix=%s", stage_cfg.get("name", stage_idx + 1), stage_steps, stage_scale, stage_mix)
            for _ in range(stage_steps):
                if app_cfg.train.get("max_steps") is not None and global_step >= int(app_cfg.train.max_steps):
                    break
                if all(token_budget.get(dom, None) and domain_tokens.get(dom, 0) >= token_budget[dom] for dom in token_budget):
                    log.info("Token budgets exhausted; stopping early at step %d", global_step)
                    break

                example = next(mixed_iter)
                domain = example.get("__domain", "unknown")
                if domain in token_budget and token_budget[domain] and domain_tokens.get(domain, 0) >= token_budget[domain]:
                    continue
                text = example.get(app_cfg.dataset.text_key, "")
                if not isinstance(text, str) or not text.strip():
                    continue

                chunk_factor = random.choices(factors, weights=weights, k=1)[0]
                target_tokens = int(chunk_factor * curriculum.base_tokens_per_chunk)
                max_tokens = int(app_cfg.dataset.get("max_tokens", target_tokens))
                enc = encoder_tok(text, add_special_tokens=False, truncation=True, max_length=max_tokens)
                token_ids = enc.get("input_ids") or []
                if len(token_ids) < int(app_cfg.dataset.get("min_tokens", curriculum.base_tokens_per_chunk)):
                    continue
                token_ids = token_ids[:target_tokens]
                if len(token_ids) < target_tokens:
                    pad_id = encoder_tok.pad_token_id or 0
                    token_ids = token_ids + [pad_id] * (target_tokens - len(token_ids))
                text_snippet = text.strip().replace("\n", " ")[:512]

                packed_tokens, attention_mask = _pack_tokens(token_ids, rcfg.compression, encoder_tok.pad_token_id or 0)
                seg = {
                    "type": "encoder",
                    "raw_text": text,
                    "tokens": token_ids,
                    "packed_tokens": packed_tokens,
                    "attention_mask": attention_mask,
                    "meta": {"domain": example.get("__domain", "unknown"), "chunk_factor": chunk_factor},
                }

                logits_list = model.encoder_projected_logits([seg])
                targets_list = refrag_tok.build_encoder_targets([seg], decoder_tok, getattr(model.decoder.config, "vocab_size", 50257))
                if not logits_list or not targets_list:
                    continue

                logits = logits_list[0][1].to(device)
                targets = targets_list[0][1].to(device)
                logp = torch.nn.functional.log_softmax(logits, dim=-1)
                loss = torch.nn.functional.kl_div(logp, targets, reduction="batchmean")
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(trainable_params, float(app_cfg.train.grad_clip))
                optimizer.step()
                scheduler.step()

                loss_value = float(loss.item())
                losses.append(loss_value)
                global_step += 1
                perplexity = math.exp(min(20.0, loss_value))
                writer.add_scalar("reconstruction/loss", loss_value, global_step)
                writer.add_scalar("reconstruction/perplexity_proxy", perplexity, global_step)
                writer.add_scalar(f"reconstruction/perplexity_domain/{domain}", perplexity, global_step)
                writer.add_scalar(f"chunks/perplexity/{chunk_factor}", perplexity, global_step)
                writer.add_scalar("reconstruction/chunk_factor", chunk_factor, global_step)
                writer.add_scalar("reconstruction/stage", stage_idx + 1, global_step)
                chunk_loss_history[chunk_factor].append(loss_value)
                chunk_ppl_history[chunk_factor].append(perplexity)
                if len(chunk_loss_history[chunk_factor]) > 200:
                    chunk_loss_history[chunk_factor] = chunk_loss_history[chunk_factor][-200:]
                    chunk_ppl_history[chunk_factor] = chunk_ppl_history[chunk_factor][-200:]

                if domain in domain_tokens and token_budget.get(domain):
                    domain_tokens[domain] += target_tokens
                    writer.add_scalar(f"tokens/{domain}", domain_tokens[domain], global_step)
                total_tokens_processed += target_tokens
                elapsed = time.time() - start_time
                tokens_per_sec = total_tokens_processed / max(1e-6, elapsed)
                writer.add_scalar("reconstruction/tokens_per_sec", tokens_per_sec, global_step)
                if text_log_every:
                    text_example_idx += 1
                    sample_no = text_example_idx
                    context_cell = _escape_md_cell(text_snippet)

                    # Targets: decode encoder tokens per packed row respecting mask (skip padding).
                    target_chunks: List[str] = []
                    row_lengths: List[int] = []
                    for row_ids, row_mask in zip(packed_tokens, attention_mask):
                        trimmed = [tid for tid, m in zip(row_ids, row_mask) if m]
                        row_lengths.append(len(trimmed))
                        target_chunks.append(encoder_tok.decode(trimmed, skip_special_tokens=True))
                    num_rows = max(1, len(target_chunks))
                    target_text = " | ".join(_escape_md_cell(tc) for tc in target_chunks if tc) or "(n/a)"

                    # Decode obtained tokens per row from logits (argmax).
                    obtained_chunks: List[str] = []
                    if logits is not None and logits.numel() > 0:
                        preds = logits.argmax(dim=-1).tolist()
                        if isinstance(preds, list):
                            if preds and isinstance(preds[0], list):
                                flat_preds = preds
                            else:
                                flat_preds = [[p] for p in (preds if isinstance(preds, list) else [])]
                            for idx, row_preds in enumerate(flat_preds[:num_rows]):
                                trim_len = row_lengths[idx] if idx < len(row_lengths) else None
                                trimmed_preds = row_preds[:trim_len] if trim_len else row_preds
                                obtained_chunks.append(decoder_tok.decode(trimmed_preds, skip_special_tokens=True))
                    obtained_text = " | ".join(_escape_md_cell(oc) for oc in obtained_chunks if oc) if obtained_chunks else "(n/a)"

                    row = f"| {sample_no} | {context_cell} | ***{target_text}*** | ***{obtained_text}*** |"
                    text_examples.append(row)
                    if global_step % text_log_every == 0 or len(text_examples) >= 10:
                        samples = text_examples[-10:]
                        header = "| Sample No | Context | Target | Obtained |\n| --- | --- | --- | --- |"
                        writer.add_text("reconstruction/examples", "\n".join([header] + samples), global_step)
                        text_examples.clear()
                if app_cfg.train.get("log_every") and global_step % int(app_cfg.train.log_every) == 0:
                    window = max(1, len(losses) // 4)
                    avg_loss = sum(losses[-window:]) / max(1, len(losses[-window:]))
                    domain_tok = domain_tokens.get(domain, 0)
                    domain_budget = token_budget.get(domain)
                    budget_str = f"{domain_tok}/{domain_budget}" if domain_budget else str(domain_tok)
                    step_time = elapsed / max(1, global_step)
                    eta_seconds = step_time * max(0, total_steps - global_step)
                    for cf, values in chunk_loss_history.items():
                        if values:
                            writer.add_histogram(f"chunks/loss_hist/{cf}", torch.tensor(values), global_step)
                            writer.add_scalar(f"chunks/loss_avg/{cf}", sum(values) / len(values), global_step)
                    for cf, values in chunk_ppl_history.items():
                        if values:
                            writer.add_histogram(f"chunks/perplexity_hist/{cf}", torch.tensor(values), global_step)
                            writer.add_scalar(f"chunks/perplexity_avg/{cf}", sum(values) / len(values), global_step)
                    log.info(
                        "step=%d stage=%s domain=%s chunk=%d loss=%.4f ppl=%.2f avg=%.4f lr=%.2e tokens=%s toks/s=%.1f eta=%s",
                        global_step,
                        stage_cfg.get("name", stage_idx + 1),
                        domain,
                        chunk_factor,
                        loss_value,
                        perplexity,
                        avg_loss,
                        optimizer.param_groups[0]["lr"],
                        budget_str,
                        tokens_per_sec,
                        _format_seconds(eta_seconds),
                    )

                if app_cfg.train.get("save_steps") and global_step % int(app_cfg.train.save_steps) == 0:
                    ckpt_dir = output_dir / f"checkpoint-{global_step}"
                    ckpt_dir.mkdir(parents=True, exist_ok=True)
                    model.save_pretrained(str(ckpt_dir))

                if eval_every and global_step % eval_every == 0:
                    eval_result = self._run_eval(app_cfg, encoder_tok, decoder_tok, refrag_tok, model, device)
                    if eval_result is not None:
                        eval_loss, eval_ppl, eval_domains = eval_result
                        writer.add_scalar("reconstruction/eval_loss", eval_loss, global_step)
                        writer.add_scalar("reconstruction/eval_perplexity_proxy", eval_ppl, global_step)
                        for dom, (dom_loss, dom_ppl) in eval_domains.items():
                            writer.add_scalar(f"reconstruction/eval_perplexity_domain/{dom}", dom_ppl, global_step)
                            writer.add_scalar(f"reconstruction/eval_loss_domain/{dom}", dom_loss, global_step)
                        domain_msg = " ".join([f"{d}:ppl={dom_ppl:.2f}" for d, (_, dom_ppl) in eval_domains.items()])
                        log.info("eval step=%d loss=%.4f ppl=%.2f domains=[%s]", global_step, eval_loss, eval_ppl, domain_msg)

                if app_cfg.train.get("max_steps") is not None and global_step >= int(app_cfg.train.max_steps):
                    break

        # Final save + logs
        model.save_pretrained(str(output_dir))
        writer.add_text("run_summary", f"finished_steps={global_step}")
        writer.flush()
        writer.close()
        log.info("Finished reconstruction training, steps=%d, saved to %s", global_step, output_dir)
        return 0
