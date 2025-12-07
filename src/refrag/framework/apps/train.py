import os
import time

from datasets import interleave_datasets, load_dataset
from omegaconf import DictConfig, OmegaConf
from transformers import AutoModelForSequenceClassification, AutoTokenizer, AutoModelForCausalLM, TrainingArguments, get_linear_schedule_with_warmup, get_cosine_schedule_with_warmup
import torch
import logging

from torch.utils.tensorboard import SummaryWriter
from dataclasses import dataclass
from refrag.model.modeling_refrag import RefragModel
from refrag.model.config_refrag import RefragConfig
from trl import SFTTrainer
from typing import List


def _bf16_supported(device: torch.device) -> bool:
    """Best-effort bf16 capability check for the target device."""
    if device.type == "cuda":
        return torch.cuda.is_available() and getattr(torch.cuda, "is_bf16_supported", lambda: False)()
    cpu_backend = getattr(torch.backends, "cpu", None)
    if device.type == "cpu" and cpu_backend:
        return getattr(cpu_backend, "is_bf16_supported", lambda: False)()
    return False


def to_text(example):
    return {"text": format_example(example)}


log = logging.getLogger(__name__)


def format_example(example) -> str:
    msgs = example.get("messages", [])
    turns: List[str] = []
    for m in msgs:
        role = (m.get("role") or "").strip()
        content = (m.get("content") or "").strip()
        if not content:
            continue
        if role == "user":
            turns.append(f"User: {content}")
        else:
            turns.append(f"Assistant: {content}")
    return "\n".join(turns) + "\nAssistant:"  # ensure model continues as assistant


@dataclass
class TrainerApp:
    def __init__(self, **cfg: DictConfig) -> None:
        self.cfg = cfg

    def _build_datamix(self, cfg_app: DictConfig):
        # Assume cfg_app.datamix.datasets is correctly configured
        ds_list = []
        probs = []
        for d in cfg_app.datamix.datasets:
            ds = load_dataset(d.name, split=d.split)
            limit = d.get("max_samples") or cfg_app.datamix.get("max_samples_per_dataset")
            if limit:
                ds = ds.select(range(min(len(ds), int(limit))))
            ds_list.append(ds)
            probs.append(d.weight)
        return interleave_datasets(ds_list, probabilities=probs, seed=42) if len(ds_list) > 1 else ds_list[0]

    def run(self, cfg: DictConfig) -> int:
        cfg_app = cfg.app
        model_name = cfg_app.model_name
        target_device = torch.device(cfg.device)
        use_bf16 = _bf16_supported(target_device)
        torch_dtype = torch.bfloat16 if use_bf16 else None

        model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch_dtype)
        model = model.to(target_device)
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            tokenizer.pad_token_id = tokenizer.eos_token_id

        raw_ds = self._build_datamix(cfg_app)
        train_ds = raw_ds.map(to_text, remove_columns=list(raw_ds.column_names))

        training_args_cfg = OmegaConf.to_container(cfg_app.training_args, resolve=True)
        if use_bf16:
            training_args_cfg.setdefault("bf16", True)
            training_args_cfg.pop("fp16", None)
        else:
            training_args_cfg.pop("bf16", None)

        training_args = TrainingArguments(**training_args_cfg)
        # Ensure tensorboard logging
        if not training_args.report_to:
            training_args.report_to = ["tensorboard"]

        trainer = SFTTrainer(
            model=model,
            tokenizer=tokenizer,
            train_dataset=train_ds,
            formatting_func=format_example,
            args=training_args,

        )

        log.info("Starting TRL SFT training (model=%s)", model_name)
        trainer.train()

        if trainer.is_world_process_zero:
            save_dir = training_args.output_dir or f"outputs/sft-{int(time.time())}"
            os.makedirs(save_dir, exist_ok=True)
            trainer.save_model(save_dir)
            tokenizer.save_pretrained(save_dir)
            log.info("Saved final model to %s", save_dir)
        return 0


@dataclass
class RefragTrainerApp:

    def __init__(self, **cfg: DictConfig):
        self.cfg = cfg.get("app", cfg)

    def run(self, cfg: DictConfig) -> int:

        """Train function that accepts a Hydra DictConfig. This is NOT decorated with @hydra.main so it can be called
        from a central Hydra driver (e.g., `old-run.py`) without nested Hydra initializations.
        """
        model_name = self.cfg.model_name
        steps = int(self.cfg.steps_per_epoch)
        lr = float(cfg.lr)
        batch_size = int(cfg.batch_size)
        save_dir = os.getenv("MODEL_SAVE_DIR", cfg.get("save_dir", "./outputs/refrag_model"))

        log.info("Loading model %s", model_name)
        target_device = torch.device(cfg.get("device", "cuda" if torch.cuda.is_available() else "cpu"))
        use_bf16 = _bf16_supported(target_device)
        torch_dtype = torch.bfloat16 if use_bf16 else torch.float32

        # If model_name == 'refrag' we expect the conf/model to provide encoder/decoder
        if model_name == "refrag":
            # Build RefragConfig from Hydra config
            rcfg = RefragConfig(
                encoder_name_or_path=cfg.model.encoder if "encoder" in cfg.model else "prajjwal1/bert-mini",
                decoder_name_or_path=cfg.model.decoder if "decoder" in cfg.model else "sshleifer/tiny-gpt2",
                projector_hidden_dim=cfg.model.get("projector_hidden_dim", 768),
                compression=cfg.train.get("compression", 4),
                pad_token_id=cfg.model.get("pad_token_id", 0),
                training_model=cfg.model.get("training_model", "ped"),
            )
            model = RefragModel.from_config(rcfg)
            model = model.to(device=target_device, dtype=torch_dtype)
        else:
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            model = AutoModelForSequenceClassification.from_pretrained(
                model_name, num_labels=2, torch_dtype=torch_dtype
            )
            model = model.to(device=target_device)

        device = target_device
        model.train()

        # Respect training flags: ensure parameters have requires_grad as configured
        # (RefragModel already applied training flags on initialization)

        # Dummy data using decoder tokenizer when available
        if hasattr(model, "decoder"):
            # Use decoder tokenizer to produce training inputs
            decoder_tok = AutoTokenizer.from_pretrained(model.config.decoder_name_or_path)
            inputs = decoder_tok(["Hello world"] * batch_size, padding=True, truncation=True, return_tensors="pt")
            input_kwargs = {k: v for k, v in inputs.items()}
            labels = torch.zeros(batch_size, dtype=torch.long)
            optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=lr)
        else:
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            inputs = tokenizer(["Hello world"] * batch_size, padding=True, truncation=True, return_tensors="pt")
            labels = torch.zeros(batch_size, dtype=torch.long)
            optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=lr)

        # Experiment tracking
        run_name = cfg.get("run_name", None) or f"refrag-{int(time.time())}"

        tb_writer = SummaryWriter(log_dir=os.path.join(save_dir, "tb"))

        # Build a tiny placeholder dataset loader (user should replace with real dataset)
        # For now we construct batches of decoder-only input using decoder tokenizer
        if hasattr(model, "decoder"):
            decoder_tok = AutoTokenizer.from_pretrained(model.config.decoder_name_or_path)
        else:
            decoder_tok = AutoTokenizer.from_pretrained(model_name)

        def data_generator():
            # yields dict with input_ids tensor and labels tensor
            while True:
                enc = decoder_tok(["Hello world this is a training example"] * batch_size, padding=True,
                                  truncation=True, return_tensors="pt")
                yield enc

        gen = data_generator()

        # optimizer & scheduler
        trainable_params = [p for p in model.parameters() if p.requires_grad]
        optimizer = torch.optim.AdamW(trainable_params, lr=lr)
        total_steps = steps
        warmup_steps = int(cfg.train.get("warmup_steps", max(1, total_steps // 10)))
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, total_steps - warmup_steps))

        # If user requested linear warmup (use transformers helper for accuracy)
        use_transformers_warmup = cfg.train.get("use_transformers_warmup", True)
        scheduler_type = cfg.train.get("scheduler", "cosine")
        if use_transformers_warmup and scheduler_type == "cosine":
            scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps,
                                                        num_training_steps=total_steps)
        elif use_transformers_warmup and scheduler_type == "linear":
            scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps,
                                                        num_training_steps=total_steps)
        else:
            # fallback to PyTorch cosine annealing after warmup
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, total_steps - warmup_steps))

        grad_clip = float(cfg.train.get("grad_clip", 1.0))

        kld_loss_fn = torch.nn.KLDivLoss(reduction="batchmean")

        for step in range(steps):
            # For demonstration, create text containing an encoder doc so we exercise encoder projection
            sample_text = """Hello <doc id="1">This is encoder content to be projected</doc> world"""
            # Parse and tokenize into segments
            from refrag.model.tokenization_refrag import from_pretrained as tokenizer_factory

            # build a refrag tokenizer using model's encoder/decoder names
            reconf = model.config if isinstance(model, RefragModel) else None
            if reconf:
                rtok = tokenizer_factory(reconf.encoder_name_or_path, reconf.decoder_name_or_path,
                                         compression=reconf.compression)
            else:
                # fallback to decoder-only tokenizer
                rtok = tokenizer_factory(model_name, model_name, compression=4)

            segments = rtok.parse_and_tokenize(sample_text)

            # get encoder projected logits
            enc_logits = None
            enc_targets = None
            if isinstance(model, RefragModel):
                enc_logits = model.encoder_projected_logits(segments)
                vocab_size = getattr(model.decoder.config, "vocab_size", 50257)
                enc_targets = rtok.build_encoder_targets(segments, rtok.decoder_tokenizer, vocab_size)

            batch = next(gen)
            input_ids = batch["input_ids"]
            attention_mask = batch.get("attention_mask", None)

            if hasattr(model, "decoder"):
                model.decoder.train()
                outputs = model.decoder(input_ids=input_ids.to(model.decoder.device), attention_mask=(
                    attention_mask.to(model.decoder.device) if attention_mask is not None else None),
                                        labels=input_ids.to(model.decoder.device))
                # LM loss (CrossEntropy) from HF model
                loss_ce = outputs.loss
                loss = loss_ce
                # KLD over encoder rows
                if enc_logits and enc_targets:
                    # enc_logits: list of (seg_idx, tensor(num_rows, vocab))
                    kld_sum = 0.0
                    kcount = 0
                    for (sidx, logits), (tidx, tprobs) in zip(enc_logits, enc_targets):
                        # logits (num_rows, vocab) -> move to device
                        logits = logits.to(device)
                        logp = torch.nn.functional.log_softmax(logits, dim=-1)
                        target = tprobs.to(device)
                        kld = torch.nn.functional.kl_div(logp, target, reduction="batchmean")
                        kld_sum = kld_sum + kld
                        kcount += 1
                    if kcount > 0:
                        loss_kld = kld_sum / float(kcount)
                        kld_weight = float(cfg.train.get("kld_weight", 1.0))
                        loss = loss_ce + kld_weight * loss_kld
                        # log kld
                        tb_writer.add_scalar("trainer/kld", float(loss_kld.item()), step)

            else:
                # fallback: no decoder; compute dummy loss
                loss = torch.tensor(0.0)

            # backward
            optimizer.zero_grad()
            loss.backward()
            # gradient clipping
            torch.nn.utils.clip_grad_norm_(trainable_params, grad_clip)
            optimizer.step()
            scheduler.step()

            # Save periodic checkpoint
            lr_now = optimizer.param_groups[0]["lr"]
            save_steps = int(cfg.train.get("save_steps", max(1, steps // 5)))
            if (step + 1) % save_steps == 0 or (step + 1) == steps:
                ckpt_dir = os.path.join(save_dir, f"checkpoint-step-{step + 1}")
                os.makedirs(ckpt_dir, exist_ok=True)
                if isinstance(model, RefragModel):
                    model.save_pretrained(ckpt_dir)
                else:
                    model.save_pretrained(ckpt_dir)
                # save a small training state
                try:
                    torch.save({
                        "step": step + 1,
                        "optimizer": optimizer.state_dict(),
                        "scheduler": scheduler.state_dict() if hasattr(scheduler, "state_dict") else {},
                    }, os.path.join(ckpt_dir, "training_state.pt"))
                except Exception:
                    pass


            tb_writer.add_scalar("trainer/loss", float(loss.item()), step)
            tb_writer.add_scalar("trainer/lr", lr_now, step)

        # Save model
        os.makedirs(save_dir, exist_ok=True)
        if isinstance(model, RefragModel):
            model.save_pretrained(save_dir)
        else:
            model.save_pretrained(save_dir)

        # Quick reload test
        if isinstance(model, RefragModel):
            reloaded = RefragModel.from_pretrained(save_dir)
            log.info("Reloaded RefragModel: encoder=%s decoder=%s", reloaded.config.encoder_name_or_path,
                     reloaded.config.decoder_name_or_path)

        # Close loggers
        try:
            tb_writer.flush()
            tb_writer.close()
        except Exception:
            pass

        log.info("Done")
        return 0
