"""PyTorch-based alignment training script.

Usage examples:
  python main_alignment.py
  python main_alignment.py trainer.epochs=1 trainer.steps_per_epoch=5 trainer.batch_size=2

This uses Hugging Face PyTorch models and tokenizers with HF Trainer. Metrics go to TensorBoard via Trainer reporting.
"""
import random

import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel, AutoModelForCausalLM
import hydra
from omegaconf import DictConfig


def _bf16_supported(device: torch.device) -> bool:
    if device.type == "cuda":
        return torch.cuda.is_available() and getattr(torch.cuda, "is_bf16_supported", lambda: False)()
    cpu_backend = getattr(torch.backends, "cpu", None)
    if device.type == "cpu" and cpu_backend:
        return getattr(cpu_backend, "is_bf16_supported", lambda: False)()
    return False


class Projector(nn.Module):
    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.dense = nn.Linear(in_dim, out_dim)

    def forward(self, x):
        return self.dense(x)


def load_tokenizers(cfg):
    enc_tok = AutoTokenizer.from_pretrained(cfg.model.encoder, use_fast=True)
    dec_tok = AutoTokenizer.from_pretrained(cfg.model.decoder, use_fast=True)
    return enc_tok, dec_tok


def load_models(cfg, device):
    use_bf16 = _bf16_supported(device)
    torch_dtype = torch.bfloat16 if use_bf16 else None
    enc = AutoModel.from_pretrained(cfg.model.encoder, torch_dtype=torch_dtype).to(device)
    dec = AutoModelForCausalLM.from_pretrained(cfg.model.decoder, torch_dtype=torch_dtype).to(device)
    return enc, dec


def example_wikipedia_texts(num_examples=10):
    samples = [
        "Albert Einstein was a theoretical physicist who developed the theory of relativity.",
        "Python is a programming language that emphasizes code readability.",
        "The Eiffel Tower is located in Paris and is a global cultural icon of France.",
        "Machine learning enables computers to learn from data without being explicitly programmed.",
        "The moon orbits the Earth and influences the ocean tides.",
    ]
    out = [random.choice(samples) for _ in range(num_examples)]
    return out


@hydra.main(config_path="conf", config_name="config")
def main(cfg: DictConfig):
    from transformers import TrainingArguments
    from refrag.model.alignment_trainer import (
        AlignmentModel,
        AlignmentDataset,
        DataCollatorAlignment,
        AlignmentTrainer,
        build_loss_adapter_from_cfg,
        ConsoleLogCallback,
    )

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    use_bf16 = _bf16_supported(device)
    enc_tok, dec_tok = load_tokenizers(cfg)

    # Models
    enc_model, dec_model = load_models(cfg, device)
    enc_dim = enc_model.config.hidden_size
    dec_dim = dec_model.config.hidden_size
    projector = Projector(enc_dim, dec_dim).to(device)
    model = AlignmentModel(enc_model, dec_model, projector)

    # Data
    texts = None
    if cfg.data.get('use_builtin', True):
        texts = example_wikipedia_texts(max(64, cfg.train.batch_size * max(1, cfg.train.get('steps_per_epoch', 1))))
    else:
        # fallback simple loader from text file (one example per line)
        data_path = cfg.data.get('path')
        if not data_path:
            raise ValueError("cfg.data.path must be provided when data.use_builtin is False")
        with open(data_path, 'r', encoding='utf-8') as f:
            texts = [line.strip() for line in f if line.strip()]

    train_dataset = AlignmentDataset(texts, enc_tok, dec_tok, max_len=cfg.data.get('max_len'))
    collator = DataCollatorAlignment(enc_pad_id=enc_tok.pad_token_id or 0, dec_pad_id=dec_tok.pad_token_id or 0)

    # Loss adapters from Hydra config
    loss_adapter = build_loss_adapter_from_cfg(cfg.loss)

    # HF Trainer args
    # Map steps_per_epoch to max_steps if provided; if >0, it overrides epochs
    eff_max_steps = cfg.trainer.get('max_steps', -1)
    if eff_max_steps <= 0:
        spe = int(cfg.train.get('steps_per_epoch', 0) or 0)
        if spe > 0:
            eff_max_steps = spe * int(cfg.train.epochs)

    training_args = TrainingArguments(
        output_dir=cfg.trainer.output_dir,
        per_device_train_batch_size=cfg.train.batch_size,
        num_train_epochs=cfg.train.epochs,
        learning_rate=cfg.train.lr,
        logging_steps=max(1, cfg.trainer.logging_steps),
        save_steps=cfg.trainer.save_steps,
        report_to=["tensorboard"],
        remove_unused_columns=False,
        gradient_accumulation_steps=cfg.trainer.get('grad_accum_steps', 1),
        max_steps=eff_max_steps,
        max_grad_norm=cfg.train.get('grad_clip_norm', 1.0),
        bf16=use_bf16,
    )

    trainer = AlignmentTrainer(
        loss_adapter=loss_adapter,
        enc_tok=enc_tok,
        dec_tok=dec_tok,
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=collator,
        callbacks=[ConsoleLogCallback()],
    )

    trainer.train()
    print("Training finished")


if __name__ == '__main__':
    main()
