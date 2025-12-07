"""PyTorch-based end-to-end training script for the Refrag model.

This script trains the RefragModel without a separate alignment pre-training step.

Usage examples:
  python main_e2e.py
  python main_e2e.py train.epochs=1 train.steps_per_epoch=5 train.batch_size=2
"""
import random
from typing import List, Dict, Any

import torch
import torch.nn as nn
from transformers import AutoTokenizer, Trainer, TrainingArguments
import hydra
from omegaconf import DictConfig

from refrag.model.modeling_refrag import RefragModel, RefragConfig
from refrag.model.tokenization_refrag import RefragTokenizer


class E2EDataset(torch.utils.data.Dataset):
    def __init__(self, texts: List[str], tokenizer: RefragTokenizer, max_len: int):
        self.texts = texts
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        return self.tokenizer.parse_and_tokenize(text)


class DataCollatorE2E:
    def __init__(self, tokenizer: RefragTokenizer):
        self.tokenizer = tokenizer
        self.decoder_pad_token_id = self.tokenizer.decoder_tokenizer.pad_token_id

    def __call__(self, features: List[List[Dict[str, Any]]]) -> Dict[str, Any]:
        return {"segments_batch": features}


class RefragTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        segments_batch = inputs["segments_batch"]
        total_loss = 0
        batch_size = len(segments_batch)
        all_logits = []

        for segments in segments_batch:
            embeddings, labels = model(segments)
            
            lm_head = model.decoder.get_output_embeddings()
            if lm_head is None:
                 lm_head = model.decoder.lm_head

            logits = lm_head(embeddings.unsqueeze(0))
            
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            
            loss_fct = nn.CrossEntropyLoss(ignore_index=-100)
            loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
            
            total_loss += loss
            if return_outputs:
                all_logits.append(logits)

        final_loss = total_loss / batch_size if batch_size > 0 else torch.tensor(0.0)
        
        if return_outputs:
            # For simplicity, only return the logits from the last item in the batch
            return (final_loss, {"logits": all_logits[-1]}) if all_logits else (final_loss, {})
        return final_loss


def example_e2e_texts(num_examples=10):
    docs = [
        "The Eiffel Tower is a wrought-iron lattice tower on the Champ de Mars in Paris, France.",
        "The Great Wall of China is a series of fortifications made of stone, brick, tamped earth, wood, and other materials.",
        "The Colosseum is an oval amphitheatre in the centre of the city of Rome, Italy.",
    ]
    questions = [
        "Where is the Eiffel Tower?",
        "What is the Great Wall of China made of?",
        "What shape is the Colosseum?",
    ]
    
    out = []
    for _ in range(num_examples):
        doc = random.choice(docs)
        question = random.choice(questions)
        out.append(f"Question: {question} <doc id='1'>{doc}</doc> Answer:")
    return out


@hydra.main(config_path="conf", config_name="config")
def main(cfg: DictConfig):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Tokenizers
    enc_tok = AutoTokenizer.from_pretrained(cfg.model.encoder, use_fast=True)
    dec_tok = AutoTokenizer.from_pretrained(cfg.model.decoder, use_fast=True)
    
    if dec_tok.pad_token is None:
        dec_tok.pad_token = dec_tok.eos_token

    refrag_tokenizer = RefragTokenizer(
        encoder_tokenizer=enc_tok,
        decoder_tokenizer=dec_tok,
        compression=cfg.model.get('compression', 4),
        encoder_pad_token_id=enc_tok.pad_token_id or 0,
    )

    # Model
    refrag_config = RefragConfig(
        encoder_name_or_path=cfg.model.encoder,
        decoder_name_or_path=cfg.model.decoder,
        compression=cfg.model.get('compression', 4),
    )
    model = RefragModel(refrag_config).to(device)

    # Data
    texts = example_e2e_texts(max(64, cfg.train.batch_size * max(1, cfg.train.get('steps_per_epoch', 1))))
    train_dataset = E2EDataset(texts, refrag_tokenizer, max_len=cfg.data.get('max_len', 512))
    collator = DataCollatorE2E(tokenizer=refrag_tokenizer)

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
        max_grad_norm=cfg.train.get('grad_clip_norm', 1.0),
    )

    trainer = RefragTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=collator,
    )

    trainer.train()
    print("E2E Training finished")


if __name__ == '__main__':
    main()
