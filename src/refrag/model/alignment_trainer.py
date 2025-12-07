from typing import List, Dict, Any, Optional, Tuple, Callable
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import Trainer, PreTrainedModel, TrainingArguments, TrainerCallback

from .tokenizer.tokenizer_utils import compute_dp_matrix, alignment_pairs_from_dp
from .alignment_model import AlignmentModel, AlignmentDataset, DataCollatorAlignment


# Loss adapters
class BaseLossAdapter:
    def __call__(self, **kwargs) -> torch.Tensor:
        raise NotImplementedError


class SimpleLossAdapter(BaseLossAdapter):
    def __init__(self, fn: Callable[..., torch.Tensor], required: List[str]):
        self.fn = fn
        self.required = required

    def __call__(self, **kwargs) -> torch.Tensor:
        args = [kwargs[k] for k in self.required]
        return self.fn(*args)


class CompositeLossAdapter(BaseLossAdapter):
    def __init__(self, adapters: Dict[str, Tuple[BaseLossAdapter, float]]):
        self.adapters = adapters

    def __call__(self, **kwargs) -> Tuple[torch.Tensor, Dict[str, float]]:
        totals = {}
        loss_sum = 0.0
        for name, (adapter, weight) in self.adapters.items():
            val = adapter(**kwargs)
            totals[name] = float(val.detach().cpu().item())
            loss_sum = loss_sum + weight * val
        return loss_sum, totals


# Registry of simple loss functions

def mse_loss_fn(de_vec: torch.Tensor, re_vec: torch.Tensor) -> torch.Tensor:
    return F.mse_loss(de_vec, re_vec)


def kld_loss_fn(p_de: torch.Tensor, p_re: torch.Tensor) -> torch.Tensor:
    return torch.sum(p_de * (torch.log(p_de + 1e-9) - torch.log(p_re + 1e-9)))


def argmax_loss_fn(p_de: torch.Tensor, p_re: torch.Tensor) -> torch.Tensor:
    idx = torch.argmax(p_de, dim=-1).item()
    return -torch.log(p_re[0, idx] + 1e-9)


LOSS_FN_REGISTRY = {
    'mse': (mse_loss_fn, ['de_vec', 're_vec']),
    'kld': (kld_loss_fn, ['p_de', 'p_re']),
    'argmax': (argmax_loss_fn, ['p_de', 'p_re']),
}


def build_loss_adapter_from_cfg(cfg_loss) -> CompositeLossAdapter:
    adapters = {}
    for name, sub in cfg_loss.items():
        typ = sub.get('type', 'simple')
        weight = float(sub.get('weight', 1.0))
        if typ == 'simple':
            fn_name = sub.get('fn')
            if fn_name not in LOSS_FN_REGISTRY:
                raise ValueError(f"Unknown loss fn '{fn_name}'")
            fn, required = LOSS_FN_REGISTRY[fn_name]
            adapters[name] = (SimpleLossAdapter(fn, required), weight)
        else:
            raise ValueError(f"Unsupported loss adapter type: {typ}")
    return CompositeLossAdapter(adapters)


class ConsoleLogCallback(TrainerCallback):
    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs is None:
            return
        # Print compact logs
        msg = ', '.join(f"{k}={v:.4f}" for k, v in logs.items() if isinstance(v, (int, float)))
        if msg:
            print(f"[log] step={state.global_step} {msg}")


class AlignmentTrainer(Trainer):
    def __init__(self, loss_adapter: CompositeLossAdapter, enc_tok, dec_tok, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.loss_adapter = loss_adapter
        self.enc_tok = enc_tok
        self.dec_tok = dec_tok

    def compute_loss(self, model: AlignmentModel, inputs: Dict[str, torch.Tensor], return_outputs=False):
        device = self.args.device
        enc_input_ids = inputs['enc_input_ids'].to(device)
        enc_attention_mask = inputs['enc_attention_mask'].to(device)
        dec_input_ids = inputs['dec_input_ids'].to(device)
        dec_attention_mask = inputs['dec_attention_mask'].to(device)

        # Encoder -> projector
        enc_outputs = model.encoder(input_ids=enc_input_ids, attention_mask=enc_attention_mask)
        enc_last = enc_outputs.last_hidden_state  # B x L_enc x H_enc
        proj_all = model.projector(enc_last)      # B x L_enc x H_dec

        # Decoder token embeddings
        dec_embed_layer = model.decoder.get_input_embeddings()
        dec_embeds_all = dec_embed_layer(dec_input_ids)  # B x L_dec x H_dec

        B = enc_input_ids.size(0)
        total_loss = None
        comp_totals: Dict[str, float] = {}

        for b in range(B):
            # Convert ids to lists without padding
            enc_ids = enc_input_ids[b].tolist()
            dec_ids = dec_input_ids[b].tolist()
            enc_len = int(enc_attention_mask[b].sum().item())
            dec_len = int(dec_attention_mask[b].sum().item())
            enc_ids = enc_ids[:enc_len]
            dec_ids = dec_ids[:dec_len]

            dp = compute_dp_matrix(decoder_ids=dec_ids, encoder_ids=enc_ids,
                                   dec_decode_fn=lambda ids: self.dec_tok.decode(ids),
                                   enc_decode_fn=lambda ids: self.enc_tok.decode(ids))
            pairs = alignment_pairs_from_dp(dp)
            if not pairs:
                continue

            example_losses = None
            # Per-pair losses
            for (dec_idx, enc_idx) in pairs:
                d_i = dec_idx - 1
                e_i = enc_idx - 1
                if d_i < 0 or e_i < 0:
                    continue
                if d_i >= dec_embeds_all.size(1) or e_i >= proj_all.size(1):
                    continue

                de_vec = dec_embeds_all[b:b+1, d_i:d_i+1, :]
                re_vec = proj_all[b:b+1, e_i:e_i+1, :]

                # Build inputs_embeds variants
                inputs_embeds_de = dec_embeds_all[b:b+1]
                inputs_embeds_re = torch.cat([dec_embeds_all[b:b+1, :d_i, :], re_vec, dec_embeds_all[b:b+1, d_i+1:, :]], dim=1)

                logits_de = model.decoder(inputs_embeds=inputs_embeds_de, attention_mask=dec_attention_mask[b:b+1]).logits
                logits_re = model.decoder(inputs_embeds=inputs_embeds_re, attention_mask=dec_attention_mask[b:b+1]).logits
                logit_de = logits_de[:, d_i, :]
                logit_re = logits_re[:, d_i, :]
                p_de = F.softmax(logit_de, dim=-1)
                p_re = F.softmax(logit_re, dim=-1)

                loss_val, parts = self.loss_adapter(de_vec=de_vec, re_vec=re_vec, p_de=p_de, p_re=p_re)
                if example_losses is None:
                    example_losses = loss_val
                else:
                    example_losses = example_losses + loss_val
                # accumulate parts for logging (averaged later)
                for k, v in parts.items():
                    comp_totals[k] = comp_totals.get(k, 0.0) + v

            if example_losses is None:
                continue
            # average over number of aligned pairs
            example_losses = example_losses / max(1, len(pairs))
            if total_loss is None:
                total_loss = example_losses
            else:
                total_loss = total_loss + example_losses

        if total_loss is None:
            total_loss = torch.tensor(0.0, device=device, requires_grad=True)
        else:
            total_loss = total_loss / max(1, B)

        # Log component means per step
        if comp_totals:
            scale = 1.0 / max(1, B)
            log_dict = {f"loss/{k}": v * scale for k, v in comp_totals.items()}
            self.log(log_dict)

        return (total_loss, {}) if return_outputs else total_loss
