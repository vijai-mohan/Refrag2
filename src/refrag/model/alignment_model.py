from typing import List, Dict, Any, Optional, Tuple, Callable
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import Trainer, PreTrainedModel, TrainingArguments, TrainerCallback

class AlignmentModel(nn.Module):
    """Container model holding encoder, decoder, and projector.
    The Trainer will optimize all submodule parameters.
    """
    def __init__(self, encoder: PreTrainedModel, decoder: PreTrainedModel, projector: nn.Module):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.projector = projector

    def forward(self, **kwargs):
        # Not used: loss is computed in Trainer.compute_loss override
        return {}


class AlignmentDataset(torch.utils.data.Dataset):
    def __init__(self, texts: List[str], enc_tok, dec_tok, max_len: Optional[int] = None):
        self.texts = texts
        self.enc_tok = enc_tok
        self.dec_tok = dec_tok
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        enc = self.enc_tok(text, return_tensors=None, padding=False, truncation=True, max_length=self.max_len)
        dec = self.dec_tok(text, return_tensors=None, padding=False, truncation=True, max_length=self.max_len)
        return {
            'enc_input_ids': torch.tensor(enc['input_ids'], dtype=torch.long),
            'enc_attention_mask': torch.tensor(enc.get('attention_mask', [1] * len(enc['input_ids'])), dtype=torch.long),
            'dec_input_ids': torch.tensor(dec['input_ids'], dtype=torch.long),
            'dec_attention_mask': torch.tensor(dec.get('attention_mask', [1] * len(dec['input_ids'])), dtype=torch.long),
        }


class DataCollatorAlignment:
    def __init__(self, enc_pad_id: int, dec_pad_id: int):
        self.enc_pad_id = enc_pad_id
        self.dec_pad_id = dec_pad_id

    def __call__(self, features: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        # Pad enc side
        enc_max = max(f['enc_input_ids'].size(0) for f in features)
        dec_max = max(f['dec_input_ids'].size(0) for f in features)

        enc_input_ids = []
        enc_attention_mask = []
        dec_input_ids = []
        dec_attention_mask = []

        for f in features:
            ei = f['enc_input_ids']
            ea = f['enc_attention_mask']
            di = f['dec_input_ids']
            da = f['dec_attention_mask']
            # pad enc
            enc_pad_len = enc_max - ei.size(0)
            if enc_pad_len > 0:
                ei = torch.cat([ei, torch.full((enc_pad_len,), self.enc_pad_id, dtype=torch.long)], dim=0)
                ea = torch.cat([ea, torch.zeros((enc_pad_len,), dtype=torch.long)], dim=0)
            # pad dec
            dec_pad_len = dec_max - di.size(0)
            if dec_pad_len > 0:
                di = torch.cat([di, torch.full((dec_pad_len,), self.dec_pad_id, dtype=torch.long)], dim=0)
                da = torch.cat([da, torch.zeros((dec_pad_len,), dtype=torch.long)], dim=0)
            enc_input_ids.append(ei)
            enc_attention_mask.append(ea)
            dec_input_ids.append(di)
            dec_attention_mask.append(da)

        batch = {
            'enc_input_ids': torch.stack(enc_input_ids, dim=0),
            'enc_attention_mask': torch.stack(enc_attention_mask, dim=0),
            'dec_input_ids': torch.stack(dec_input_ids, dim=0),
            'dec_attention_mask': torch.stack(dec_attention_mask, dim=0),
        }
        return batch
