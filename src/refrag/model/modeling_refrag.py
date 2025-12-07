"""Minimal Refrag model wrapper.

This composes an encoder model, a projector (linear), and a decoder model. It expects segments prepared by the tokenizer.

For encoder segments: the encoder processes each packed row (batch of size compression) and we take the CLS (or first token) embedding and then project it to decoder embedding dim.
For decoder segments: we use decoder tokenizer embeddings (via decoder.get_input_embeddings()).

This is a simplified API for demonstration and testing.
"""
from typing import List, Dict, Any, Tuple
import os
import torch
import torch.nn as nn
from transformers import AutoModel, AutoConfig, PreTrainedModel, AutoModelForCausalLM
from .config_refrag import RefragConfig


class RefragModel(PreTrainedModel):
    config_class = RefragConfig

    def __init__(self, config: "RefragConfig"):
        # NOTE: PreTrainedModel expects a config object
        super().__init__(config)
        self.config = config
        # load encoder and decoder base models
        self.encoder = AutoModel.from_pretrained(config.encoder_name_or_path)
        # Use causal LM for decoder so we can compute logits
        self.decoder = AutoModelForCausalLM.from_pretrained(config.decoder_name_or_path)
        # projector: map encoder hidden dim -> decoder hidden dim
        enc_dim = getattr(self.encoder.config, "hidden_size", 768)
        dec_dim = getattr(self.decoder.config, "hidden_size", 768)
        self.projector = nn.Linear(enc_dim, dec_dim)
        # LM head projection (map projected encoder embedding to decoder vocab logits)
        vocab_size = getattr(self.decoder.config, "vocab_size", None) or getattr(self.decoder.config, "n_positions", None)
        self.lm_head_proj = nn.Linear(dec_dim, getattr(self.decoder.config, "vocab_size", 50257))

        # Set trainable flags according to config.training_model (p/e/d)
        self._set_trainable_flags()

    @classmethod
    def from_config(cls, cfg):
        # ensure config_class is set
        cls.config_class = type(cfg)
        model = cls(cfg)
        return model

    def _set_trainable_flags(self):
        # projector
        for p in self.projector.parameters():
            p.requires_grad = self.config.train_projector
        # encoder
        for p in self.encoder.parameters():
            p.requires_grad = self.config.train_encoder
        # decoder
        for p in self.decoder.parameters():
            p.requires_grad = self.config.train_decoder

    def forward(self, segments: List[Dict[str, Any]]) -> Tuple[torch.Tensor, torch.Tensor]:
        """Process the segments and return combined embeddings and labels."""
        all_embeddings = []
        all_labels = []
        dec_embed = self.decoder.get_input_embeddings()
        device = self.projector.weight.device

        for seg in segments:
            if seg["type"] == "decoder":
                token_ids = seg.get("tokens", [])
                if token_ids:
                    ids = torch.tensor(token_ids, dtype=torch.long, device=device)
                    emb = dec_embed(ids)
                    all_embeddings.append(emb)
                    all_labels.append(ids)
            else: # encoder
                packed = seg.get("packed_tokens", [])
                masks = seg.get("attention_mask", [])
                row_embs = []
                for row_ids, row_mask in zip(packed, masks):
                    ids = torch.tensor(row_ids, dtype=torch.long, device=device).unsqueeze(0)
                    attn = torch.tensor(row_mask, dtype=torch.long, device=device).unsqueeze(0)
                    enc_out = self.encoder(input_ids=ids, attention_mask=attn)
                    
                    if hasattr(enc_out, "pooler_output") and enc_out.pooler_output is not None:
                        cls_emb = enc_out.pooler_output.squeeze(0)
                    else:
                        cls_emb = enc_out.last_hidden_state[:, 0, :].squeeze(0)
                    
                    proj = self.projector(cls_emb)
                    row_embs.append(proj)

                if row_embs:
                    stacked = torch.stack(row_embs, dim=0)
                    all_embeddings.append(stacked)
                    dummy_labels = torch.full((stacked.shape[0],), -100, dtype=torch.long, device=device)
                    all_labels.append(dummy_labels)

        final_embeddings = torch.cat(all_embeddings, dim=0)
        final_labels = torch.cat(all_labels, dim=0)
        
        return final_embeddings, final_labels

    def encoder_projected_logits(self, segments: List[Dict[str, Any]]):
        """Compute projected encoder row logits over decoder vocab for each encoder row.

        Returns a list of (segment_index, row_logits_tensor) where row_logits_tensor is (num_rows, vocab_size).
        """
        results = []
        for i, seg in enumerate(segments):
            if seg["type"] == "encoder":
                packed = seg.get("packed_tokens", [])
                masks = seg.get("attention_mask", [])
                row_logits = []
                for row_ids, row_mask in zip(packed, masks):
                    ids = torch.tensor(row_ids, dtype=torch.long).unsqueeze(0)
                    attn = torch.tensor(row_mask, dtype=torch.long).unsqueeze(0)
                    enc_out = self.encoder(input_ids=ids, attention_mask=attn)
                    if hasattr(enc_out, "pooler_output") and enc_out.pooler_output is not None:
                        cls_emb = enc_out.pooler_output.squeeze(0)
                    else:
                        cls_emb = enc_out.last_hidden_state[:, 0, :].squeeze(0)
                    proj = self.projector(cls_emb)  # (dec_dim,)
                    logits = self.lm_head_proj(proj)  # (vocab_size,)
                    row_logits.append(logits.unsqueeze(0))
                if row_logits:
                    row_logits = torch.cat(row_logits, dim=0)
                else:
                    row_logits = torch.empty(0)
                results.append((i, row_logits))
        return results

    # override save_pretrained to include HF standard
    def save_pretrained(self, save_directory: str):
        # save config
        self.config.save_pretrained(save_directory)
        # save encoder and decoder as subfolders using HF APIs
        enc_dir = os.path.join(save_directory, "encoder")
        dec_dir = os.path.join(save_directory, "decoder")
        proj_file = os.path.join(save_directory, "projector.pt")
        os.makedirs(enc_dir, exist_ok=True)
        os.makedirs(dec_dir, exist_ok=True)
        # save underlying models using their save_pretrained if available
        try:
            self.encoder.save_pretrained(enc_dir)
        except Exception:
            # fall back to state_dict
            torch.save(self.encoder.state_dict(), os.path.join(enc_dir, "pytorch_model.bin"))
        try:
            self.decoder.save_pretrained(dec_dir)
        except Exception:
            torch.save(self.decoder.state_dict(), os.path.join(dec_dir, "pytorch_model.bin"))
        # save projector weights
        torch.save(self.projector.state_dict(), proj_file)

    @classmethod
    def from_pretrained(cls, load_directory: str, **kwargs):
        # load config
        cfg = cls.config_class.from_pretrained(load_directory)
        model = cls(cfg)
        # load encoder/decoder
        enc_dir = os.path.join(load_directory, "encoder")
        dec_dir = os.path.join(load_directory, "decoder")
        proj_file = os.path.join(load_directory, "projector.pt")
        try:
            model.encoder = AutoModel.from_pretrained(enc_dir)
        except Exception:
            # try state_dict
            state = torch.load(os.path.join(enc_dir, "pytorch_model.bin"), map_location="cpu")
            model.encoder.load_state_dict(state)
        try:
            # decoder is a causal LM
            model.decoder = AutoModelForCausalLM.from_pretrained(dec_dir)
        except Exception:
            state = torch.load(os.path.join(dec_dir, "pytorch_model.bin"), map_location="cpu")
            model.decoder.load_state_dict(state)
        # load projector
        if os.path.exists(proj_file):
            state = torch.load(proj_file, map_location="cpu")
            model.projector.load_state_dict(state)
        return model
