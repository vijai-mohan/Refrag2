"""Refrag tokenizer that composes an encoder tokenizer and a decoder tokenizer.

Tokenization rules:
- Input contains document tags: <doc id="1">...content...</doc> and free text outside tags.
- For decoder-tokenized segments (outside <doc>), use decoder tokenizer normally.
- For encoder-tokenized segments (inside <doc>), use encoder tokenizer; then pack tokens into rows of `compression` width and produce a 2D matrix (num_rows x compression) with padding and mask. Each row produces a CLS embedding via encoder+projector later.

This tokenizer will expose a `parse_and_tokenize` method that returns a list of segment dicts in original order. Each segment dict has:
  - type: 'decoder' or 'encoder'
  - raw_text: original text
  - tokens: token ids (list) produced by the corresponding tokenizer
  - token_type: same as type
  - For encoder segments: 'packed_tokens' -> List[List[int]] (rows) and 'attention_mask' -> List[List[int]] matching packed_tokens.

This module intentionally keeps implementation minimal and depends on Hugging Face tokenizers being passed in.
"""
from typing import List, Dict, Any, Tuple
import re

DOC_RE = re.compile(r"<doc\s+id=['\"](?P<id>[^'\"]+)['\"]>(?P<body>.*?)</doc>", re.DOTALL)


class RefragTokenizer:
    def __init__(self, encoder_tokenizer, decoder_tokenizer, compression: int = 4, encoder_pad_token_id: int = 0):
        """encoder_tokenizer and decoder_tokenizer are huggingface tokenizer instances.

        compression: how many encoder tokens to pack per row.
        encoder_pad_token_id: pad id to use for encoder packed rows.
        """
        self.encoder_tokenizer = encoder_tokenizer
        self.decoder_tokenizer = decoder_tokenizer
        self.compression = int(compression)
        self.encoder_pad_token_id = encoder_pad_token_id

    def parse_segments(self, text: str) -> List[Dict[str, Any]]:
        """Parse the input text into ordered segments: decoder-text and encoder-docs.

        Returns list of dicts: {'type': 'decoder'|'encoder', 'raw_text': str, 'meta': {..}}
        """
        segments: List[Dict[str, Any]] = []
        last = 0
        for m in DOC_RE.finditer(text):
            # text before this doc is decoder-type
            if m.start() > last:
                pre = text[last:m.start()]
                if pre.strip():
                    segments.append({"type": "decoder", "raw_text": pre})
            # doc body is encoder-type
            doc_id = m.group("id")
            body = m.group("body")
            segments.append({"type": "encoder", "raw_text": body, "meta": {"id": doc_id}})
            last = m.end()
        # any trailing text
        if last < len(text):
            tail = text[last:]
            if tail.strip():
                segments.append({"type": "decoder", "raw_text": tail})
        return segments

    def _pack_encoder_tokens(self, token_ids: List[int]) -> Tuple[List[List[int]], List[List[int]]]:
        """Pack encoder token ids into rows of width `compression`. Return (rows, masks).

        Mask is 1 for real tokens and 0 for pads. Keep relative order.
        """
        if self.compression <= 0:
            raise ValueError("compression must be > 0")
        rows: List[List[int]] = []
        masks: List[List[int]] = []
        for i in range(0, len(token_ids), self.compression):
            chunk = token_ids[i : i + self.compression]
            mask = [1] * len(chunk)
            if len(chunk) < self.compression:
                # pad
                pad_len = self.compression - len(chunk)
                chunk = chunk + [self.encoder_pad_token_id] * pad_len
                mask = mask + [0] * pad_len
            rows.append(chunk)
            masks.append(mask)
        if not rows:
            # produce one empty padded row
            rows = [[self.encoder_pad_token_id] * self.compression]
            masks = [[0] * self.compression]
        return rows, masks

    def parse_and_tokenize(self, text: str) -> List[Dict[str, Any]]:
        """Parse, tokenize and pack where necessary. Maintains original order.

        Returns list of segments with tokenization results.
        """
        segments = self.parse_segments(text)
        out: List[Dict[str, Any]] = []
        for seg in segments:
            t = seg["type"]
            raw = seg["raw_text"]
            if t == "decoder":
                enc = self.decoder_tokenizer(raw, add_special_tokens=False)
                tokens = enc.get("input_ids") or enc.get("ids") or []
                out.append({"type": "decoder", "raw_text": raw, "tokens": tokens})
            else:
                enc = self.encoder_tokenizer(raw, add_special_tokens=False)
                tokens = enc.get("input_ids") or enc.get("ids") or []
                rows, masks = self._pack_encoder_tokens(tokens)
                out.append(
                    {
                        "type": "encoder",
                        "raw_text": raw,
                        "tokens": tokens,
                        "packed_tokens": rows,
                        "attention_mask": masks,
                        "meta": seg.get("meta", {}),
                    }
                )
        return out

    def build_encoder_targets(self, segments: List[Dict[str, Any]], decoder_tokenizer, vocab_size: int, smoothing: float = 1e-6):
        """Build target probability distributions over decoder vocab for each encoder packed row.

        Strategy (heuristic):
        - For each encoder segment, tokenize the segment.raw_text with the decoder tokenizer.
        - Split the resulting decoder token id sequence into N parts where N == number of packed rows for the encoder segment.
          Tokens are distributed in order, roughly equally (first rows may get one extra token when not divisible).
        - For each part, build a probability vector over the decoder vocab by counting token frequencies and normalizing.
        - Apply small smoothing to avoid zeros.

        Returns: list of tuples (segment_index, tensor(num_rows, vocab_size)) where tensors are torch.FloatTensor on CPU.
        """
        import torch

        results = []
        for idx, seg in enumerate(segments):
            if seg["type"] != "encoder":
                continue
            raw = seg.get("raw_text", "")
            # decoder-side tokenization of the encoder raw text
            dec_tok = decoder_tokenizer(raw, add_special_tokens=False)
            dec_ids = dec_tok.get("input_ids") or dec_tok.get("ids") or []

            packed = seg.get("packed_tokens", [])
            num_rows = len(packed)
            if num_rows <= 0:
                continue

            # Split dec_ids into num_rows roughly equally
            parts = [[] for _ in range(num_rows)]
            for i, tid in enumerate(dec_ids):
                parts[i % num_rows].append(tid)

            # Build probability vectors
            row_probs = []
            for part in parts:
                vec = torch.full((vocab_size,), smoothing, dtype=torch.float32)
                if len(part) > 0:
                    for t in part:
                        if 0 <= t < vocab_size:
                            vec[t] += 1.0
                vec = vec / vec.sum()
                row_probs.append(vec.unsqueeze(0))
            if row_probs:
                row_probs = torch.cat(row_probs, dim=0)
            else:
                row_probs = torch.empty((0, vocab_size), dtype=torch.float32)
            results.append((idx, row_probs))
        return results


# Small convenience factory to create a RefragTokenizer from pretrained names
from transformers import AutoTokenizer


def from_pretrained(encoder_name: str, decoder_name: str, compression: int = 4) -> RefragTokenizer:
    enc = AutoTokenizer.from_pretrained(encoder_name)
    dec = AutoTokenizer.from_pretrained(decoder_name)
    return RefragTokenizer(enc, dec, compression=compression, encoder_pad_token_id=enc.pad_token_id or 0)
