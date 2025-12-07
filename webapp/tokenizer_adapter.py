from typing import Any, List
from dataclasses import dataclass

@dataclass
class TokenizerAdapter:
    tokenizer: Any

    def encode(self, text: str, add_special_tokens: bool = False) -> List[int]:
        attempts = [
            lambda: self.tokenizer.encode(text, add_special_tokens=add_special_tokens),
            lambda: self.tokenizer(text, return_tensors='pt')["input_ids"].tolist()[0],
            lambda: [self.tokenizer.convert_tokens_to_ids(t) for t in text.split()],
        ]
        for fn in attempts:
            try:
                res = fn()
                return list(res) if not isinstance(res, list) else res
            except Exception:
                continue
        return []

    def decode(self, ids: List[int], skip_special_tokens: bool = True, clean_up_tokenization_spaces: bool = False) -> str:
        attempts = [
            lambda: self.tokenizer.decode(ids, skip_special_tokens=skip_special_tokens, clean_up_tokenization_spaces=clean_up_tokenization_spaces),
            lambda: " ".join(str(i) for i in ids),
        ]
        for fn in attempts:
            try:
                return str(fn())
            except Exception:
                continue
        return ""

    def apply_chat_template(self, chat, add_generation_prompt: bool = True):
        attempts = [
            lambda: getattr(self.tokenizer, 'apply_chat_template')(chat, tokenize=False, add_generation_prompt=add_generation_prompt),
            lambda: getattr(self.tokenizer, 'apply_chat_template')(chat, add_generation_prompt=add_generation_prompt),
            lambda: getattr(self.tokenizer, 'apply_chat_template')(tuple((c.get('role'), c.get('content')) if isinstance(c, dict) else c for c in chat), tokenize=False, add_generation_prompt=add_generation_prompt),
        ]
        for fn in attempts:
            try:
                return str(fn())
            except Exception:
                continue
        raise RuntimeError('apply_chat_template unsupported')

