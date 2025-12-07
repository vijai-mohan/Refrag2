from typing import List

from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig


def transform_overrides(cfg, overrides: List[str], group_key: str, target_value: str | None = None) -> List[str]:
    """Shared override transformer:
    - For tokens starting with '<group_key>.', drop them (dependent attributes)
    - For the first token 'group_key=...', if target_value is provided, replace it with 'group_key=target_value'
      otherwise drop it
    - Leave other tokens untouched and preserve order
    Does not inject a new selector if none existed.
    """
    out: List[str] = []
    replaced = False
    prefix_attr = f"{group_key}."
    prefix_eq = f"{group_key}="
    for tok in overrides:
        if not isinstance(tok, str):
            continue
        if tok.startswith(prefix_attr):
            continue
        if tok.startswith(prefix_eq):
            if not replaced and target_value is not None:
                out.append(f"{group_key}={target_value}")
                replaced = True
            # skip original selector
            continue
        out.append(tok)

    out.append(f"app_name={cfg.app_name}")
    out.append(f"hydra.run.dir={HydraConfig.get().runtime.output_dir}")

    return out

__all__ = ["transform_overrides"]
