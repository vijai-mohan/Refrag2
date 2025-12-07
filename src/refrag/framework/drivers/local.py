from __future__ import annotations
from dataclasses import dataclass
from omegaconf import DictConfig
from . import transform_overrides
import hydra
@dataclass
class LocalDriver:
    group_key: str = "env"
    target_value: str = "local"

    def run(self, cfg: DictConfig):
        # Use the exact system Hydra overrides and transform only the selected group
        try:
            original_overrides = list(getattr(getattr(cfg, "hydra", None), "overrides", {}).get("task", []))
        except Exception:
            original_overrides = []
        _ = transform_overrides(cfg,original_overrides, group_key=self.group_key, target_value=self.target_value)

        # Run app in-process
        app_obj= hydra.utils.instantiate(cfg.app)
        result = app_obj.run(cfg)
        return {"status": "SUCCEEDED", "inprocess": True, "result": result}
