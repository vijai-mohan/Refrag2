from __future__ import annotations
from dataclasses import dataclass
from omegaconf import DictConfig
from hydra.utils import instantiate
import hydra
@dataclass
class InlineDriver:
    def run(self, cfg: DictConfig):
        app=hydra.utils.instantiate(cfg.app)
        return app.run(cfg)

