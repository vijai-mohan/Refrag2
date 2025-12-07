from __future__ import annotations
from dataclasses import dataclass

from hydra.utils import instantiate
from omegaconf import DictConfig
import hydra
@dataclass
class LocalLauncher:
    def run(self, cfg: DictConfig):

        driver = instantiate(cfg.driver)
        return driver.run( cfg)
