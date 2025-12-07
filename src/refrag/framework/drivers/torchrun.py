from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, List
from omegaconf import DictConfig
import os, sys, subprocess, shlex
from . import transform_overrides


@dataclass
class TorchrunDriver:
    # mp.spawn / distributed parameters configured via YAML
    nproc_per_node: int = 1
    nnodes: int = 1
    node_rank: int = 0
    master_addr: str = "127.0.0.1"
    master_port: int = 29500

    # Additional distributed config (for compatibility / future use)
    backend: str = "gloo"
    rdzv_backend: str = "c10d"
    rdzv_endpoint: Optional[str] = None
    rdzv_id: Optional[str] = None
    max_restarts: int = 0
    monitor_interval: float = 5.0
    standalone: bool = True
    no_python: bool = False

    def _should_nccl(self) -> bool:
        return self.backend == "nccl" or (self.backend == "auto" and torch.cuda.is_available())

    def _build_base(self) -> List[str]:
        base = ["torchrun",
                "--nproc_per_node", str(self.nproc_per_node),
                "--nnodes", str(self.nnodes),
                "--node_rank", str(self.node_rank),
                "--master_addr", str(self.master_addr),
                "--master_port", str(self.master_port)]
        if not self.standalone and self.rdzv_endpoint:
            base += ["--rdzv_backend", self.rdzv_backend,
                     "--rdzv_endpoint", self.rdzv_endpoint]
            if self.rdzv_id:
                base += ["--rdzv_id", self.rdzv_id]
        return base

    def run(self, cfg: DictConfig):
        # If we detect we are already under torchrun (RANK present) just run app in-process
        if os.environ.get("RANK") is not None:
            from hydra.utils import instantiate
            app_obj = instantiate(cfg.app)
            return app_obj.run(cfg)

        # Build original hydra overrides from cfg.hydra overrides
        try:
            overrides = list(getattr(getattr(cfg, "hydra", None), "overrides", {}).get("task", []))
        except Exception:
            overrides = []
        # Ensure env selector transformed to local to avoid nested env effects
        overrides_transformed = transform_overrides(cfg, overrides, group_key="env", target_value="local")

        script = "run.py"  # we assume run.py is the entrypoint inside current working dir
        host_cmd = list(sys.argv) +  [*overrides_transformed]
        torch_cmd = self._build_base() + host_cmd

        print("[torchrun-driver host]", " ".join(shlex.quote(x) for x in host_cmd))
        print("[torchrun-driver exec]", " ".join(shlex.quote(x) for x in torch_cmd))

        env = os.environ.copy()
        env.setdefault("TORCH_DISTRIBUTED_DEFAULT_BACKEND", self.backend)
        try:
            rc = subprocess.call(torch_cmd, env=env)
        except Exception as e:
            raise RuntimeError(f"torchrun invocation failed: {e}")
        if rc != 0:
            raise RuntimeError(f"torchrun exited with status {rc}")
        return {"status": "SUCCEEDED", "nproc_per_node": self.nproc_per_node, "nnodes": self.nnodes}
