from __future__ import annotations
import os, subprocess, shlex, sys
from dataclasses import dataclass, field
from typing import List
from omegaconf import DictConfig
from refrag.framework.drivers import transform_overrides
from .launcher_utils import build_host_cmd, make_paths_relative


@dataclass
class DockerLauncher:
    image: str = "refrag:latest"
    mounts: List[str] = field(default_factory=list)
    workdir: str = "/workspace"
    env_passthrough: List[str] = field(default_factory=list)
    detach: bool = False
    group_key: str = "env"
    target_value: str = "local"
    gpus_flag: str = "all"  # value passed to docker --gpus when a GPU device is requested

    def run(self, cfg: DictConfig):
        host_cmd = build_host_cmd(sys.argv, python_bin="python3")
        base_cmd = transform_overrides(
            cfg, host_cmd, group_key=self.group_key, target_value=self.target_value
        )
        base_cmd = make_paths_relative(base_cmd)

        docker_cmd: List[str] = ["docker", "run", "--rm"]
        if self.detach:
            docker_cmd.append("-d")

        desired_device = None
        try:
            desired_device = cfg.device
        except Exception:
            desired_device = cfg.get("device") if hasattr(cfg, "get") else None
        if isinstance(desired_device, str) and desired_device.lower().startswith(("cuda", "gpu")):
            docker_cmd.extend(["--gpus", self.gpus_flag])

        resolved_mounts = list(self.mounts)
        # Reuse host Hugging Face cache when available to avoid re-downloading models
        hf_cache = os.path.join(os.path.expanduser("~"), ".cache", "huggingface")
        if os.path.isdir(hf_cache):
            hf_spec = f"{hf_cache}:/root/.cache/huggingface"
            hosts = {spec.split(":", 1)[0] for spec in resolved_mounts}
            if hf_cache not in hosts:
                resolved_mounts.append(hf_spec)

        for mount in resolved_mounts:
            docker_cmd.extend(["-v", mount])

        if self.workdir:
            docker_cmd.extend(["-w", self.workdir])

        for key in self.env_passthrough:
            if key in os.environ:
                docker_cmd.extend(["-e", f"{key}={os.environ[key]}"])

        docker_cmd.append(self.image)
        docker_cmd.extend(base_cmd)

        print("[docker-env host]", " ".join(shlex.quote(tok) for tok in host_cmd))
        print("[docker-env docker]", " ".join(shlex.quote(tok) for tok in docker_cmd))

        try:
            if self.detach:
                container_id = subprocess.check_output(docker_cmd, text=True).strip()
                return {
                    "status": "STARTED",
                    "container_id": container_id,
                    "image": self.image,
                }
            rc = subprocess.call(docker_cmd)
        except Exception as e:
            raise RuntimeError(f"docker run failed: {e}")

        if rc != 0:
            raise RuntimeError(f"docker exited with status {rc}")

        return {"status": "SUCCEEDED", "image": self.image, "detached": self.detach}

