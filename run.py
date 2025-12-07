from __future__ import annotations
import sys, os, socket, logging, time
from pathlib import Path
from dataclasses import dataclass
from typing import Any, Dict
import hydra
from omegaconf import DictConfig, OmegaConf
from huggingface_hub import login
from hydra.core.hydra_config import HydraConfig
from faker import Faker

REPO_ROOT = Path(__file__).resolve().parent
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))
@dataclass
class AppResult:
    status: str
    info: Dict[str, Any]


def run_name(seed=23):
    if 'RUN_NAME' in os.environ:
        return os.environ['RUN_NAME']

    #Let's use Faker to generate random names that are memorable
    #
    fake = Faker()
    x = fake.first_name().lower()
    y = fake.last_name().lower()
    result= f"{x}_{y}"
    os.environ['RUN_NAME']=result
    return result


def get_group_choice(group):
    return HydraConfig.get().runtime.choices.get(group)

OmegaConf.register_new_resolver("group_choice", get_group_choice)
OmegaConf.register_new_resolver("run_name", run_name)

# ---------------- Logging -----------------

def setup_rank_logging() -> str:
    run_dir = HydraConfig.get().run.dir
    rank = os.environ.get("RANK", "0")
    host = socket.gethostname()
    log_dir = os.path.join(run_dir, "logs")
    os.makedirs(log_dir, exist_ok=True)
    logfile = os.path.join(log_dir, f"run-{rank}.log")
    root = logging.getLogger()
    # Reconfigure root completely every process (simpler + deterministic)
    for h in list(root.handlers):
        root.removeHandler(h)
    fmt = logging.Formatter("%(asctime)s - %(levelname)s - %(name)s - %(message)s")
    fh = logging.FileHandler(logfile, encoding="utf-8")
    fh.setFormatter(fmt)
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)
    root.addHandler(fh)
    root.addHandler(sh)
    root.setLevel(logging.INFO)
    logging.getLogger(__name__).info(f"Per-rank logging initialized: {logfile}")
    logging.getLogger(__name__).info("Hydra run dir: %s", run_dir)
    logging.getLogger(__name__).info("Global rank=%s", rank)
    return logfile

# ---------------- Main -----------------

@hydra.main(config_path="conf", config_name="config", version_base="1.3")
def main(cfg: DictConfig) -> None:
    # Fail-fast philosophy: assume config correctness.
    setup_rank_logging()
    print("Run Name:", cfg.run_name)
    print("CFG:\n", OmegaConf.to_yaml(cfg, resolve=True))

    hf_token = str(cfg.get("HF_TOKEN") or "").strip()
    if not hf_token or hf_token.startswith("hf_fake"):
        raise RuntimeError("HF_TOKEN is not set; export HF_TOKEN in your environment (never hardcode secrets).")
    login(hf_token)

    env = hydra.utils.instantiate(cfg.env)
    result = env.run(cfg=cfg)

    print("RESULT:", result)
    logging.getLogger(__name__).info("Result: %s", result)

if __name__ == "__main__":
    main()
