from pathlib import Path
from typing import Any, Dict

try:
    from omegaconf import OmegaConf
except Exception:  # OmegaConf optional
    OmegaConf = None  # type: ignore

import yaml

# Location of conf/models-old.yaml relative to this file
_CONF_PATH = Path(__file__).resolve().parent/ 'models.yaml'


def _load_yaml(path: Path) -> Dict[str, Any]:
    try:
        with open(path, 'r', encoding='utf-8') as f:
            data = yaml.safe_load(f)
            return data or {}
    except Exception:
        return {}


RAW_CFG: Dict[str, Any] = {}
if OmegaConf is not None:
    try:
        RAW_CFG = OmegaConf.to_container(OmegaConf.load(_CONF_PATH), resolve=True)  # type: ignore
    except Exception:
        RAW_CFG = _load_yaml(_CONF_PATH)
else:
    RAW_CFG = _load_yaml(_CONF_PATH)


def get_model_config(model_name: str) -> Dict[str, Any]:
    """Merge defaults and model-specific overrides from conf/models-old.yaml.
    Data-driven; no hard-coded model logic here.
    Structure expected:
    defaults: {...}
    models:
      default: {...}
      <model_name>: {...}
    """
    models = (RAW_CFG.get('models') or {}) if isinstance(RAW_CFG, dict) else {}
    defaults = (RAW_CFG.get('defaults') or {}) if isinstance(RAW_CFG, dict) else {}
    base: Dict[str, Any] = {}
    base.update(defaults or {})
    spec = {}
    if isinstance(models, dict):
        spec = models.get(model_name) or models.get('default') or {}
    base.update(spec or {})
    return base

