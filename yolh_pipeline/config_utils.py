from pathlib import Path
import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "configs" / "pipeline.yaml"


def load_pipeline_config(config_path: str = None) -> dict:
    path = Path(config_path) if config_path else DEFAULT_CONFIG_PATH
    if not path.is_absolute():
        path = (PROJECT_ROOT / path).resolve()
    if not path.exists():
        raise FileNotFoundError(f"Pipeline config not found: {path}")

    with path.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}
    if not isinstance(cfg, dict):
        raise ValueError(f"Invalid pipeline config format: {path}")
    return cfg


def get_step_cfg(cfg: dict, step_key: str) -> dict:
    step_cfg = cfg.get(step_key, {})
    if step_cfg is None:
        return {}
    if not isinstance(step_cfg, dict):
        raise ValueError(f"Config section '{step_key}' must be a mapping")
    return step_cfg
