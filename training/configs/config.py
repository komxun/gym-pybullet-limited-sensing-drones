"""Centralized configuration loader.

Loads a YAML config file and provides dot-access via a simple namespace.
CLI overrides are merged on top of the YAML values.
"""

import os
import yaml
import argparse
from types import SimpleNamespace


def _dict_to_namespace(d: dict) -> SimpleNamespace:
    """Recursively convert a nested dict to SimpleNamespace for dot-access."""
    ns = SimpleNamespace()
    for k, v in d.items():
        if isinstance(v, dict):
            setattr(ns, k, _dict_to_namespace(v))
        else:
            setattr(ns, k, v)
    return ns


def _namespace_to_dict(ns: SimpleNamespace) -> dict:
    """Convert SimpleNamespace back to dict."""
    d = {}
    for k, v in vars(ns).items():
        if isinstance(v, SimpleNamespace):
            d[k] = _namespace_to_dict(v)
        else:
            d[k] = v
    return d


class Config(SimpleNamespace):
    """Thin wrapper around SimpleNamespace with helper methods."""

    def to_dict(self) -> dict:
        return _namespace_to_dict(self)

    def __repr__(self):
        return yaml.dump(self.to_dict(), default_flow_style=False)


def _deep_merge(base: dict, override: dict) -> dict:
    """Merge *override* into *base* recursively (override wins)."""
    merged = base.copy()
    for k, v in override.items():
        if k in merged and isinstance(merged[k], dict) and isinstance(v, dict):
            merged[k] = _deep_merge(merged[k], v)
        else:
            merged[k] = v
    return merged


def load_config(config_path: str = None, cli_overrides: dict = None) -> Config:
    """Load YAML config, apply optional CLI overrides, return Config object.

    Parameters
    ----------
    config_path : str, optional
        Path to a YAML config file.  Falls back to ``configs/default.yaml``.
    cli_overrides : dict, optional
        Flat or nested dict of values that override the YAML.

    Returns
    -------
    Config
        A namespace-like object with dot-access to all settings.
    """
    if config_path is None:
        config_path = os.path.join(os.path.dirname(__file__), "default.yaml")

    with open(config_path, "r") as f:
        raw = yaml.safe_load(f)

    if cli_overrides:
        raw = _deep_merge(raw, cli_overrides)

    # Convert to Config (which is a SimpleNamespace subclass)
    ns = _dict_to_namespace(raw)
    cfg = Config(**vars(ns))
    return cfg


def parse_cli_to_overrides() -> tuple:
    """Minimal CLI parser that returns (config_path, overrides_dict).

    Usage examples::

        python train.py --config my.yaml --agent.gamma 0.95 --training.max_episodes 5000
    """
    parser = argparse.ArgumentParser(
        description="DRL Drone Collision Avoidance Training"
    )
    parser.add_argument(
        "--config", type=str, default=None, help="Path to YAML config file"
    )
    # Capture everything else as key=value overrides
    args, unknown = parser.parse_known_args()

    overrides = {}
    i = 0
    while i < len(unknown):
        token = unknown[i]
        if token.startswith("--"):
            key = token.lstrip("-")
            # Next token is the value
            if i + 1 < len(unknown) and not unknown[i + 1].startswith("--"):
                val_str = unknown[i + 1]
                i += 2
            else:
                val_str = "true"
                i += 1
            # Auto-cast
            val = _auto_cast(val_str)
            # Support dotted keys like agent.gamma -> {"agent": {"gamma": ...}}
            _set_nested(overrides, key.split("."), val)
        else:
            i += 1

    return args.config, overrides


def _auto_cast(s: str):
    """Try to cast a string to int, float, bool, or leave as str."""
    if s.lower() in ("true", "yes"):
        return True
    if s.lower() in ("false", "no"):
        return False
    try:
        return int(s)
    except ValueError:
        pass
    try:
        return float(s)
    except ValueError:
        pass
    # Handle lists like "[42,12,34]"
    if s.startswith("[") and s.endswith("]"):
        inner = s[1:-1]
        return [_auto_cast(x.strip()) for x in inner.split(",") if x.strip()]
    return s


def _set_nested(d: dict, keys: list, value):
    """Set a value in a nested dict given a list of keys."""
    for k in keys[:-1]:
        d = d.setdefault(k, {})
    d[keys[-1]] = value
