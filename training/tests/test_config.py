"""Unit tests for the config system."""

import os
import sys
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from training.configs.config import load_config, _deep_merge, _auto_cast


class TestConfig:
    def test_load_default_config(self):
        cfg = load_config()
        assert hasattr(cfg, "env")
        assert hasattr(cfg, "agent")
        assert hasattr(cfg, "reward")
        assert hasattr(cfg, "training")
        assert cfg.env.num_drones == 10
        assert cfg.agent.gamma == 0.99

    def test_override_values(self):
        cfg = load_config(cli_overrides={"agent": {"gamma": 0.5}})
        assert cfg.agent.gamma == 0.5
        # Other values should remain default
        assert cfg.env.num_drones == 10

    def test_deep_merge(self):
        base = {"a": {"b": 1, "c": 2}, "d": 3}
        override = {"a": {"b": 99}}
        merged = _deep_merge(base, override)
        assert merged["a"]["b"] == 99
        assert merged["a"]["c"] == 2
        assert merged["d"] == 3

    def test_auto_cast(self):
        assert _auto_cast("42") == 42
        assert _auto_cast("3.14") == 3.14
        assert _auto_cast("true") is True
        assert _auto_cast("false") is False
        assert _auto_cast("hello") == "hello"

    def test_config_to_dict(self):
        cfg = load_config()
        d = cfg.to_dict()
        assert isinstance(d, dict)
        assert "env" in d
        assert d["env"]["num_drones"] == 10


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
