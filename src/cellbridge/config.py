"""Configuration loading with environment expansion and optional local overlays."""

from __future__ import annotations

import os
from collections.abc import Mapping
from pathlib import Path
from string import Template

import yaml


def _defaults() -> dict[str, str]:
    root = Path(__file__).resolve().parents[2]
    workspace = root.parents[1]
    return {
        "CELLBRIDGE_ROOT": str(root),
        "CELLBRIDGE_DATA_ROOT": str(workspace / "data"),
        "CELLBRIDGE_REPOS_ROOT": str(workspace / "repos"),
        "CELLBRIDGE_ENV_ROOT": str(workspace / "envs"),
        "CELLBRIDGE_OUTPUTS_ROOT": str(workspace / "outputs"),
        "CELLBRIDGE_CHECKPOINTS_ROOT": str(workspace / "checkpoints"),
    }


def _merge(base, override):
    if isinstance(base, Mapping) and isinstance(override, Mapping):
        result = dict(base)
        for key, value in override.items():
            result[key] = _merge(result.get(key), value)
        return result
    return override


def _expand(value, environment):
    if isinstance(value, str):
        try:
            return Template(value).substitute(environment)
        except KeyError as error:
            raise ValueError(f"undefined configuration variable: {error.args[0]}") from error
    if isinstance(value, list):
        return [_expand(item, environment) for item in value]
    if isinstance(value, Mapping):
        return {key: _expand(item, environment) for key, item in value.items()}
    return value


def load_config(path, *, overlay=None) -> dict:
    """Load YAML, merge a local overlay, and expand ``$VARS`` recursively.

    ``CELLBRIDGE_CONFIG_OVERLAY`` supplies a machine-local overlay when the
    caller does not pass one explicitly. Shared paths have portable defaults
    inferred from this checkout and can be overridden by environment variables.
    """
    path = Path(path)
    with open(path) as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict):
        raise ValueError(f"{path}: configuration must be a YAML mapping")

    overlay = overlay or os.environ.get("CELLBRIDGE_CONFIG_OVERLAY")
    if overlay:
        overlay_path = Path(overlay)
        with open(overlay_path) as handle:
            local = yaml.safe_load(handle)
        if not isinstance(local, dict):
            raise ValueError(f"{overlay_path}: overlay must be a YAML mapping")
        config = _merge(config, local)

    environment = {**_defaults(), **os.environ}
    return _expand(config, environment)


def config_value(path, dotted_key, *, overlay=None):
    """Return one required value from a loaded config using ``a.b.c`` syntax."""
    value = load_config(path, overlay=overlay)
    for part in dotted_key.split("."):
        if not isinstance(value, Mapping) or part not in value:
            raise KeyError(f"{path}: missing configuration key {dotted_key!r}")
        value = value[part]
    return value
