"""Configuration loading with environment expansion and optional local overlays."""

from __future__ import annotations

import argparse
import os
import shutil
from collections.abc import Mapping
from pathlib import Path
from string import Template

import yaml


def config_root() -> Path:
    """Directory containing configs, in a checkout or an installed distribution."""
    if root := os.environ.get("CELLBRIDGE_ROOT"):
        return Path(root).expanduser().resolve()
    checkout = Path(__file__).resolve().parents[2]
    if (checkout / "configs/inference").is_dir():
        return checkout
    return Path(__file__).resolve().parent


def _defaults() -> dict[str, str]:
    root = config_root()
    checkout = Path(__file__).resolve().parents[2]
    # Keep the research workspace layout for checkouts; wheels use the caller's
    # working directory, never site-packages, for mutable data and results.
    workspace = Path(
        os.environ.get(
            "CELLBRIDGE_WORKSPACE",
            checkout.parents[1]
            if (checkout / "src/cellbridge").is_dir()
            else Path.cwd(),
        )
    )
    return {
        "CELLBRIDGE_ROOT": str(root),
        **{
            f"CELLBRIDGE_{name}_ROOT": str(workspace / folder)
            for name, folder in (
                ("DATA", "data"),
                ("REPOS", "repos"),
                ("ENV", "envs"),
                ("OUTPUTS", "outputs"),
                ("CHECKPOINTS", "checkpoints"),
            )
        },
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
            raise ValueError(
                f"undefined configuration variable: {error.args[0]}"
            ) from error
    if isinstance(value, list):
        return [_expand(item, environment) for item in value]
    if isinstance(value, Mapping):
        return {key: _expand(item, environment) for key, item in value.items()}
    return value


def _read_config(path: Path) -> dict:
    with open(path) as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict):
        raise TypeError(f"{path}: configuration must be a YAML mapping")

    defaults = config.pop("defaults", None)
    if defaults is not None:
        if not isinstance(defaults, str):
            raise TypeError(f"{path}: defaults must be a path")
        defaults_path = Path(defaults)
        if not defaults_path.is_absolute():
            defaults_path = path.parent / defaults_path
        config = _merge(_read_config(defaults_path), config)
    return config


def load_config(path, *, overlay=None) -> dict:
    """Load YAML defaults and overlays, then expand ``$VARS`` recursively.

    A relative ``defaults`` path is resolved from the referring configuration.
    ``CELLBRIDGE_CONFIG_OVERLAY`` then supplies a machine-local overlay when the
    caller does not pass one explicitly. Shared paths have portable environment
    defaults inferred from this checkout.
    """
    path = Path(path)
    config = _read_config(path)

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


def main():
    """Copy the bundled configuration tree to a new, editable project directory."""
    parser = argparse.ArgumentParser(description=main.__doc__)
    parser.add_argument("destination", type=Path, help="Create DESTINATION/configs")
    args = parser.parse_args()
    destination = args.destination.expanduser().resolve()
    shutil.copytree(config_root() / "configs", destination / "configs")
    print(f"Copied configurations to {destination / 'configs'}")
    print(f"Set CELLBRIDGE_ROOT to {destination} when using these configurations.")


if __name__ == "__main__":
    main()
