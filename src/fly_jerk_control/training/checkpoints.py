"""Checkpoint I/O with config, seeds, versions, and provenance."""

from __future__ import annotations

import json
import platform
import subprocess
import sys
from pathlib import Path
from typing import Any

import torch
import yaml

from fly_jerk_control import __version__


def git_commit() -> str | None:
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            stderr=subprocess.DEVNULL,
            text=True,
        )
        return out.strip()
    except (subprocess.CalledProcessError, FileNotFoundError, OSError):
        return None


def package_versions() -> dict[str, str]:
    versions = {
        "python": sys.version,
        "platform": platform.platform(),
        "fly_jerk_control": __version__,
    }
    for name in ("numpy", "torch", "gymnasium", "networkx", "pandas", "matplotlib", "yaml"):
        try:
            mod = __import__(name if name != "yaml" else "yaml")
            versions[name] = getattr(mod, "__version__", "unknown")
        except Exception:
            versions[name] = "missing"
    return versions


def save_checkpoint(
    path: str | Path,
    *,
    model: torch.nn.Module,
    cfg: dict[str, Any],
    extra: dict[str, Any] | None = None,
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "model_state": model.state_dict(),
        "cfg": cfg,
        "versions": package_versions(),
        "git_commit": git_commit(),
        "extra": extra or {},
    }
    torch.save(payload, path)
    sidecar = path.with_suffix(".meta.yaml")
    meta = {
        "versions": payload["versions"],
        "git_commit": payload["git_commit"],
        "cfg": cfg,
        "extra": extra or {},
    }
    sidecar.write_text(yaml.safe_dump(meta, sort_keys=False), encoding="utf-8")


def load_checkpoint(path: str | Path, model: torch.nn.Module | None = None) -> dict[str, Any]:
    payload = torch.load(Path(path), map_location="cpu", weights_only=False)
    if model is not None:
        model.load_state_dict(payload["model_state"])
    return payload
