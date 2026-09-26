"""Where MACE models, their copies, and large downloads live.

**The MACE folder** holds the foundation models and fine-tuned models
(``finetuned/``); the panel's default model comes from here. It is, in order:

1. ``$SAMSON_MLIP_MACE_DIR``;
2. ``mace_dir`` in the per-user config file (``config.json`` next to the
   panel's settings);
3. ``$XDG_CACHE_HOME/mace``, mace-torch's own rule;
4. ``~/.cache/mace``.

**The mirror folder** (optional; ``$SAMSON_MLIP_MIRROR_DIR`` or ``mirror_dir``
in the config, set with :func:`set_mirror_dir`) is a second copy, typically on
another drive. Installed fine-tuned models are copied there too, and large
downloads (the ~595 MB Materials Project replay set) go there instead of the
MACE folder. It is used only while it is reachable: with the drive
unplugged, everything falls back to the MACE folder.
"""

from __future__ import annotations

import json
import os
import shutil
from pathlib import Path

MACE_DIR_ENV = "SAMSON_MLIP_MACE_DIR"
MIRROR_DIR_ENV = "SAMSON_MLIP_MIRROR_DIR"
MACE_MP_SMALL = "20231210mace128L0_energy_epoch249model"


def config_file() -> Path:
    from .remote.protocol import data_dir

    return data_dir() / "config.json"


def _config() -> dict:
    try:
        return json.loads(config_file().read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}


def _set(key: str, value: str | None) -> None:
    config = _config()
    if value is None:
        config.pop(key, None)
    else:
        config[key] = value
    target = config_file()
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(config, indent=1), encoding="utf-8")


def mace_dir() -> Path:
    """The MACE folder (see the module docstring for the order)."""
    configured = os.environ.get(MACE_DIR_ENV) or _config().get("mace_dir")
    if configured:
        return Path(configured).expanduser()
    return Path(os.environ.get("XDG_CACHE_HOME") or Path.home() / ".cache") / "mace"


def set_mace_dir(path: str | Path | None) -> None:
    """Remember ``path`` as this user's MACE folder (``None``: back to the default)."""
    _set("mace_dir", None if path is None else str(Path(path).expanduser().resolve()))


def mirror_dir() -> Path | None:
    """The mirror folder, or ``None`` if none is set or its drive is not there."""
    configured = os.environ.get(MIRROR_DIR_ENV) or _config().get("mirror_dir")
    if not configured:
        return None
    path = Path(configured).expanduser()
    return path if Path(path.anchor or ".").exists() else None


def set_mirror_dir(path: str | Path | None) -> None:
    """Remember ``path`` as this user's mirror folder (``None``: no mirror)."""
    _set("mirror_dir", None if path is None else str(Path(path).expanduser().resolve()))


def finetuned_dir() -> Path:
    """Where installed fine-tuned models go: ``<MACE folder>/finetuned``."""
    return mace_dir() / "finetuned"


def foundation_model(name: str = MACE_MP_SMALL) -> Path:
    """A foundation model file in the MACE folder (MACE-MP-0 small by default)."""
    return mace_dir() / name


def downloads_dir() -> Path:
    """Where large downloads go: the mirror folder when it is there, else the MACE folder."""
    return mirror_dir() or mace_dir()


def mace_environment(environment: dict | None = None) -> dict:
    """``environment`` (default: this process's) with ``XDG_CACHE_HOME`` set so
    mace-torch reads and downloads in :func:`downloads_dir` (when it is named ``mace``)."""
    environment = dict(os.environ if environment is None else environment)
    folder = downloads_dir()
    if folder.name == "mace":
        environment["XDG_CACHE_HOME"] = str(folder.parent)
    return environment


def mirror(path: str | Path) -> Path | None:
    """Copy a file or folder inside the MACE folder to the same place in the
    mirror folder; returns the copy, or ``None`` without a reachable mirror."""
    target_root = mirror_dir()
    if target_root is None:
        return None
    path = Path(path).resolve()
    try:
        relative = path.relative_to(mace_dir().resolve())
    except ValueError:
        relative = Path(path.name)
    target = target_root / relative
    target.parent.mkdir(parents=True, exist_ok=True)
    if path.is_dir():
        shutil.copytree(path, target, dirs_exist_ok=True)
    else:
        shutil.copy2(path, target)
    return target
