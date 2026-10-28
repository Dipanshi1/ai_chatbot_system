"""Hashing and version capture for the B1 provenance envelope (spec §14)."""
import hashlib
import importlib.metadata
import platform
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, Optional

import config

B1_PACKAGES = (
    "torch",
    "sentence-transformers",
    "transformers",
    "tokenizers",
    "huggingface_hub",
    "safetensors",
    "scikit-learn",
    "numpy",
    "scipy",
    "psutil",
    "pandas",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 16), b""):
            digest.update(chunk)
    return digest.hexdigest()


def package_versions(packages: Iterable[str] = B1_PACKAGES) -> Dict[str, Optional[str]]:
    versions: Dict[str, Optional[str]] = {}
    for pkg in packages:
        try:
            versions[pkg] = importlib.metadata.version(pkg)
        except importlib.metadata.PackageNotFoundError:
            versions[pkg] = None
    return versions


def stage_a_hashes() -> Dict[str, str]:
    return {str(p.relative_to(config.REPO_ROOT)): sha256_file(p) for p in config.STAGE_A_PROTECTED_FILES}


def git_head() -> Optional[str]:
    try:
        out = subprocess.run(
            ["git", "-C", str(config.REPO_ROOT), "rev-parse", "HEAD"],
            capture_output=True, text=True, check=True,
        )
        return out.stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def timestamp_utc() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def python_version() -> str:
    return platform.python_version()
