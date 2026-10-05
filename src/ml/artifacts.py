"""
Artifact manifest generation and integrity verification for the University Support Chatbot.

Provides tools to build, persist, and verify the integrity manifest of canonical
model and vectorizer artifacts according to Phase 5A requirements.
"""
import hashlib
import importlib.metadata
import json
import platform
from pathlib import Path
from typing import Any, Dict, Optional, Union

import joblib

from .config import ARTIFACTS_DIR, MANIFEST_PATH, MODEL_PATH, VECTORIZER_PATH

# Relevant runtime packages whose versions should be recorded
RELEVANT_PACKAGES = (
    "scikit-learn",
    "joblib",
    "numpy",
    "pandas",
    "openpyxl",
    "streamlit",
    "matplotlib",
    "seaborn",
)


def compute_sha256(path: Union[Path, str]) -> str:
    """
    Compute the full SHA-256 hexadecimal digest of a file.

    Parameters
    ----------
    path : Path or str
        Path to the file to hash.

    Returns
    -------
    str
        64-character lowercase hexadecimal SHA-256 digest.
    """
    file_path = Path(path)
    if not file_path.exists():
        raise FileNotFoundError(f"File not found for SHA-256 calculation: {file_path}")

    digest = hashlib.sha256()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _get_package_versions() -> Dict[str, str]:
    """Inspect installed versions of relevant packages."""
    versions = {}
    for pkg in RELEVANT_PACKAGES:
        try:
            versions[pkg] = importlib.metadata.version(pkg)
        except importlib.metadata.PackageNotFoundError:
            continue
    return versions


def build_manifest(
    model_path: Optional[Union[Path, str]] = None,
    vectorizer_path: Optional[Union[Path, str]] = None,
) -> Dict[str, Any]:
    """
    Inspect existing model and vectorizer artifacts and build an integrity manifest dict.

    Does NOT train or modify any artifacts.

    Parameters
    ----------
    model_path : Path or str, optional
        Path to the saved intent classification model (defaults to config.MODEL_PATH).
    vectorizer_path : Path or str, optional
        Path to the saved TF-IDF vectorizer (defaults to config.VECTORIZER_PATH).

    Returns
    -------
    Dict[str, Any]
        Manifest dictionary containing metadata, hashes, versions, and feature counts.

    Raises
    ------
    FileNotFoundError
        If either artifact file does not exist.
    """
    m_path = Path(model_path) if model_path is not None else MODEL_PATH
    v_path = Path(vectorizer_path) if vectorizer_path is not None else VECTORIZER_PATH

    if not m_path.exists():
        raise FileNotFoundError(f"Model artifact not found: {m_path}")
    if not v_path.exists():
        raise FileNotFoundError(f"Vectorizer artifact not found: {v_path}")

    # Compute SHA-256 hashes
    model_sha256 = compute_sha256(m_path)
    vectorizer_sha256 = compute_sha256(v_path)

    # Inspect model and vectorizer content safely (read-only)
    model = joblib.load(m_path)
    vectorizer = joblib.load(v_path)

    classes = [str(c) for c in getattr(model, "classes_", [])]
    num_classes = len(classes)
    vocabulary = getattr(vectorizer, "vocabulary_", {})
    num_features = len(vocabulary)

    # Construct model_version matching pipeline fingerprint
    model_version = f"linsvc-{model_sha256[:12]}+tfidf-{vectorizer_sha256[:12]}"
    vectorizer_version = f"tfidf-{vectorizer_sha256[:12]}"

    pkg_versions = _get_package_versions()

    manifest = {
        "model_filename": m_path.name,
        "vectorizer_filename": v_path.name,
        "model_sha256": model_sha256,
        "vectorizer_sha256": vectorizer_sha256,
        "model_version": model_version,
        "vectorizer_version": vectorizer_version,
        "num_classes": num_classes,
        "number_of_classes": num_classes,
        "num_features": num_features,
        "number_of_tfidf_features": num_features,
        "classes": classes,
        "python_version": platform.python_version(),
        "package_versions": pkg_versions,
        "packages": pkg_versions,
    }

    return manifest


def write_manifest(
    manifest_path: Optional[Union[Path, str]] = None,
    model_path: Optional[Union[Path, str]] = None,
    vectorizer_path: Optional[Union[Path, str]] = None,
) -> Path:
    """
    Generate and write artifacts/manifest.json from existing artifacts.

    Does NOT train or overwrite any model/vectorizer artifacts.

    Parameters
    ----------
    manifest_path : Path or str, optional
        Target path for the manifest JSON file (defaults to config.MANIFEST_PATH).
    model_path : Path or str, optional
        Path to the model artifact.
    vectorizer_path : Path or str, optional
        Path to the vectorizer artifact.

    Returns
    -------
    Path
        Path to the written manifest file.
    """
    target_path = Path(manifest_path) if manifest_path is not None else MANIFEST_PATH
    manifest = build_manifest(model_path=model_path, vectorizer_path=vectorizer_path)

    target_path.parent.mkdir(parents=True, exist_ok=True)
    with open(target_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    return target_path


def verify_artifacts(
    manifest_path: Optional[Union[Path, str]] = None,
    artifacts_dir: Optional[Union[Path, str]] = None,
    model_path: Optional[Union[Path, str]] = None,
    vectorizer_path: Optional[Union[Path, str]] = None,
) -> bool:
    """
    Verify existing artifacts against the manifest.

    Validates:
    - manifest exists and is valid JSON
    - model and vectorizer files exist
    - SHA-256 hashes match the manifest exactly
    - number of classes matches the manifest
    - number of TF-IDF features matches the manifest

    Parameters
    ----------
    manifest_path : Path or str, optional
        Path to manifest.json (defaults to config.MANIFEST_PATH).
    artifacts_dir : Path or str, optional
        Directory containing artifacts (defaults to config.ARTIFACTS_DIR).
    model_path : Path or str, optional
        Explicit path to model file (overrides artifacts_dir lookup if provided).
    vectorizer_path : Path or str, optional
        Explicit path to vectorizer file (overrides artifacts_dir lookup if provided).

    Returns
    -------
    bool
        True if all verification checks pass.

    Raises
    ------
    FileNotFoundError
        If manifest or any referenced artifact is missing.
    ValueError
        If any hash, class count, or feature count check fails.
    """
    mf_path = Path(manifest_path) if manifest_path is not None else MANIFEST_PATH
    art_dir = Path(artifacts_dir) if artifacts_dir is not None else ARTIFACTS_DIR

    if not mf_path.exists():
        raise FileNotFoundError(f"Manifest not found at: {mf_path}")

    with open(mf_path, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    # Resolve artifact file paths
    m_path = (
        Path(model_path)
        if model_path is not None
        else art_dir / manifest.get("model_filename", "intent_model.pkl")
    )
    v_path = (
        Path(vectorizer_path)
        if vectorizer_path is not None
        else art_dir / manifest.get("vectorizer_filename", "tfidf_vectorizer.pkl")
    )

    if not m_path.exists():
        raise FileNotFoundError(f"Model artifact not found at: {m_path}")
    if not v_path.exists():
        raise FileNotFoundError(f"Vectorizer artifact not found at: {v_path}")

    # Check SHA-256 hashes
    actual_model_hash = compute_sha256(m_path)
    expected_model_hash = manifest.get("model_sha256")
    if actual_model_hash != expected_model_hash:
        raise ValueError(
            f"Model SHA-256 mismatch for {m_path.name}!\n"
            f"  Expected: {expected_model_hash}\n"
            f"  Actual:   {actual_model_hash}"
        )

    actual_vec_hash = compute_sha256(v_path)
    expected_vec_hash = manifest.get("vectorizer_sha256")
    if actual_vec_hash != expected_vec_hash:
        raise ValueError(
            f"Vectorizer SHA-256 mismatch for {v_path.name}!\n"
            f"  Expected: {expected_vec_hash}\n"
            f"  Actual:   {actual_vec_hash}"
        )

    # Load artifacts and verify structural contents
    model = joblib.load(m_path)
    vectorizer = joblib.load(v_path)

    expected_classes = manifest.get("num_classes", manifest.get("number_of_classes"))
    actual_classes = len(getattr(model, "classes_", []))
    if actual_classes != expected_classes:
        raise ValueError(
            f"Model classes count mismatch! Expected {expected_classes}, found {actual_classes}"
        )

    expected_features = manifest.get("num_features", manifest.get("number_of_tfidf_features"))
    actual_features = len(getattr(vectorizer, "vocabulary_", {}))
    if actual_features != expected_features:
        raise ValueError(
            f"Vectorizer features count mismatch! Expected {expected_features}, found {actual_features}"
        )

    return True
