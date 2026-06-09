import json
import os
import hashlib
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from mlProject import logger


def compute_file_hash(filepath: Path) -> str:
    """Compute SHA-256 hash of a file."""
    sha256 = hashlib.sha256()
    with open(filepath, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            sha256.update(chunk)
    return sha256.hexdigest()[:16]


def load_registry(registry_path: Path) -> dict:
    """Load model registry from JSON file."""
    if registry_path.exists():
        try:
            with open(registry_path, "r") as f:
                return json.load(f)
        except (json.JSONDecodeError, IOError) as e:
            logger.error(f"Failed to load registry: {e}")
    return {"production": None, "staging": None, "versions": []}


def save_registry(registry_path: Path, registry: dict):
    """Save model registry to JSON file."""
    registry_path.parent.mkdir(parents=True, exist_ok=True)
    with open(registry_path, "w") as f:
        json.dump(registry, f, indent=2)
    logger.info(f"Model registry saved to {registry_path}")


def get_version_id() -> str:
    """Generate a globally unique version ID using timestamp and UUID."""
    ts = datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')
    suffix = uuid.uuid4().hex[:8]
    return f"v{ts}_{suffix}"


def register_model(
    registry_path: Path,
    model_path: Path,
    version_id: str,
    metrics: dict,
    params: dict,
    data_hash: Optional[str] = None,
    max_versions_to_keep: int = 10,
    quality_gate_max_rmse_degradation_pct: float = 5.0,
) -> dict:
    """Register a model version and enforce quality gates."""
    registry = load_registry(registry_path)

    for v in registry.get("versions", []):
        if v.get("id") == version_id:
            raise ValueError(f"Version ID {version_id} already exists in registry")

    current_production = registry.get("production")
    previous_metrics = None
    if current_production:
        for v in registry.get("versions", []):
            if v.get("id") == current_production:
                previous_metrics = v.get("metrics", {})
                break

    status = "staging"
    if previous_metrics and "rmse" in previous_metrics and "rmse" in metrics:
        prev_rmse = previous_metrics["rmse"]
        new_rmse = metrics["rmse"]
        if prev_rmse > 0:
            degradation_pct = ((new_rmse - prev_rmse) / prev_rmse) * 100
            if degradation_pct > quality_gate_max_rmse_degradation_pct:
                status = "rejected"
                logger.warning(
                    f"Model {version_id} REJECTED: RMSE degradation {degradation_pct:.2f}% "
                    f"exceeds threshold {quality_gate_max_rmse_degradation_pct}%"
                )
            else:
                status = "production"
                registry["production"] = version_id
                logger.info(
                    f"Model {version_id} PROMOTED to production: "
                    f"RMSE degradation {degradation_pct:.2f}% within threshold"
                )
        else:
            status = "production"
            registry["production"] = version_id
    else:
        status = "production"
        registry["production"] = version_id
        logger.info(f"First model {version_id} registered as production")

    entry = {
        "id": version_id,
        "path": str(model_path),
        "metrics": metrics,
        "params": params,
        "date": datetime.now(timezone.utc).isoformat(),
        "data_hash": data_hash or "",
        "status": status,
    }

    registry["versions"].insert(0, entry)

    if len(registry["versions"]) > max_versions_to_keep:
        archived = registry["versions"][max_versions_to_keep:]
        registry["versions"] = registry["versions"][:max_versions_to_keep]
        for v in archived:
            archived_path = Path(v["path"])
            if archived_path.exists():
                archived_path.unlink()
                logger.info(f"Deleted archived model file: {archived_path}")
            sha_path = Path(str(archived_path) + ".sha256")
            if sha_path.exists():
                sha_path.unlink()
                logger.info(f"Deleted archived checksum: {sha_path}")

    save_registry(registry_path, registry)
    return entry


def get_production_model_path(registry_path: Path) -> Optional[Path]:
    """Get the production model path from the registry."""
    registry = load_registry(registry_path)
    production_id = registry.get("production")
    if production_id:
        for v in registry.get("versions", []):
            if v.get("id") == production_id:
                return Path(v["path"])
    return None


def get_staging_model_path(registry_path: Path) -> Optional[Path]:
    """Get the staging model path from the registry."""
    registry = load_registry(registry_path)
    staging_id = registry.get("staging")
    if staging_id:
        for v in registry.get("versions", []):
            if v.get("id") == staging_id:
                return Path(v["path"])
    return None


def rollback_to_version(registry_path: Path, version_id: str) -> bool:
    """Rollback production alias to a specific version."""
    registry = load_registry(registry_path)
    for v in registry.get("versions", []):
        if v.get("id") == version_id:
            registry["production"] = version_id
            save_registry(registry_path, registry)
            logger.info(f"Rolled back production to version {version_id}")
            return True
    logger.error(f"Version {version_id} not found in registry")
    return False