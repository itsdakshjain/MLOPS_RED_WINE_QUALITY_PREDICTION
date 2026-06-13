import json
import tempfile
import unittest
from pathlib import Path

from mlProject.utils.model_registry import (
    get_version_id,
    load_registry,
    register_model,
    rollback_to_version,
    validate_registry,
)


class TestModelRegistry(unittest.TestCase):
    def test_get_version_id_is_unique(self):
        ids = {get_version_id() for _ in range(100)}
        self.assertEqual(len(ids), 100)

    def test_get_version_id_format(self):
        vid = get_version_id()
        self.assertTrue(vid.startswith("v"))
        self.assertIn("_", vid)

    def test_register_model_rejects_duplicate_version_id(self):
        with tempfile.TemporaryDirectory() as tmp:
            registry_path = Path(tmp) / "registry.json"
            model_path = Path(tmp) / "model.joblib"
            model_path.write_text("dummy")
            vid = "v20260609_143021_test"

            register_model(
                registry_path=registry_path,
                model_path=model_path,
                version_id=vid,
                metrics={"rmse": 0.5},
                params={"alpha": 0.1},
            )

            with self.assertRaises(ValueError):
                register_model(
                    registry_path=registry_path,
                    model_path=model_path,
                    version_id=vid,
                    metrics={"rmse": 0.6},
                    params={"alpha": 0.2},
                )

    def test_archived_model_files_cleaned_up(self):
        with tempfile.TemporaryDirectory() as tmp:
            registry_path = Path(tmp) / "registry.json"
            model_paths = []
            for i in range(3):
                mp = Path(tmp) / f"model_v{i}.joblib"
                mp.write_text(f"dummy{i}")
                (Path(str(mp) + ".sha256")).write_text(f"hash{i}")
                model_paths.append(mp)

            for i, mp in enumerate(model_paths):
                register_model(
                    registry_path=registry_path,
                    model_path=mp,
                    version_id=f"v{i:04d}",
                    metrics={"rmse": 0.5},
                    params={"alpha": 0.1},
                    max_versions_to_keep=2,
                )

            registry = load_registry(registry_path)
            self.assertEqual(len(registry["versions"]), 2)
            self.assertFalse(model_paths[0].exists())
            self.assertFalse(Path(str(model_paths[0]) + ".sha256").exists())


    def test_rollback_copies_versioned_file_to_stable(self):
        with tempfile.TemporaryDirectory() as tmp:
            registry_path = Path(tmp) / "registry.json"
            stable_path = Path(tmp) / "model.joblib"
            versioned_path = Path(tmp) / "model_v001.joblib"
            versioned_path.write_text("versioned_model_weights")
            stable_path.write_text("old_weights")

            register_model(
                registry_path=registry_path,
                model_path=versioned_path,
                version_id="v001",
                metrics={"rmse": 0.5},
                params={"alpha": 0.1},
            )

            stable_path.write_text("corrupted_weights")

            result = rollback_to_version(registry_path, "v001")
            self.assertTrue(result)
            self.assertEqual(stable_path.read_text(), "versioned_model_weights")
            registry = load_registry(registry_path)
            self.assertEqual(registry["production"], "v001")

    def test_rollback_fails_when_versioned_file_missing(self):
        with tempfile.TemporaryDirectory() as tmp:
            registry_path = Path(tmp) / "registry.json"
            versioned_path = Path(tmp) / "model_v001.joblib"
            versioned_path.write_text("weights")

            register_model(
                registry_path=registry_path,
                model_path=versioned_path,
                version_id="v001",
                metrics={"rmse": 0.5},
                params={"alpha": 0.1},
            )

            versioned_path.unlink()

            result = rollback_to_version(registry_path, "v001")
            self.assertFalse(result)

    def test_rollback_fails_for_nonexistent_version(self):
        with tempfile.TemporaryDirectory() as tmp:
            registry_path = Path(tmp) / "registry.json"
            result = rollback_to_version(registry_path, "v_nonexistent")
            self.assertFalse(result)

    def test_validate_registry_reports_missing_files(self):
        with tempfile.TemporaryDirectory() as tmp:
            registry_path = Path(tmp) / "registry.json"
            existing_path = Path(tmp) / "model_v001.joblib"
            existing_path.write_text("weights")
            (Path(str(existing_path) + ".sha256")).write_text("hash")
            (Path(tmp) / "model.joblib").write_text("weights")
            missing_path = Path(tmp) / "model_v002.joblib"

            register_model(
                registry_path=registry_path,
                model_path=existing_path,
                version_id="v001",
                metrics={"rmse": 0.5},
                params={"alpha": 0.1},
            )
            register_model(
                registry_path=registry_path,
                model_path=missing_path,
                version_id="v002",
                metrics={"rmse": 0.6},
                params={"alpha": 0.2},
            )

            issues = validate_registry(registry_path)
            self.assertTrue(any("v002" in issue for issue in issues))
            v001_issues = [i for i in issues if "v001" in i]
            self.assertEqual(len(v001_issues), 0)

    def test_validate_registry_passes_with_all_files_present(self):
        with tempfile.TemporaryDirectory() as tmp:
            registry_path = Path(tmp) / "registry.json"
            model_path = Path(tmp) / "model_v001.joblib"
            model_path.write_text("weights")
            sha_path = Path(str(model_path) + ".sha256")
            sha_path.write_text("hash")
            (Path(tmp) / "model.joblib").write_text("weights")

            register_model(
                registry_path=registry_path,
                model_path=model_path,
                version_id="v001",
                metrics={"rmse": 0.5},
                params={"alpha": 0.1},
            )

            issues = validate_registry(registry_path)
            self.assertEqual(issues, [])


if __name__ == "__main__":
    unittest.main()
