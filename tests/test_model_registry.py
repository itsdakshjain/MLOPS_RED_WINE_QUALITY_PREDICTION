import json
import tempfile
import unittest
from pathlib import Path

from mlProject.utils.model_registry import (
    get_version_id,
    load_registry,
    register_model,
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


if __name__ == "__main__":
    unittest.main()
