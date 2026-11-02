"""Integrity of the saved models frozen from published releases."""
import hashlib
import json
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FIXTURES = ROOT / "tests/fixtures/saved-models"


class SavedModelFixtureTests(unittest.TestCase):
    def manifests(self):
        return [(p.parent, json.loads(p.read_text())) for p in sorted(FIXTURES.glob("*/manifest.json"))]

    def test_first_public_release_is_frozen(self):
        self.assertIn("0.2.0", [manifest["release"] for _, manifest in self.manifests()])

    def test_files_match_their_recorded_hashes(self):
        for root, manifest in self.manifests():
            self.assertEqual(root.name, manifest["release"])
            recorded = set()
            for family in manifest["families"].values():
                for name, digest in family["sha256"].items():
                    with self.subTest(release=root.name, file=name):
                        self.assertEqual(hashlib.sha256((root / name).read_bytes()).hexdigest(), digest)
                    recorded.add(name)
            present = {p.relative_to(root).as_posix() for p in root.glob("*/*")}
            self.assertEqual(recorded, present)

    def test_models_record_the_revision_of_their_release(self):
        for root, manifest in self.manifests():
            for name in manifest["families"]:
                with self.subTest(release=root.name, family=name):
                    model = json.loads((root / "models" / f"{name}.json").read_text())
                    self.assertEqual(model["algorithm_version"], manifest["algorithm_version"])


if __name__ == "__main__":
    unittest.main()
