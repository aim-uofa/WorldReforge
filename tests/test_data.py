from pathlib import Path
import tempfile
import unittest

from worldreforge.data.manifest import load_manifest, summary
from worldreforge.io import checked_file, contained


class InputTests(unittest.TestCase):
    def test_manifest(self):
        path = Path(__file__).resolve().parents[1] / "manifests/eval1200.jsonl"
        result = summary(load_manifest(path))
        self.assertEqual(result["windows"], 1200)
        self.assertEqual(result["domains"], {"sim": 1200})
        self.assertEqual(result["motion_reference_windows"], 1200)

    def test_path_escape_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(ValueError):
                contained(directory, "../outside")
            path = Path(directory) / "link"
            path.symlink_to(Path(directory).parent)
            with self.assertRaises(ValueError):
                contained(directory, "link/outside")

    def test_input_file_only_requires_an_existing_relative_path(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "sample.json").write_text("{}")
            self.assertEqual(checked_file(root, {"path": "sample.json"}), root / "sample.json")
            for path in ("missing.json", "."):
                with self.subTest(path=path), self.assertRaises(FileNotFoundError):
                    checked_file(root, {"path": path})


if __name__ == "__main__":
    unittest.main()
