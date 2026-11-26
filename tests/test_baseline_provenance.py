import ast, hashlib, json
from pathlib import Path
import unittest
from classical_studies import baselines


class BaselineProvenanceTests(unittest.TestCase):
    def test_migrated_library_has_no_quantum_backend_or_cross_repository_imports(self):
        root = Path(baselines.__file__).parent
        blocked = {"qiskit", "pennylane", "pyqsp", "triple_alpha", "qutip"}
        for path in root.rglob("*.py"):
            tree = ast.parse(path.read_text())
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    names = [v.name.split(".")[0] for v in node.names]
                elif isinstance(node, ast.ImportFrom):
                    names = [(node.module or "").split(".")[0]]
                else:
                    continue
                self.assertTrue(blocked.isdisjoint(names), (path, names))
            self.assertNotIn("/Users/", path.read_text())
            self.assertNotIn("sys.path.insert", path.read_text())
        evidence = json.loads((root / "provenance.json").read_text())
        self.assertEqual(len(evidence["source_repository_commit"]), 40)
        self.assertGreater(len(evidence["files"]), 15)
        for item in evidence["files"]:
            self.assertTrue((root / item["destination"]).is_file())
            self.assertEqual(len(item["source_sha256"]), 64)

    def test_public_catalog_and_forecasting_data_match_source_hashes(self):
        root = Path(baselines.__file__).parent
        evidence = json.loads((root / "provenance.json").read_text())
        for item in evidence["files"]:
            if "/data/" in item["destination"]:
                self.assertEqual(
                    hashlib.sha256(
                        (root / item["destination"]).read_bytes()
                    ).hexdigest(),
                    item["source_sha256"],
                )


if __name__ == "__main__":
    unittest.main()
