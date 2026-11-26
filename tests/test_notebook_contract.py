import ast, json, unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


class NotebookContractTests(unittest.TestCase):
    def test_exactly_four_consolidated_studies(self):
        catalog = json.loads((ROOT / "study_catalog.json").read_text())
        self.assertEqual(
            set(catalog), {"nonconvex", "unit_commitment", "fuzzy", "option_pricing"}
        )
        self.assertEqual(
            {str(p.relative_to(ROOT)) for p in (ROOT / "projects").rglob("*.ipynb")},
            {v["notebook"] for v in catalog.values()},
        )

    def test_notebooks_are_executable_concise_and_honestly_scoped(self):
        for p in (ROOT / "projects").rglob("*.ipynb"):
            nb = json.loads(p.read_text())
            words = 0
            execution = False
            for cell in nb["cells"]:
                text = "".join(cell["source"])
                self.assertTrue(
                    all(ord(c) >= 32 or c in "\n\t" for c in text),
                    (p, "hidden control character"),
                )
                if cell["cell_type"] == "code":
                    ast.parse(text)
                    self.assertNotIn("pip install", text)
                    execution |= "report = run('" in text
                    self.assertNotIn("/Users/", text)
                    self.assertNotIn("/private/tmp/", text)
                else:
                    words += len(text.split())
            self.assertLess(words, 250, p)
            self.assertTrue(execution, p)

    def test_internal_guidelines_are_ignored(self):
        self.assertIn("AGENTS.md", (ROOT / ".gitignore").read_text().splitlines())


if __name__ == "__main__":
    unittest.main()
