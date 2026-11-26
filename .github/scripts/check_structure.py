"""Fast repository contracts; no scientific computation or backend dependency."""

import ast, json, subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
catalog = json.loads((ROOT / "study_catalog.json").read_text())
expected = {v["notebook"] for v in catalog.values()}
actual = {str(p.relative_to(ROOT)) for p in (ROOT / "projects").rglob("*.ipynb")}
assert len(expected) == 4 and actual == expected
for path in actual:
    nb = json.loads((ROOT / path).read_text())
    assert nb["nbformat"] == 4
    for c in nb["cells"]:
        text = "".join(c["source"])
        assert all(ord(ch) >= 32 or ch in "\n\t" for ch in text)
        if c["cell_type"] == "code":
            ast.parse(text)
            assert "pip install" not in text
        assert not any(o.get("output_type") == "error" for o in c.get("outputs", []))
tracked = subprocess.check_output(["git", "ls-files"], cwd=ROOT, text=True).splitlines()
assert not any(Path(p).name == "AGENTS.md" for p in tracked), (
    "Internal guidelines are tracked"
)
assert not any("/__pycache__/" in p or "/.ipynb_checkpoints/" in p for p in tracked)
print(
    "PASS: four source notebooks, valid code, no saved errors or tracked internal guidelines."
)
