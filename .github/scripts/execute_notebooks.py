"""Execute only the selected public experiment, using this interpreter's kernel."""

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import nbformat
from nbclient import NotebookClient

ROOT = Path(__file__).resolve().parents[2]
CATALOG = json.loads((ROOT / "study_catalog.json").read_text())


def execute(study, publish=False):
    spec = CATALOG[study]
    source = ROOT / spec["notebook"]
    artifact = ROOT / "artifacts" / study
    artifact.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="classical-notebook-") as temporary:
        temp = Path(temporary)
        env_keys = {
            "JUPYTER_PATH": str(temp / "share/jupyter"),
            "JUPYTER_RUNTIME_DIR": str(temp / "runtime"),
            "IPYTHONDIR": str(temp / "ipython"),
            "MPLCONFIGDIR": str(temp / "mpl"),
            "MPLBACKEND": "Agg",
        }
        old = {k: os.environ.get(k) for k in env_keys}
        os.environ.update(env_keys)
        try:
            subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "ipykernel",
                    "install",
                    "--prefix",
                    str(temp),
                    "--name",
                    "classical-studies",
                ],
                check=True,
                capture_output=True,
            )
            nb = nbformat.read(source, as_version=4)
            client = NotebookClient(
                nb,
                kernel_name="classical-studies",
                timeout=240,
                resources={"metadata": {"path": str(source.parent)}},
            )
            client.execute()
            nbformat.write(nb, artifact / "executed.ipynb")
            if publish:
                nbformat.write(nb, source)
                results = source.parent / "results"
                results.mkdir(exist_ok=True)
                for name in ("results.json", "benchmark.svg"):
                    (results / name).write_bytes((artifact / name).read_bytes())
        finally:
            for k, v in old.items():
                if v is None:
                    os.environ.pop(k, None)
                else:
                    os.environ[k] = v
    print("PASS", study, flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--study", choices=["all", *CATALOG], default="all")
    p.add_argument("--publish-evidence", action="store_true")
    a = p.parse_args()
    for name in CATALOG if a.study == "all" else [a.study]:
        execute(name, a.publish_evidence)
