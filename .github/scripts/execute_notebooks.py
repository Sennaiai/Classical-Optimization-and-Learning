from pathlib import Path

import nbformat
from nbclient import NotebookClient


ROOT = Path(__file__).resolve().parents[2]


def main():
    project_root = ROOT / "projects"
    notebooks = sorted(project_root.rglob("*.ipynb"))
    runnable = [path for path in notebooks if path.parent.name != "forecasting"]

    if not runnable:
        raise SystemExit("No offline ML-AI notebooks were found to execute.")

    for path in notebooks:
        if path not in runnable:
            print(
                f"SKIP {path.relative_to(ROOT)}: local forecasting dataset is not included",
                flush=True,
            )

    for path in runnable:
        print(f"RUN {path.relative_to(ROOT)}", flush=True)
        notebook = nbformat.read(path, as_version=4)
        client = NotebookClient(
            notebook,
            kernel_name="python3",
            timeout=900,
            resources={"metadata": {"path": str(path.parent)}},
        )
        client.execute()
        print(f"PASS {path.relative_to(ROOT)}", flush=True)


if __name__ == "__main__":
    main()
