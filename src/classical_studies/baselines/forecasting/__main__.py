import argparse, json
from pathlib import Path
from .api import run_classical

p = argparse.ArgumentParser()
p.add_argument("--data", type=Path)
p.add_argument("--output", type=Path, required=True)
a = p.parse_args()
r = run_classical(a.data)
a.output.parent.mkdir(parents=True, exist_ok=True)
a.output.write_text(json.dumps(r, indent=2, allow_nan=False) + "\n")
