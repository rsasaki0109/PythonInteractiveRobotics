"""Bundle the pure-Python `pir` package + flagship examples into a zip.

The zip is loaded directly in the browser with `pyodide.unpackArchive`, so the
Pyodide playground runs the *real* example code with no build step and without
pulling matplotlib (the headless loop path is numpy-only). Re-run this whenever
`pir/` or a bundled example changes; the output is committed so GitHub Pages can
serve it.

Usage:
    python3 scripts/build_pyodide_bundle.py
"""

from __future__ import annotations

import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "docs" / "pyodide" / "pir_bundle.zip"

# Examples exposed to the browser PoC. Add to this list as more flagship loops
# get a JS renderer (see docs/pyodide_playground_strategy.md).
BUNDLED_EXAMPLES = [
    "examples/manipulation/01_pick_and_retry.py",
]


def iter_pir_sources():
    for path in sorted((ROOT / "pir").rglob("*.py")):
        if "__pycache__" in path.parts:
            continue
        yield path


def build() -> Path:
    OUT.parent.mkdir(parents=True, exist_ok=True)
    files = list(iter_pir_sources()) + [ROOT / rel for rel in BUNDLED_EXAMPLES]

    with zipfile.ZipFile(OUT, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for path in files:
            arcname = path.relative_to(ROOT).as_posix()
            zf.write(path, arcname)

    size_kb = OUT.stat().st_size / 1024
    print(f"wrote {OUT.relative_to(ROOT)}  ({len(files)} files, {size_kb:.1f} KB)")
    return OUT


if __name__ == "__main__":
    build()
