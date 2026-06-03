"""Guard the committed Pyodide bundle against drift from the source tree.

The browser PoC (docs/pyodide/poc.html) runs the real `pir` code by unpacking
docs/pyodide/pir_bundle.zip. If `pir/` or a bundled example changes but the zip
is not rebuilt, the browser would silently run stale code. These tests fail in
that case and tell you to re-run scripts/build_pyodide_bundle.py.
"""

from __future__ import annotations

import importlib.util
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BUNDLE = ROOT / "docs" / "pyodide" / "pir_bundle.zip"


def _load_builder():
    spec = importlib.util.spec_from_file_location(
        "build_pyodide_bundle", ROOT / "scripts" / "build_pyodide_bundle.py"
    )
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_bundle_exists() -> None:
    assert BUNDLE.exists(), "run: python3 scripts/build_pyodide_bundle.py"


def test_bundle_contents_match_source() -> None:
    builder = _load_builder()
    expected = {p.relative_to(ROOT).as_posix() for p in builder.iter_pir_sources()}
    expected |= set(builder.BUNDLED_EXAMPLES)

    with zipfile.ZipFile(BUNDLE) as zf:
        archived = {info.filename: zf.read(info.filename) for info in zf.infolist()}

    assert set(archived) == expected, (
        "bundle file list is stale; re-run python3 scripts/build_pyodide_bundle.py"
    )

    for arcname, data in archived.items():
        on_disk = (ROOT / arcname).read_bytes()
        assert data == on_disk, (
            f"{arcname} differs from source; re-run python3 scripts/build_pyodide_bundle.py"
        )


def test_bundled_example_runs_headless_without_matplotlib() -> None:
    """The browser path imports numpy only; prove the loop never needs matplotlib."""
    import sys

    class _Block:
        def find_spec(self, name, path=None, target=None):
            if name == "matplotlib" or name.startswith("matplotlib."):
                raise ImportError("blocked for test")
            return None

    blocker = _Block()
    sys.meta_path.insert(0, blocker)
    try:
        example = ROOT / "examples" / "manipulation" / "01_pick_and_retry.py"
        spec = importlib.util.spec_from_file_location("pick_and_retry_pyodide", example)
        assert spec and spec.loader
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        trace = module.run(seed=3, render=False)
        summary = trace.summary()
        assert summary.success
        assert summary.failure_counts.get("grasp_miss", 0) >= 1
    finally:
        sys.meta_path.remove(blocker)
