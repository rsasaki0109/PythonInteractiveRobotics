from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]

NOTEBOOKS = {
    "notebooks/pick_and_retry.ipynb": "examples/manipulation/01_pick_and_retry.py",
    "notebooks/safety_filter_cbf.ipynb": "examples/navigation/29_safety_filter_cbf.py",
    "notebooks/human_correction_replanning.ipynb": (
        "examples/navigation/34_human_correction_replanning.py"
    ),
    "notebooks/clarifying_question.ipynb": "examples/embodied_ai/35_clarifying_question.py",
}


def test_colab_notebooks_are_valid_and_reference_examples() -> None:
    for notebook_path, example_path in NOTEBOOKS.items():
        notebook = json.loads((ROOT / notebook_path).read_text(encoding="utf-8"))
        assert notebook["nbformat"] == 4
        assert notebook["cells"]
        assert (ROOT / example_path).exists()

        source = "\n".join(
            "".join(cell.get("source", [])) for cell in notebook["cells"]
        )
        assert example_path in source
        assert "sys.modules[spec.name] = module" in source
        assert "trace.summary()" in source
        assert "colab.research.google.com" in source
        assert "docs/assets/gifs/" in source
