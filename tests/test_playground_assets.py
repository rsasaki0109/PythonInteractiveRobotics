from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_playground_assets_support_shareable_scenarios() -> None:
    html = (ROOT / "docs/playground.html").read_text(encoding="utf-8")
    css = (ROOT / "docs/playground.css").read_text(encoding="utf-8")
    js = (ROOT / "docs/playground.js").read_text(encoding="utf-8")

    for asset in ("playground.css", "playground.js"):
        assert asset in html

    for element_id in (
        "scenarioSelect",
        "answerSelect",
        "failureFilter",
        "copyLinkButton",
        "copyTraceButton",
        "copyStatus",
        "replaySlider",
        "replayValue",
        "traceRows",
    ):
        assert f'id="{element_id}"' in html

    for value in (
        "scenario",
        "answer",
        "autoplay",
        "failure",
        "ambiguous_goal",
        "unsafe_nominal_step",
        "grasp_miss",
        "human_correction",
        "clampReplayIndex",
        "snapshotForReplayIndex",
        "trace-active",
    ):
        assert value in js

    assert "URLSearchParams" in js
    assert "navigator.clipboard" in js
    assert "formatTraceText" in js
    assert ".copy-status" in css
    assert ".replay-strip" in css


def test_readme_links_to_shareable_live_trace() -> None:
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    index = (ROOT / "docs/index.html").read_text(encoding="utf-8")
    share_url = "playground.html?scenario=household&answer=red&autoplay=1"

    assert "Shareable live trace" in readme
    assert share_url in readme
    assert share_url in index
