from __future__ import annotations

import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
IMAGE_PATTERN = re.compile(r"!\[[^\]]*\]\(([^)]+)\)")

EXPECTED_CATEGORY_GIFS = {
    "examples/manipulation/README.md": {
        "pick_and_retry.gif",
        "reactive_grasping.gif",
        "closed_loop_ik.gif",
        "moving_target_reaching.gif",
        "object_search_and_pick.gif",
        "push_then_grasp.gif",
        "probabilistic_suction_sorting.gif",
    },
    "examples/navigation/README.md": {
        "reactive_obstacle_avoidance.gif",
        "dynamic_obstacle_avoidance.gif",
        "online_replanning_astar.gif",
        "frontier_exploration.gif",
        "belief_based_navigation.gif",
        "active_slam_toy.gif",
        "interactive_mpc.gif",
        "blocked_path_recovery.gif",
        "human_correction_replanning.gif",
    },
    "examples/embodied_ai/README.md": {
        "goal_command_pick.gif",
        "door_search_pomdp.gif",
        "goal_conditioned_minikitchen.gif",
        "tiny_vla_loop.gif",
        "clarifying_question.gif",
        "household_task_agent.gif",
    },
}


def test_markdown_image_paths_exist_and_category_galleries_are_populated() -> None:
    image_paths_by_markdown: dict[str, set[str]] = {}

    for markdown_path in ROOT.rglob("*.md"):
        text = markdown_path.read_text(encoding="utf-8")
        relative_markdown = markdown_path.relative_to(ROOT).as_posix()
        image_paths: set[str] = set()

        for match in IMAGE_PATTERN.finditer(text):
            raw_target = match.group(1).strip()
            image_target = raw_target.split()[0].strip("<>")
            if image_target.startswith(("http://", "https://", "#")):
                continue

            resolved = (markdown_path.parent / image_target).resolve()
            assert resolved.exists(), f"{relative_markdown} references missing image {image_target}"
            assert ROOT.resolve() in resolved.parents
            image_paths.add(Path(image_target).name)

        image_paths_by_markdown[relative_markdown] = image_paths

    for relative_markdown, expected_gifs in EXPECTED_CATEGORY_GIFS.items():
        assert expected_gifs <= image_paths_by_markdown[relative_markdown]
