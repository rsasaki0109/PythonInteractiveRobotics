from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType

import numpy as np


ROOT = Path(__file__).resolve().parents[1]


def load_example(relative_path: str) -> ModuleType:
    path = ROOT / relative_path
    spec = importlib.util.spec_from_file_location(path.stem, path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_sense_act_loop_runs_headless() -> None:
    module = load_example("examples/runtime/01_sense_act_loop.py")

    trace = module.run(seed=0, render_enabled=False, max_steps=8)

    assert len(trace.actions) > 0
    assert len(trace.actions) == len(trace.rewards)


def test_trace_replay_runs_headless() -> None:
    module = load_example("examples/runtime/26_trace_replay.py")

    trace = module.run(seed=0, render=False, max_steps=12)
    summary = trace.summary()

    assert summary.steps == 12
    assert summary.success is False
    assert summary.final_info["replay_index"] == 11
    assert summary.final_info["source_steps"] == 12
    assert "cumulative_reward" in summary.final_info
    assert len(trace.observations) == len(trace.actions)
    assert not trace.failures()


def test_pick_and_retry_runs_headless() -> None:
    module = load_example("examples/manipulation/01_pick_and_retry.py")

    trace = module.run(seed=3, render=False, max_steps=20)

    assert len(trace.actions) > 0
    assert len(trace.actions) == len(trace.infos)
    assert any(failure.kind == "grasp_miss" for failure in trace.failures())


def test_reactive_grasping_runs_headless() -> None:
    module = load_example("examples/manipulation/02_reactive_grasping.py")

    trace = module.run(seed=4, render=False, max_steps=60)

    assert trace.infos[-1]["success"] is True
    assert trace.infos[-1]["miss_count"] == 1
    assert trace.infos[-1]["servo_steps"] >= 10
    assert trace.infos[-1]["reactive_updates"] >= 3
    assert any(failure.kind == "grasp_miss" for failure in trace.failures())


def test_closed_loop_ik_runs_headless() -> None:
    module = load_example("examples/manipulation/03_closed_loop_ik.py")

    trace = module.run(seed=2, render=False, max_steps=80)

    assert trace.infos[-1]["success"] is True
    assert trace.infos[-1]["tracking_error"] < 0.03
    assert trace.infos[-1]["stable_count"] >= 7
    assert trace.infos[-1]["servo_updates"] >= 10
    assert trace.infos[-1]["belief_updates"] >= 5
    assert not trace.failures()


def test_moving_target_reaching_runs_headless() -> None:
    module = load_example("examples/manipulation/04_moving_target_reaching.py")

    trace = module.run(seed=5, render=False, max_steps=90)

    assert trace.infos[-1]["success"] is True
    assert trace.infos[-1]["reach_error"] < 0.04
    assert trace.infos[-1]["touch_count"] >= 4
    assert trace.infos[-1]["occlusion_count"] > 0
    assert trace.infos[-1]["prediction_updates"] > 0
    assert any(failure.kind == "target_occluded" for failure in trace.failures())


def test_object_search_and_pick_runs_headless() -> None:
    module = load_example("examples/manipulation/05_object_search_and_pick.py")

    trace = module.run(seed=7, render=False, max_steps=30)

    assert trace.infos[-1]["success"] is True
    assert trace.infos[-1]["memory_count"] >= 3
    assert trace.infos[-1]["search_failure_count"] >= 1
    assert trace.infos[-1]["retry_count"] == 1
    assert trace.infos[-1]["target_confidence"] >= 0.9
    assert any(failure.kind == "target_not_visible" for failure in trace.failures())
    assert any(failure.kind == "grasp_miss" for failure in trace.failures())


def test_push_then_grasp_runs_headless() -> None:
    module = load_example("examples/manipulation/06_push_then_grasp.py")

    trace = module.run(seed=9, render=False, max_steps=25)

    assert trace.infos[-1]["success"] is True
    assert trace.infos[-1]["push_count"] == 1
    assert trace.infos[-1]["retry_count"] == 1
    assert trace.infos[-1]["blocked_count"] == 1
    assert trace.infos[-1]["environment_changes"] == 1
    assert any(failure.kind == "blocked_grasp" for failure in trace.failures())


def test_probabilistic_suction_sorting_runs_headless() -> None:
    module = load_example("examples/manipulation/07_probabilistic_suction_sorting.py")

    trace = module.run(seed=11, render=False, max_steps=40)

    assert trace.infos[-1]["success"] is True
    assert trace.infos[-1]["sorted_count"] == 3
    assert trace.infos[-1]["failure_count"] == 1
    assert trace.infos[-1]["retry_count"] == 1
    assert trace.infos[-1]["prepare_count"] == 1
    assert any(failure.kind == "suction_miss" for failure in trace.failures())


def test_belief_grasp_selection_runs_headless() -> None:
    module = load_example("examples/manipulation/08_belief_grasp_selection.py")

    trace = module.run(seed=0, render=False, max_steps=10, true_pose=0)

    final = trace.infos[-1]
    assert final["success"] is True
    assert final["belief_update_count"] >= 1
    assert final["failed_attempts"] >= 1
    assert any(failure.kind == "grasp_miss" for failure in trace.failures())
    final_belief = final["belief"]
    assert int(max(range(len(final_belief)), key=lambda i: final_belief[i])) == 0


def test_clear_path_before_pick_runs_headless() -> None:
    module = load_example("examples/manipulation/25_clear_path_before_pick.py")

    trace = module.run(seed=0, render=False, max_steps=15)

    final = trace.infos[-1]
    assert final["success"] is True
    assert final["precondition_failure_count"] == 1
    assert final["clear_step_count"] == 1
    assert final["retry_count"] == 1
    assert any(failure.kind == "precondition_blocked" for failure in trace.failures())
    assert any(info.get("agent_state") == "clear_obstacle" for info in trace.infos)
    assert any(info.get("agent_state") == "place_obstacle" for info in trace.infos)
    assert any(info.get("agent_state") == "retry_target" for info in trace.infos)


def test_active_viewpoint_for_grasp_runs_headless() -> None:
    module = load_example("examples/manipulation/09_active_viewpoint_for_grasp.py")

    trace = module.run(seed=4, render=False, max_steps=14, true_pose=2)

    final = trace.infos[-1]
    assert final["success"] is True
    assert final["view_count"] >= 2
    assert final["belief_update_count"] >= 2
    assert final["failed_attempts"] >= 1
    assert any(failure.kind == "grasp_miss" for failure in trace.failures())
    assert any(info.get("action_type") == "look" for info in trace.infos)
    assert any(info.get("action_type") == "grasp" for info in trace.infos)


def test_reactive_obstacle_avoidance_runs_headless() -> None:
    module = load_example("examples/navigation/02_reactive_obstacle_avoidance.py")

    trace = module.run(seed=0, render=False, max_steps=80)

    assert trace.infos[-1]["success"] is True
    assert trace.infos[-1]["avoidance_count"] > 0
    assert not trace.failures()


def test_dynamic_obstacle_avoidance_runs_headless() -> None:
    module = load_example("examples/navigation/03_dynamic_obstacle_avoidance.py")

    trace = module.run(seed=0, render=False, max_steps=90)

    assert trace.infos[-1]["success"] is True
    assert trace.infos[-1]["reactive_avoid_count"] > 0
    assert not trace.failures()


def test_online_replanning_astar_runs_headless() -> None:
    module = load_example("examples/navigation/04_online_replanning_astar.py")

    trace = module.run(seed=0, render=False, max_steps=100)

    assert trace.infos[-1]["success"] is True
    assert trace.infos[-1]["replan_count"] >= 2
    assert trace.infos[-1]["path_invalidations"] >= 1
    assert not trace.failures()


def test_frontier_exploration_runs_headless() -> None:
    module = load_example("examples/navigation/05_frontier_exploration.py")

    trace = module.run(seed=0, render=False, max_steps=120, coverage_goal=0.58)

    assert trace.infos[-1]["success"] is True
    assert trace.infos[-1]["coverage"] >= 0.58
    assert trace.infos[-1]["frontier_switches"] > 1
    assert not trace.failures()


def test_belief_based_navigation_runs_headless() -> None:
    module = load_example("examples/navigation/06_belief_based_navigation.py")

    trace = module.run(seed=0, render=False, max_steps=90)

    assert trace.infos[-1]["success"] is True
    assert trace.infos[-1]["entropy"] < 1.0
    assert trace.infos[-1]["localization_count"] >= 1
    assert not trace.failures()


def test_goal_command_pick_runs_headless() -> None:
    module = load_example("examples/embodied_ai/01_goal_command_pick.py")

    trace = module.run(
        command="find the red block and pick it",
        seed=3,
        render=False,
        max_steps=40,
    )

    assert trace.infos[-1]["success"] is True
    assert trace.infos[-1]["parsed_goal"] == {
        "intent": "find_and_pick",
        "object": "block",
        "color": "red",
    }
    assert trace.infos[-1]["memory_count"] >= 1
    assert trace.infos[-1]["retry_count"] >= 1
    assert any(failure.kind == "grasp_miss" for failure in trace.failures())


def test_goal_command_parser_rejects_unknown_command() -> None:
    module = load_example("examples/embodied_ai/01_goal_command_pick.py")

    parsed = module.parse_goal_command("pick anything")

    assert parsed["intent"] == "unknown"
    assert parsed["message"] == "unsupported command"


def test_active_slam_toy_runs_headless() -> None:
    module = load_example("examples/navigation/07_active_slam_toy.py")

    trace = module.run(seed=0, render=False, max_steps=100)

    assert trace.infos[-1]["success"] is True
    assert trace.infos[-1]["pose_entropy"] <= 1.15
    assert trace.infos[-1]["map_entropy"] <= 0.36
    assert "information_gain" in trace.infos[-1]
    assert not trace.failures()


def test_interactive_mpc_runs_headless() -> None:
    module = load_example("examples/navigation/08_interactive_mpc.py")

    trace = module.run(seed=0, render=False, max_steps=120)

    assert trace.infos[-1]["success"] is True
    assert trace.infos[-1]["replan_count"] >= len(trace.actions)
    assert "best_cost" in trace.infos[-1]
    assert "predicted_collision_risk" in trace.infos[-1]
    assert not trace.failures()


def test_blocked_path_recovery_runs_headless() -> None:
    module = load_example("examples/navigation/09_blocked_path_recovery.py")

    trace = module.run(seed=0, render=False, max_steps=80)

    assert trace.infos[-1]["success"] is True
    assert trace.infos[-1]["detected_block_count"] == 1
    assert trace.infos[-1]["recovery_count"] >= 1
    assert trace.infos[-1]["replan_count"] >= 2
    assert any(failure.kind == "blocked_path" for failure in trace.failures())


def test_information_gain_navigation_runs_headless() -> None:
    module = load_example("examples/navigation/24_information_gain_navigation.py")

    trace = module.run(seed=0, render=False, max_steps=100, candidate_open=True)

    final = trace.infos[-1]
    assert final["success"] is True
    assert final["info_gain_step_count"] >= 1
    assert final["navigation_step_count"] >= 1
    assert final["observed_candidate"] is True
    assert final["belief"] == 1.0
    assert final["replan_count"] >= 2
    assert any(info.get("agent_state") == "scout" for info in trace.infos)
    assert any(info.get("agent_state") == "navigate" for info in trace.infos)
    assert not trace.failures()


def test_information_gain_navigation_closed_gate() -> None:
    module = load_example("examples/navigation/24_information_gain_navigation.py")

    trace = module.run(seed=0, render=False, max_steps=100, candidate_open=False)

    final = trace.infos[-1]
    assert final["success"] is True
    assert final["belief"] == 0.0
    assert final["observed_candidate"] is True
    assert final["info_gain_step_count"] >= 1


def test_multi_agent_avoidance_runs_headless() -> None:
    module = load_example("examples/navigation/27_multi_agent_avoidance.py")

    trace = module.run(seed=0, render=False, max_steps=80)

    final = trace.infos[-1]
    assert final["success"] is True
    assert final["replan_count"] >= 2
    assert final["avoidance_count"] >= 1
    assert not trace.failures()
    assert any("predicted_next" in info for info in trace.infos)
    # both other agents should be predicted at some step
    seen_predictions = set()
    for info in trace.infos:
        for aid in info.get("predicted_next", {}):
            seen_predictions.add(aid)
    assert seen_predictions == {"A", "B"}


def test_curiosity_grid_exploration_runs_headless() -> None:
    module = load_example("examples/embodied_ai/28_curiosity_grid_exploration.py")

    trace = module.run(seed=0, render=False, max_steps=120, coverage_threshold=0.70)

    final = trace.infos[-1]
    assert final["success"] is True
    assert final["coverage"] >= 0.70
    assert final["target_switches"] >= 2
    # commitment to paths: not switching every single step
    assert final["target_switches"] <= len(trace.actions) // 2
    assert not trace.failures()


def test_safety_filter_cbf_runs_headless() -> None:
    module = load_example("examples/navigation/29_safety_filter_cbf.py")

    trace = module.run(seed=0, render=False, max_steps=200)

    final = trace.infos[-1]
    assert final["success"] is True
    # the filter must engage along the way - this is the whole point
    assert final["filter_active_count"] >= 5
    # but the filter must also let the robot reach the goal
    assert final["stuck_count"] == 0
    # closest approach should be inside the safety margin but strictly above
    # collision (clearance > robot_radius)
    assert final["closest_approach"] > 0.0
    # no collision or timeout
    assert not trace.failures()
    # nominal velocity must differ from safe velocity on at least one step
    assert any(
        info.get("u_nominal") is not None
        and info.get("u_safe") is not None
        and not np.allclose(info["u_nominal"], info["u_safe"])
        for info in trace.infos
    )


def test_localization_uncertainty_recovery_runs_headless() -> None:
    module = load_example("examples/navigation/10_localization_uncertainty_recovery.py")

    trace = module.run(seed=0, render=False, max_steps=60)

    final_info = trace.infos[-1]
    assert final_info["success"] is True
    assert final_info["localization_recovery_count"] >= 1
    assert final_info["entropy"] < 0.55
    assert any(info.get("agent_state") == "localize" for info in trace.infos)
    assert any(info.get("agent_state") == "go_to_goal" for info in trace.infos)
    assert not trace.failures()


def test_door_search_pomdp_runs_headless() -> None:
    module = load_example("examples/embodied_ai/10_door_search_pomdp.py")

    trace = module.run(seed=0, render=False, max_steps=40)

    assert trace.infos[-1]["success"] is True
    assert trace.infos[-1]["locked_failures"] == 1
    assert trace.infos[-1]["not_found_failures"] == 1
    assert trace.infos[-1]["key_belief"]["storage"] == 0.0
    assert trace.infos[-1]["key_belief"]["bedroom"] == 0.0
    assert any(failure.kind == "locked_door" for failure in trace.failures())
    assert any(failure.kind == "not_found" for failure in trace.failures())


def test_goal_conditioned_minikitchen_runs_headless() -> None:
    module = load_example("examples/embodied_ai/18_goal_conditioned_minikitchen.py")

    trace = module.run(
        command="bring mug to table",
        seed=0,
        render=False,
        max_steps=35,
    )

    assert trace.infos[-1]["success"] is True
    assert trace.infos[-1]["parsed_goal"] == {
        "intent": "bring",
        "object": "mug",
        "destination": "table",
    }
    assert trace.infos[-1]["closed_failures"] == 1
    assert trace.infos[-1]["not_found_failures"] == 1
    assert trace.infos[-1]["open_count"] == 1
    assert trace.infos[-1]["object_memory"]["mug"] == "table"
    assert any(failure.kind == "target_not_found" for failure in trace.failures())
    assert any(failure.kind == "container_closed" for failure in trace.failures())


def test_tiny_vla_loop_runs_headless() -> None:
    module = load_example("examples/embodied_ai/19_tiny_vla_loop.py")

    trace = module.run(
        command="place red block in blue bin",
        seed=0,
        render=False,
        max_steps=25,
    )

    assert trace.infos[-1]["success"] is True
    assert trace.infos[-1]["parsed_goal"] == {
        "intent": "place_in",
        "object_color": "red",
        "object_name": "block",
        "destination_color": "blue",
        "destination_name": "bin",
    }
    assert trace.infos[-1]["skill_count"] == 3
    assert trace.infos[-1]["recovery_count"] == 1
    assert trace.infos[-1]["visual_updates"] >= 4
    assert any(failure.kind == "visual_pose_uncertain" for failure in trace.failures())


def test_object_permanence_toy_runs_headless() -> None:
    module = load_example("examples/embodied_ai/21_object_permanence_toy.py")

    trace = module.run(seed=0, render=False, max_steps=30)

    final = trace.infos[-1]
    assert final["success"] is True
    assert final["observation_count"] >= 1
    assert final["memory_persistence_count"] >= 1
    assert final["has_memory"] is True
    assert not trace.failures()
    assert any(info.get("action_type") == "peek" for info in trace.infos)
    assert any(info.get("action_type") == "move" for info in trace.infos)


def test_tiny_world_model_planning_runs_headless() -> None:
    module = load_example("examples/world_models/20_tiny_world_model_planning.py")

    trace = module.run(seed=0, render=False, max_steps=80)

    assert trace.infos[-1]["success"] is True
    assert trace.infos[-1]["model_error_count"] >= 1
    assert trace.infos[-1]["model_update_count"] >= len(trace.actions)
    assert trace.infos[-1]["learned_residual_norm"] > 0.02
    assert any(failure.kind == "model_error" for failure in trace.failures())


def test_model_error_recovery_runs_headless() -> None:
    module = load_example("examples/world_models/23_model_error_recovery.py")

    trace = module.run(seed=0, render=False, max_steps=50)

    final = trace.infos[-1]
    assert final["success"] is True
    assert final["model_error_count"] >= 1
    assert final["model_update_count"] >= 1
    assert final["recovery_count"] >= 1
    learned = final["learned_offset"]
    assert abs(learned[0]) > 0.01
    assert abs(learned[1]) > 0.01
    assert any(failure.kind == "model_drift" for failure in trace.failures())
    assert any(info.get("action_type") == "probe" for info in trace.infos)
    assert any(info.get("agent_state") == "system_id" for info in trace.infos)
    assert any(info.get("agent_state") == "go_to_goal" for info in trace.infos)
