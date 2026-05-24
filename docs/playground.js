(function () {
  "use strict";

  const map = [
    "############",
    "#..........#",
    "#..........#",
    "#..........#",
    "#..........#",
    "#..........#",
    "#..........#",
    "#..........#",
    "############",
  ];

  const sourceLinks = {
    clarifying:
      "https://github.com/rsasaki0109/PythonInteractiveRobotics/blob/main/examples/embodied_ai/35_clarifying_question.py",
    household:
      "https://github.com/rsasaki0109/PythonInteractiveRobotics/blob/main/examples/embodied_ai/36_household_task_agent.py",
  };
  const scenarioValues = new Set(["clarifying", "household"]);
  const answerValues = new Set(["red", "blue"]);
  const failureValues = new Set([
    "all",
    "ambiguous_goal",
    "unsafe_nominal_step",
    "grasp_miss",
    "human_correction",
  ]);
  const unsafeCells = [
    [5, 1],
    [5, 2],
    [5, 3],
  ];
  const humanCells = [
    [1, 6],
    [1, 7],
    [2, 6],
    [2, 7],
  ];

  const elements = {
    scenario: document.getElementById("scenarioSelect"),
    answer: document.getElementById("answerSelect"),
    failureFilter: document.getElementById("failureFilter"),
    compare: document.getElementById("compareToggle"),
    reset: document.getElementById("resetButton"),
    step: document.getElementById("stepButton"),
    run: document.getElementById("runButton"),
    copyLink: document.getElementById("copyLinkButton"),
    copyTrace: document.getElementById("copyTraceButton"),
    copyStatus: document.getElementById("copyStatus"),
    beliefPanel: document.getElementById("beliefPanel"),
    replay: document.getElementById("replaySlider"),
    replayValue: document.getElementById("replayValue"),
    comparePanel: document.getElementById("comparePanel"),
    timeline: document.getElementById("timeline"),
    scene: document.getElementById("scene"),
    traceRows: document.getElementById("traceRows"),
    stepCounter: document.getElementById("stepCounter"),
    command: document.getElementById("commandValue"),
    target: document.getElementById("targetValue"),
    agentState: document.getElementById("agentStateValue"),
    failure: document.getElementById("failureValue"),
    source: document.getElementById("sourceLink"),
  };

  applyInitialParams();
  let state = buildState();
  let timer = null;
  let copyStatusTimer = null;

  elements.scenario.addEventListener("change", () => {
    stopRun();
    state = buildState();
    updateLocation(false);
    render();
  });
  elements.answer.addEventListener("change", () => {
    stopRun();
    state = buildState();
    updateLocation(false);
    render();
  });
  elements.failureFilter.addEventListener("change", () => {
    updateLocation(false);
    render();
  });
  elements.compare.addEventListener("change", () => {
    updateLocation(false);
    render();
  });
  elements.replay.addEventListener("input", () => {
    state.replayIndex = clampReplayIndex(elements.replay.value);
    render();
  });
  elements.reset.addEventListener("click", () => {
    stopRun();
    state = buildState();
    updateLocation(false);
    render();
  });
  elements.step.addEventListener("click", () => {
    stepOnce();
  });
  elements.run.addEventListener("click", () => {
    if (timer) {
      stopRun();
      return;
    }
    startRun();
  });
  elements.copyLink.addEventListener("click", () => {
    copyText(getShareUrl(true), "Share link copied");
  });
  elements.copyTrace.addEventListener("click", () => {
    copyText(formatTraceText(), "Trace copied");
  });

  render();
  if (readAutoplayParam()) {
    window.setTimeout(startRun, 180);
  }

  function buildState() {
    const scenario = elements.scenario.value;
    const answer = elements.answer.value;
    const config =
      scenario === "household"
        ? buildHouseholdScenario(answer)
        : buildClarifyingScenario(answer);
    return {
      scenario,
      answer,
      config,
      index: 0,
      trace: [],
      replayIndex: null,
    };
  }

  function stepOnce() {
    if (state.index >= state.config.steps.length) {
      render();
      return false;
    }
    const event = state.config.steps[state.index];
    state.trace.push(event);
    state.index += 1;
    state.replayIndex = null;
    render();
    return state.index < state.config.steps.length;
  }

  function stopRun() {
    if (timer) {
      window.clearInterval(timer);
      timer = null;
    }
    elements.run.textContent = "Run";
  }

  function startRun() {
    if (state.index >= state.config.steps.length) {
      render();
      return;
    }
    elements.run.textContent = "Stop";
    if (!stepOnce()) {
      stopRun();
      return;
    }
    timer = window.setInterval(() => {
      if (!stepOnce()) {
        stopRun();
      }
    }, 760);
  }

  function buildClarifyingScenario(answer) {
    const targetX = answer === "blue" ? 68 : 32;
    const command = "pick the block";
    const initial = {
      type: "tabletop",
      command,
      agentState: "parse_command",
      target: "unresolved",
      failure: "none",
      belief: ambiguousBelief(),
      picked: null,
      pickAt: null,
      focus: null,
    };
    return {
      command,
      totalSteps: 3,
      initial,
      steps: [
        {
          action: "ask(which_block)",
          reward: -0.02,
          failure: "ambiguous_goal",
          agentState: "update_goal_from_answer",
          snapshot: {
            ...initial,
            target: answer,
            agentState: "update_goal_from_answer",
            failure: "ambiguous_goal",
            belief: resolvedBelief(answer),
            question: "Which block?",
            answer,
            focus: answer,
          },
        },
        {
          action: "look(" + answer + ")",
          reward: -0.01,
          failure: "",
          agentState: "target_confirmed",
          snapshot: {
            ...initial,
            target: answer,
            agentState: "target_confirmed",
            belief: resolvedBelief(answer),
            focus: answer,
          },
        },
        {
          action: "pick(" + answer + ")",
          reward: 1.0,
          failure: "",
          agentState: "done",
          snapshot: {
            ...initial,
            target: answer,
            agentState: "done",
            belief: resolvedBelief(answer),
            picked: answer,
            pickAt: [targetX, 56],
            focus: answer,
          },
        },
      ],
    };
  }

  function buildHouseholdScenario(answer) {
    const redRoute = [
      [7, 1],
      [6, 1],
      [6, 2],
      [6, 3],
      [6, 4],
      [5, 4],
      [4, 4],
      [4, 3],
      [3, 3],
    ];
    const blueRoute = [
      [7, 1],
      [6, 1],
      [6, 2],
      [6, 3],
      [6, 4],
      [5, 4],
      [5, 5],
      [5, 6],
      [5, 7],
      [5, 8],
      [5, 9],
    ];
    const correctedStorageRoute = [
      [2, 5],
      [3, 5],
      [3, 6],
      [3, 7],
      [3, 8],
      [2, 8],
      [1, 8],
      [1, 9],
      [1, 10],
    ];
    const blueStorageRoute = [
      [5, 9],
      [4, 9],
      [3, 9],
      [2, 9],
      [1, 9],
      [1, 10],
    ];

    const nominalRoute = [
      [7, 1],
      [6, 1],
      [5, 1],
      [4, 1],
      [3, 1],
      [3, 2],
      [3, 3],
    ];
    const targetRoute = answer === "blue" ? blueRoute : redRoute;
    const storageRoute =
      answer === "blue"
        ? blueStorageRoute
        : [
            [3, 3],
            [3, 4],
            [3, 5],
            [2, 5],
          ].concat(correctedStorageRoute.slice(1));
    const recoveryRoute = joinRoutes(targetRoute, storageRoute);

    const context = {
      command: "put the block away",
      robot: [7, 1],
      target: "unresolved",
      belief: ambiguousBelief(),
      held: null,
      stored: null,
      blocked: [],
      corrected: [],
      trail: [[7, 1]],
      picked: [],
      failure: "none",
      agentState: "parse_command",
      path: nominalRoute.map((cell) => cell.slice()),
    };
    const initial = {
      type: "household",
      command: "put the block away",
      robot: [7, 1],
      target: "unresolved",
      belief: ambiguousBelief(),
      held: null,
      stored: null,
      blocked: [],
      corrected: [],
      trail: [[7, 1]],
      picked: [],
      failure: "none",
      agentState: "parse_command",
      path: context.path.map((cell) => cell.slice()),
    };
    const steps = [];

    addStep(steps, context, {
      action: "ask(which_block)",
      reward: -0.02,
      failure: "ambiguous_goal",
      agentState: "update_goal_from_answer",
      target: answer,
      belief: resolvedBelief(answer),
      path: answer === "blue" ? blueRoute : redRoute,
    });

    move(steps, context, "move(north)", -0.02, [6, 1], "plan_to_pick");
    addStep(steps, context, {
      action: "move(north)",
      reward: -0.16,
      failure: "unsafe_nominal_step",
      agentState: "safety_filter_replan",
      blocked: [
        unsafeCells[0],
        unsafeCells[1],
        unsafeCells[2],
      ],
      path: answer === "blue" ? blueRoute.slice(1) : redRoute.slice(1),
    });

    const routeToTarget = (answer === "blue" ? blueRoute : redRoute).slice(2);
    for (const cell of routeToTarget) {
      move(steps, context, "move(" + directionName(context.robot, cell) + ")", -0.02, cell, "follow_safe_plan");
    }

    addStep(steps, context, {
      action: "pick(" + answer + ")",
      reward: -0.15,
      failure: "grasp_miss",
      agentState: "recover_from_grasp_miss",
      path: [context.robot],
    });
    addStep(steps, context, {
      action: "pick(" + answer + ")",
      reward: 0.2,
      failure: "",
      agentState: "picked_up",
      held: answer,
      picked: [answer],
      path: answer === "blue" ? blueStorageRoute : [[3, 3], [3, 4], [3, 5], [2, 5], [2, 6]],
    });

    if (answer === "blue") {
      for (const cell of blueStorageRoute.slice(1)) {
        move(steps, context, "move(" + directionName(context.robot, cell) + ")", -0.02, cell, "plan_to_storage");
      }
    } else {
      for (const cell of [
        [3, 4],
        [3, 5],
        [2, 5],
      ]) {
        move(steps, context, "move(" + directionName(context.robot, cell) + ")", -0.02, cell, "plan_to_storage");
      }
      addStep(steps, context, {
        action: "move(east)",
        reward: -0.18,
        failure: "human_correction",
        agentState: "learn_from_human_correction",
        corrected: [
          [1, 6],
          [1, 7],
          [2, 6],
          [2, 7],
        ],
        path: correctedStorageRoute,
      });
      for (const cell of correctedStorageRoute.slice(1)) {
        move(
          steps,
          context,
          "move(" + directionName(context.robot, cell) + ")",
          -0.02,
          cell,
          "replan_after_human_correction"
        );
      }
    }

    addStep(steps, context, {
      action: "place(storage)",
      reward: 1.0,
      failure: "",
      agentState: "done",
      held: null,
      stored: answer,
      path: [context.robot],
    });

    return {
      command: "put the block away",
      totalSteps: steps.length,
      initial,
      steps,
      compare: buildHouseholdCompare(answer, steps, nominalRoute, recoveryRoute),
    };
  }

  function buildHouseholdCompare(answer, steps, nominalRoute, recoveryRoute) {
    const target = answer === "blue" ? [5, 9] : [3, 3];
    return {
      baseline: {
        kind: "baseline",
        title: "Naive shortcut",
        outcome: "blocked",
        detail: "enters unsafe_nominal_step before reaching the target",
        route: nominalRoute,
        stop: [5, 1],
        target,
        targetLabel: answer === "blue" ? "B" : "R",
        storage: [1, 10],
        steps: 3,
        failures: 1,
        reward: -0.2,
      },
      recovery: {
        kind: "recovery",
        title: "Interactive recovery",
        outcome: "delivered",
        detail:
          answer === "red"
            ? "clarifies, avoids unsafe cells, retries grasp, and accepts correction"
            : "clarifies, avoids unsafe cells, retries grasp, and stores the block",
        route: recoveryRoute,
        target,
        targetLabel: answer === "blue" ? "B" : "R",
        storage: [1, 10],
        steps: steps.length,
        failures: steps.filter((step) => step.failure).length,
        reward: steps.reduce((total, step) => total + step.reward, 0),
      },
    };
  }

  function addStep(steps, context, update) {
    Object.assign(context, update);
    if (update.robot) {
      context.robot = update.robot.slice();
      context.trail = appendUnique(context.trail, context.robot);
    }
    if (update.blocked) {
      context.blocked = update.blocked.map((cell) => cell.slice());
    }
    if (update.corrected) {
      context.corrected = update.corrected.map((cell) => cell.slice());
    }
    if (update.picked) {
      context.picked = update.picked.slice();
    }
    const event = {
      action: update.action,
      reward: update.reward,
      failure: update.failure || "",
      agentState: update.agentState,
      snapshot: {
        type: "household",
        ...snapshotFromContext(context),
      },
    };
    steps.push(event);
  }

  function move(steps, context, action, reward, robot, agentState) {
    addStep(steps, context, {
      action,
      reward,
      failure: "",
      agentState,
      robot,
      path: trimPath(context.path, robot),
    });
  }

  function ambiguousBelief() {
    const distribution = { red: 0.5, blue: 0.5 };
    return {
      red: distribution.red,
      blue: distribution.blue,
      entropy: beliefEntropy(distribution),
      askGain: beliefEntropy(distribution),
      policy: "ask",
    };
  }

  function resolvedBelief(answer) {
    const distribution = {
      red: answer === "red" ? 1 : 0,
      blue: answer === "blue" ? 1 : 0,
    };
    return {
      red: distribution.red,
      blue: distribution.blue,
      entropy: beliefEntropy(distribution),
      askGain: 0,
      policy: "act",
    };
  }

  function beliefEntropy(distribution) {
    return Object.values(distribution).reduce((total, probability) => {
      if (probability <= 0) {
        return total;
      }
      return total - probability * Math.log2(probability);
    }, 0);
  }

  function snapshotFromContext(context) {
    return {
      command: context.command,
      robot: context.robot.slice(),
      target: context.target,
      belief: copyBelief(context.belief),
      held: context.held,
      stored: context.stored,
      blocked: context.blocked.map((cell) => cell.slice()),
      corrected: context.corrected.map((cell) => cell.slice()),
      trail: context.trail.map((cell) => cell.slice()),
      picked: context.picked.slice(),
      path: (context.path || []).map((cell) => cell.slice()),
      failure: context.failure || "none",
      agentState: context.agentState,
    };
  }

  function copyBelief(belief) {
    return {
      red: belief.red,
      blue: belief.blue,
      entropy: belief.entropy,
      askGain: belief.askGain,
      policy: belief.policy,
    };
  }

  function trimPath(path, robot) {
    if (!path || !path.length) {
      return [robot];
    }
    const index = path.findIndex((cell) => sameCell(cell, robot));
    return index >= 0 ? path.slice(index) : [robot];
  }

  function appendUnique(trail, robot) {
    const last = trail[trail.length - 1];
    if (last && sameCell(last, robot)) {
      return trail.map((cell) => cell.slice());
    }
    return trail.concat([robot.slice()]);
  }

  function joinRoutes(first, second) {
    return first.concat(second.slice(1)).map((cell) => cell.slice());
  }

  function directionName(start, end) {
    const dr = end[0] - start[0];
    const dc = end[1] - start[1];
    if (dr === -1) return "north";
    if (dr === 1) return "south";
    if (dc === -1) return "west";
    if (dc === 1) return "east";
    return "stay";
  }

  function render() {
    const replayIndex = currentReplayIndex();
    const current = snapshotForReplayIndex(replayIndex);
    elements.command.textContent = state.config.command;
    elements.target.textContent = current.target || "unresolved";
    elements.agentState.textContent = current.agentState || "parse_command";
    elements.failure.textContent = current.failure || "none";
    elements.stepCounter.textContent = state.index + " / " + state.config.totalSteps;
    elements.source.href = sourceLinks[state.scenario];
    elements.step.disabled = state.index >= state.config.steps.length;
    elements.run.disabled = state.index >= state.config.steps.length && !timer;
    elements.copyTrace.disabled = state.trace.length === 0;
    elements.compare.disabled = state.scenario !== "household";

    renderReplay(replayIndex);
    renderCompare();
    renderTimeline(replayIndex);
    renderBelief(current);
    renderScene(current);
    renderTrace(replayIndex);
  }

  function renderCompare() {
    const enabled = elements.compare.checked && state.scenario === "household";
    elements.comparePanel.hidden = !enabled;
    elements.comparePanel.textContent = "";
    if (!enabled || !state.config.compare) {
      return;
    }
    [state.config.compare.baseline, state.config.compare.recovery].forEach((plan) => {
      const lane = document.createElement("section");
      lane.className = "compare-lane " + plan.kind;

      const kicker = document.createElement("span");
      kicker.className = "compare-kicker";
      kicker.textContent = plan.title;
      lane.appendChild(kicker);

      const outcome = document.createElement("div");
      outcome.className = "compare-outcome";
      outcome.textContent = plan.outcome;
      lane.appendChild(outcome);

      const detail = document.createElement("p");
      detail.className = "compare-detail";
      detail.textContent = plan.detail;
      lane.appendChild(detail);

      const metrics = document.createElement("div");
      metrics.className = "compare-metrics";
      [
        ["steps", String(plan.steps)],
        ["failures", String(plan.failures)],
        ["reward", formatReward(plan.reward)],
      ].forEach(([label, value]) => {
        const metric = document.createElement("span");
        metric.textContent = label;
        const strong = document.createElement("strong");
        strong.textContent = value;
        metric.appendChild(strong);
        metrics.appendChild(metric);
      });
      lane.appendChild(metrics);
      lane.appendChild(renderMiniMap(plan));
      elements.comparePanel.appendChild(lane);
    });
  }

  function renderReplay(replayIndex) {
    elements.replay.max = String(state.trace.length);
    elements.replay.value = String(replayIndex);
    elements.replay.disabled = state.trace.length === 0;
    elements.replayValue.textContent = replayLabel(replayIndex);
  }

  function renderTimeline(replayIndex) {
    elements.timeline.textContent = "";
    state.trace.forEach((event, index) => {
      const step = index + 1;
      const button = document.createElement("button");
      button.className = "timeline-step " + timelineClass(event);
      button.type = "button";
      button.textContent = String(step);
      button.setAttribute("aria-label", timelineLabel(event, step));
      if (step === replayIndex) {
        button.classList.add("timeline-active");
        button.setAttribute("aria-current", "step");
      }
      button.addEventListener("click", () => setReplayIndex(step));
      elements.timeline.appendChild(button);
    });
  }

  function renderBelief(snapshot) {
    const belief = snapshot.belief;
    elements.beliefPanel.textContent = "";
    if (!belief) {
      elements.beliefPanel.hidden = true;
      return;
    }
    elements.beliefPanel.hidden = false;

    const bars = document.createElement("div");
    bars.className = "belief-bars";
    [
      ["red", belief.red],
      ["blue", belief.blue],
    ].forEach(([label, probability]) => {
      bars.appendChild(renderBeliefRow(label, probability));
    });

    const metrics = document.createElement("div");
    metrics.className = "belief-metrics";
    [
      ["entropy", formatBits(belief.entropy)],
      ["ask gain", "+" + formatBits(belief.askGain)],
      ["policy", belief.policy],
    ].forEach(([label, value]) => {
      const metric = document.createElement("span");
      metric.textContent = label;
      const strong = document.createElement("strong");
      strong.textContent = value;
      metric.appendChild(strong);
      metrics.appendChild(metric);
    });

    elements.beliefPanel.appendChild(bars);
    elements.beliefPanel.appendChild(metrics);
  }

  function renderBeliefRow(label, probability) {
    const row = document.createElement("div");
    row.className = "belief-row";

    const name = document.createElement("span");
    name.textContent = label;
    row.appendChild(name);

    const track = document.createElement("div");
    track.className = "belief-track";
    const fill = document.createElement("div");
    fill.className = "belief-fill belief-" + label;
    fill.style.width = Math.round(probability * 100) + "%";
    track.appendChild(fill);
    row.appendChild(track);

    const value = document.createElement("strong");
    value.className = "belief-value";
    value.textContent = Math.round(probability * 100) + "%";
    row.appendChild(value);

    return row;
  }

  function timelineClass(event) {
    if (event.failure === "ambiguous_goal") return "timeline-ambiguous";
    if (event.failure === "unsafe_nominal_step") return "timeline-unsafe";
    if (event.failure === "grasp_miss") return "timeline-grasp";
    if (event.failure === "human_correction") return "timeline-human";
    if (event.agentState === "done") return "timeline-success";
    return "timeline-neutral";
  }

  function timelineLabel(event, step) {
    return (
      "Step " +
      step +
      ": " +
      (event.failure || (event.agentState === "done" ? "success" : event.agentState))
    );
  }

  function currentReplayIndex() {
    if (state.replayIndex === null) {
      return state.trace.length;
    }
    return clampReplayIndex(state.replayIndex);
  }

  function snapshotForReplayIndex(index) {
    if (index <= 0) {
      return state.config.initial;
    }
    return state.trace[index - 1].snapshot;
  }

  function clampReplayIndex(value) {
    const parsed = Number(value);
    if (!Number.isFinite(parsed)) {
      return state.trace.length;
    }
    return Math.max(0, Math.min(state.trace.length, Math.round(parsed)));
  }

  function replayLabel(index) {
    if (index === 0) {
      return "initial";
    }
    if (index === state.trace.length && timer && state.index < state.config.steps.length) {
      return "live";
    }
    if (index === state.trace.length) {
      return "latest";
    }
    return "step " + index;
  }

  function renderScene(snapshot) {
    elements.scene.textContent = "";
    if (snapshot.type === "household") {
      renderHousehold(snapshot);
    } else {
      renderTabletop(snapshot);
    }
  }

  function renderTabletop(snapshot) {
    const svg = createSvg("svg", {
      class: "tabletop-svg",
      viewBox: "0 0 100 100",
      "aria-label": "Clarifying question tabletop",
    });
    svg.appendChild(createSvg("rect", { x: 4, y: 4, width: 92, height: 92, rx: 2, fill: "#fbfaf7" }));
    for (let i = 10; i < 100; i += 10) {
      svg.appendChild(createSvg("line", { x1: i, y1: 5, x2: i, y2: 95, class: "tabletop-grid" }));
      svg.appendChild(createSvg("line", { x1: 5, y1: i, x2: 95, y2: i, class: "tabletop-grid" }));
    }

    drawBlock(svg, "red", 32, 56, snapshot.picked === "red");
    drawBlock(svg, "blue", 68, 56, snapshot.picked === "blue");

    if (snapshot.focus && !snapshot.picked) {
      const x = snapshot.focus === "blue" ? 68 : 32;
      svg.appendChild(
        createSvg("circle", {
          cx: x,
          cy: 56,
          r: 11,
          fill: "none",
          stroke: "#47743a",
          "stroke-width": 2,
        })
      );
    }

    if (snapshot.pickAt) {
      svg.appendChild(
        createSvg("path", {
          d:
            "M " +
            (snapshot.pickAt[0] - 5) +
            " " +
            snapshot.pickAt[1] +
            " L " +
            (snapshot.pickAt[0] + 5) +
            " " +
            snapshot.pickAt[1] +
            " M " +
            snapshot.pickAt[0] +
            " " +
            (snapshot.pickAt[1] - 5) +
            " L " +
            snapshot.pickAt[0] +
            " " +
            (snapshot.pickAt[1] + 5),
          stroke: "#17201f",
          "stroke-width": 1.8,
          "stroke-linecap": "round",
        })
      );
    }

    const caption = createSvg("text", { x: 7, y: 11, class: "svg-small" });
    caption.textContent =
      (snapshot.question ? "Q: " + snapshot.question + "  " : "") +
      (snapshot.answer ? "A: " + snapshot.answer : "command: pick the block");
    svg.appendChild(caption);
    elements.scene.appendChild(svg);
  }

  function drawBlock(svg, color, x, y, hidden) {
    if (hidden) {
      return;
    }
    const fill = color === "red" ? "#d94b3d" : "#2b6cb0";
    svg.appendChild(createSvg("circle", { cx: x, cy: y, r: 6.2, fill }));
    const label = createSvg("text", { x, y: y + 15, class: "svg-label" });
    label.textContent = color + " block";
    svg.appendChild(label);
  }

  function renderHousehold(snapshot) {
    const wrap = document.createElement("div");
    wrap.className = "house-wrap";
    const grid = document.createElement("div");
    grid.className = "house-grid";
    grid.style.setProperty("--cols", "12");
    grid.style.setProperty("--rows", "9");

    const sets = {
      safe: new Set(unsafeCells.map(key)),
      human: new Set(humanCells.map(key)),
      blocked: new Set(snapshot.blocked.map(key)),
      corrected: new Set(snapshot.corrected.map(key)),
      planned: new Set(snapshot.path.map(key)),
      trail: new Set(snapshot.trail.map(key)),
    };
    const objects = new Map();
    if (!snapshot.picked.includes("red") && snapshot.stored !== "red") {
      objects.set("3,3", { className: "red-block", label: "R" });
    }
    if (!snapshot.picked.includes("blue") && snapshot.stored !== "blue") {
      objects.set("5,9", { className: "blue-block", label: "B" });
    }
    objects.set("1,10", { className: "storage", label: "S" });

    for (let row = 0; row < map.length; row += 1) {
      for (let col = 0; col < map[row].length; col += 1) {
        const cell = document.createElement("div");
        const id = row + "," + col;
        cell.className = "cell";
        if (map[row][col] === "#") cell.classList.add("wall");
        if (sets.safe.has(id)) cell.classList.add("safe");
        if (sets.human.has(id)) cell.classList.add("human");
        if (sets.trail.has(id)) cell.classList.add("trail");
        if (sets.planned.has(id)) cell.classList.add("planned");
        if (sets.blocked.has(id)) cell.classList.add("blocked");
        if (sets.corrected.has(id)) cell.classList.add("corrected");
        const object = objects.get(id);
        if (object) {
          cell.classList.add(object.className);
          cell.textContent = object.label;
        }
        if (sameCell(snapshot.robot, [row, col])) {
          cell.className = "cell robot";
          cell.textContent = snapshot.held ? "R+" : "R";
          if (snapshot.held) cell.classList.add("holding");
        }
        grid.appendChild(cell);
      }
    }

    const legend = document.createElement("div");
    legend.className = "house-legend";
    [
      ["#f4ce82", "unsafe"],
      ["#d9c8f0", "human zone"],
      ["#805ad5", "plan"],
      ["#3182ce", "trail"],
    ].forEach(([color, label]) => {
      const item = document.createElement("span");
      const swatch = document.createElement("span");
      swatch.className = "legend-swatch";
      swatch.style.background = color;
      item.appendChild(swatch);
      item.appendChild(document.createTextNode(label));
      legend.appendChild(item);
    });
    wrap.appendChild(grid);
    wrap.appendChild(legend);
    elements.scene.appendChild(wrap);
  }

  function renderMiniMap(plan) {
    const grid = document.createElement("div");
    grid.className = "mini-grid";
    grid.style.setProperty("--cols", "12");
    grid.style.setProperty("--rows", "9");

    const route = new Set(plan.route.map(key));
    const unsafe = new Set(unsafeCells.map(key));
    const human = new Set(humanCells.map(key));
    const stopKey = plan.stop ? key(plan.stop) : "";
    const targetKey = key(plan.target);
    const storageKey = key(plan.storage);

    for (let row = 0; row < map.length; row += 1) {
      for (let col = 0; col < map[row].length; col += 1) {
        const cell = document.createElement("div");
        const id = row + "," + col;
        cell.className = "mini-cell";
        if (map[row][col] === "#") cell.classList.add("mini-wall");
        if (unsafe.has(id)) cell.classList.add("mini-unsafe");
        if (human.has(id)) cell.classList.add("mini-human");
        if (route.has(id)) cell.classList.add("mini-path");
        if (sameCell([row, col], [7, 1])) {
          setMiniMarker(cell, "mini-start", "R");
        }
        if (id === targetKey) {
          setMiniMarker(cell, "mini-target", plan.targetLabel);
        }
        if (id === storageKey) {
          setMiniMarker(cell, "mini-storage", "S");
        }
        if (id === stopKey) {
          setMiniMarker(cell, "mini-stop", "!");
        }
        grid.appendChild(cell);
      }
    }
    return grid;
  }

  function setMiniMarker(cell, className, label) {
    cell.classList.add(className);
    cell.textContent = label;
  }

  function renderTrace(replayIndex) {
    elements.traceRows.textContent = "";
    const rows = filteredTrace();
    if (!state.trace.length) {
      const empty = document.createElement("div");
      empty.className = "empty-trace";
      empty.textContent = "Trace rows appear after the first step.";
      elements.traceRows.appendChild(empty);
      return;
    }
    if (!rows.length) {
      const empty = document.createElement("div");
      empty.className = "empty-trace";
      empty.textContent = "No trace rows match the selected failure filter.";
      elements.traceRows.appendChild(empty);
      return;
    }
    rows.forEach(({ event, index }) => {
      const row = document.createElement("div");
      row.className = "trace-row";
      row.setAttribute("role", "row");
      row.setAttribute("tabindex", "0");
      row.setAttribute("aria-label", "Show step " + (index + 1));
      if (index + 1 === replayIndex) {
        row.classList.add("trace-active");
        row.setAttribute("aria-current", "step");
      }
      row.addEventListener("click", () => setReplayIndex(index + 1));
      row.addEventListener("keydown", (event) => {
        if (event.key === "Enter" || event.key === " ") {
          event.preventDefault();
          setReplayIndex(index + 1);
        }
      });
      [
        String(index + 1),
        event.action,
        formatReward(event.reward),
        event.failure || (event.agentState === "done" ? "success" : "-"),
        event.agentState,
      ].forEach((value, column) => {
        const span = document.createElement("span");
        span.textContent = value;
        if (column === 3 && event.failure) span.className = "trace-failure";
        if (column === 3 && value === "success") span.className = "trace-success";
        row.appendChild(span);
      });
      elements.traceRows.appendChild(row);
    });
  }

  function setReplayIndex(index) {
    state.replayIndex = clampReplayIndex(index);
    render();
  }

  function filteredTrace() {
    const selected = elements.failureFilter.value;
    return state.trace
      .map((event, index) => ({ event, index }))
      .filter(({ event }) => selected === "all" || event.failure === selected);
  }

  function applyInitialParams() {
    const params = new URLSearchParams(window.location.search);
    const scenario = params.get("scenario");
    const answer = params.get("answer");
    const failure = params.get("failure");
    if (scenarioValues.has(scenario)) {
      elements.scenario.value = scenario;
    }
    if (answerValues.has(answer)) {
      elements.answer.value = answer;
    }
    if (failureValues.has(failure)) {
      elements.failureFilter.value = failure;
    }
    if (["1", "true", "yes"].includes(String(params.get("compare")).toLowerCase())) {
      elements.compare.checked = true;
    }
  }

  function readAutoplayParam() {
    const raw = new URLSearchParams(window.location.search).get("autoplay");
    return ["1", "true", "yes"].includes(String(raw).toLowerCase());
  }

  function updateLocation(includeAutoplay) {
    if (!window.history || !window.history.replaceState) {
      return;
    }
    window.history.replaceState(null, "", getShareUrl(includeAutoplay));
  }

  function getShareUrl(includeAutoplay) {
    const url = new URL(window.location.href);
    url.searchParams.set("scenario", elements.scenario.value);
    url.searchParams.set("answer", elements.answer.value);
    if (elements.failureFilter.value === "all") {
      url.searchParams.delete("failure");
    } else {
      url.searchParams.set("failure", elements.failureFilter.value);
    }
    if (elements.compare.checked && elements.scenario.value === "household") {
      url.searchParams.set("compare", "1");
    } else {
      url.searchParams.delete("compare");
    }
    if (includeAutoplay) {
      url.searchParams.set("autoplay", "1");
    } else {
      url.searchParams.delete("autoplay");
    }
    return url.toString();
  }

  function formatTraceText() {
    const rows = filteredTrace();
    const filter = elements.failureFilter.value;
    const lines = [
      "PythonInteractiveRobotics live trace",
      "scenario=" + state.scenario,
      "answer=" + state.answer,
      "command=" + state.config.command,
      "compare=" + (elements.compare.checked && state.scenario === "household" ? "on" : "off"),
      "filter=" + filter,
      "",
      "#\taction\treward\tfailure\tagent_state",
    ];
    rows.forEach(({ event, index }) => {
      lines.push(
        [
          index + 1,
          event.action,
          formatReward(event.reward),
          event.failure || (event.agentState === "done" ? "success" : "-"),
          event.agentState,
        ].join("\t")
      );
    });
    if (!rows.length) {
      lines.push("(no matching rows)");
    }
    return lines.join("\n");
  }

  function copyText(text, message) {
    if (navigator.clipboard && window.isSecureContext) {
      navigator.clipboard
        .writeText(text)
        .then(() => showCopyStatus(message))
        .catch(() => fallbackCopyText(text, message));
      return;
    }
    fallbackCopyText(text, message);
  }

  function fallbackCopyText(text, message) {
    const textarea = document.createElement("textarea");
    textarea.value = text;
    textarea.setAttribute("readonly", "");
    textarea.style.position = "fixed";
    textarea.style.left = "-9999px";
    document.body.appendChild(textarea);
    textarea.select();
    document.execCommand("copy");
    document.body.removeChild(textarea);
    showCopyStatus(message);
  }

  function showCopyStatus(message) {
    elements.copyStatus.textContent = message;
    if (copyStatusTimer) {
      window.clearTimeout(copyStatusTimer);
    }
    copyStatusTimer = window.setTimeout(() => {
      elements.copyStatus.textContent = "";
      copyStatusTimer = null;
    }, 2200);
  }

  function createSvg(tag, attributes) {
    const element = document.createElementNS("http://www.w3.org/2000/svg", tag);
    Object.entries(attributes || {}).forEach(([name, value]) => {
      element.setAttribute(name, String(value));
    });
    return element;
  }

  function key(cell) {
    return cell[0] + "," + cell[1];
  }

  function sameCell(a, b) {
    return a[0] === b[0] && a[1] === b[1];
  }

  function formatReward(value) {
    return Number(value).toFixed(2);
  }

  function formatBits(value) {
    return Number(value).toFixed(2) + " bit";
  }
})();
