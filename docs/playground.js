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

  const elements = {
    scenario: document.getElementById("scenarioSelect"),
    answer: document.getElementById("answerSelect"),
    reset: document.getElementById("resetButton"),
    step: document.getElementById("stepButton"),
    run: document.getElementById("runButton"),
    scene: document.getElementById("scene"),
    traceRows: document.getElementById("traceRows"),
    stepCounter: document.getElementById("stepCounter"),
    command: document.getElementById("commandValue"),
    target: document.getElementById("targetValue"),
    agentState: document.getElementById("agentStateValue"),
    failure: document.getElementById("failureValue"),
    source: document.getElementById("sourceLink"),
  };

  let state = buildState();
  let timer = null;

  elements.scenario.addEventListener("change", () => {
    stopRun();
    state = buildState();
    render();
  });
  elements.answer.addEventListener("change", () => {
    stopRun();
    state = buildState();
    render();
  });
  elements.reset.addEventListener("click", () => {
    stopRun();
    state = buildState();
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
    timer = window.setInterval(() => {
      if (!stepOnce()) {
        stopRun();
      }
    }, 760);
    elements.run.textContent = "Stop";
  });

  render();

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

  function buildClarifyingScenario(answer) {
    const targetX = answer === "blue" ? 68 : 32;
    const command = "pick the block";
    const initial = {
      type: "tabletop",
      command,
      agentState: "parse_command",
      target: "unresolved",
      failure: "none",
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

    const context = {
      command: "put the block away",
      robot: [7, 1],
      target: "unresolved",
      held: null,
      stored: null,
      blocked: [],
      corrected: [],
      trail: [[7, 1]],
      picked: [],
      failure: "none",
      agentState: "parse_command",
      path: [
        [7, 1],
        [6, 1],
        [5, 1],
        [4, 1],
        [3, 1],
        [3, 2],
        [3, 3],
      ],
    };
    const initial = {
      type: "household",
      command: "put the block away",
      robot: [7, 1],
      target: "unresolved",
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
      path: answer === "blue" ? blueRoute : redRoute,
    });

    move(steps, context, "move(north)", -0.02, [6, 1], "plan_to_pick");
    addStep(steps, context, {
      action: "move(north)",
      reward: -0.16,
      failure: "unsafe_nominal_step",
      agentState: "safety_filter_replan",
      blocked: [
        [5, 1],
        [5, 2],
        [5, 3],
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

  function snapshotFromContext(context) {
    return {
      command: context.command,
      robot: context.robot.slice(),
      target: context.target,
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
    const current =
      state.trace.length > 0
        ? state.trace[state.trace.length - 1].snapshot
        : state.config.initial;
    elements.command.textContent = state.config.command;
    elements.target.textContent = current.target || "unresolved";
    elements.agentState.textContent = current.agentState || "parse_command";
    elements.failure.textContent = current.failure || "none";
    elements.stepCounter.textContent = state.index + " / " + state.config.totalSteps;
    elements.source.href = sourceLinks[state.scenario];
    elements.step.disabled = state.index >= state.config.steps.length;
    elements.run.disabled = state.index >= state.config.steps.length && !timer;

    renderScene(current);
    renderTrace();
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
      safe: new Set(["5,1", "5,2", "5,3"]),
      human: new Set(["1,6", "1,7", "2,6", "2,7"]),
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

  function renderTrace() {
    elements.traceRows.textContent = "";
    if (!state.trace.length) {
      const empty = document.createElement("div");
      empty.className = "empty-trace";
      empty.textContent = "Trace rows appear after the first step.";
      elements.traceRows.appendChild(empty);
      return;
    }
    state.trace.forEach((event, index) => {
      const row = document.createElement("div");
      row.className = "trace-row";
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
})();
