import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import {
  DEFAULT_WALK_FREQUENCY_HZ,
  GROUND_ESCAPE_DURATION_SECONDS,
  LocomotionController,
  MAX_GAIT_AMPLITUDE,
  groundEscapeCommand,
  mixDescendingCommands,
} from "../locomotion-controller.mjs";

const meta = JSON.parse(await readFile(new URL("../assets/model_meta.json", import.meta.url), "utf8"));
const controller = new LocomotionController(meta);
const ctrl = new Float64Array(meta.nu);

assert.deepEqual(
  Array.from(controller.freqs0),
  Array(6).fill(DEFAULT_WALK_FREQUENCY_HZ),
  "the browser controller should use FlyGym's 12 Hz walking frequency",
);
assert.deepEqual(
  Array.from(controller.phases),
  [Math.PI, 0, Math.PI, 0, Math.PI, 0],
  "reset should begin in a deterministic tripod phase relationship",
);

controller.holdNeutral(ctrl);
for (let leg = 0; leg < 6; leg++) {
  const neutral = meta.preprogrammed.legs[meta.control.leg_order[leg]].neutral;
  for (let dof = 0; dof < 7; dof++) {
    const ctrlIndex = meta.ctrl_index_by_leg_dof[leg][dof];
    assert.ok(
      Math.abs(ctrl[ctrlIndex] - neutral[dof]) < 1e-9,
      `leg ${leg} neutral DoF ${dof} should use the recorded default pose`,
    );
  }
}
assert.deepEqual(Array.from(ctrl.slice(42)), Array(6).fill(1), "all feet should adhere during warmup");

assert.deepEqual(mixDescendingCommands(0.02, -0.02), { left: 0, right: 0 });
const ordinaryTurn = mixDescendingCommands(0.8, 0.8);
assert.ok(ordinaryTurn.left > 0 && ordinaryTurn.right > 0, "ordinary forward turns should not reverse one side");
assert.ok(
  Math.max(Math.abs(ordinaryTurn.left), Math.abs(ordinaryTurn.right)) <= MAX_GAIT_AMPLITUDE,
  "recorded gait trajectories must not be extrapolated past unit amplitude",
);

const retreat = groundEscapeCommand(GROUND_ESCAPE_DURATION_SECONDS, 1);
assert.equal(retreat.phase, "retreat");
assert.ok(retreat.left < 0 && retreat.right < 0, "escape should begin by moving away from a frontal threat");
const pivot = groundEscapeCommand(GROUND_ESCAPE_DURATION_SECONDS - 0.3, 1);
const mirroredPivot = groundEscapeCommand(GROUND_ESCAPE_DURATION_SECONDS - 0.3, -1);
assert.equal(pivot.phase, "turn");
assert.equal(pivot.left, -pivot.right, "escape turn should pivot instead of tracing an ordinary arc");
assert.equal(pivot.left, mirroredPivot.right, "the escape turn should mirror the threat side");
const sprint = groundEscapeCommand(0.2, 1);
assert.equal(sprint.phase, "sprint");
assert.deepEqual([sprint.left, sprint.right], [1, 1]);
assert.ok(sprint.frequencyScale > 1, "escape sprint should be faster than exploratory walking");

controller.reset();
controller.stepCPG(ctrl, 1, 1);
assert.ok(controller.smoothedGains[0] < 0.01, "descending commands should ramp in gradually");
for (let i = 1; i < 1000; i++) controller.stepCPG(ctrl, 1, 1);
assert.ok(controller.smoothedGains[0] > 0.7 && controller.smoothedGains[0] < 0.8);
assert.ok(controller.mags.every((magnitude) => magnitude > 0.35 && magnitude < 0.55));
assert.ok(ctrl.every(Number.isFinite), "all actuator targets should remain finite");
assert.ok(ctrl.slice(42).every((value) => value === 0 || value === 1));

controller.stepCPG(ctrl, sprint.left, sprint.right, sprint);
assert.ok(
  controller._cpgFreqs.every((frequency) => Math.abs(frequency) <= DEFAULT_WALK_FREQUENCY_HZ * 1.35),
  "escape frequency should stay within the controller safety bound",
);

const circularDistance = (a, b) => Math.abs(Math.atan2(Math.sin(a - b), Math.cos(a - b)));
assert.ok(circularDistance(controller.phases[0], controller.phases[2]) < 1e-6);
assert.ok(Math.abs(circularDistance(controller.phases[0], controller.phases[1]) - Math.PI) < 1e-6);

console.log("locomotion controller: stable tripod gait, neutral warmup, smoothing, and ground escape verified");
