import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import {
  DEFAULT_WALK_FREQUENCY_HZ,
  LocomotionController,
  MAX_GAIT_AMPLITUDE,
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

controller.reset();
controller.stepCPG(ctrl, 1, 1);
assert.ok(controller.smoothedGains[0] < 0.01, "descending commands should ramp in gradually");
for (let i = 1; i < 1000; i++) controller.stepCPG(ctrl, 1, 1);
assert.ok(controller.smoothedGains[0] > 0.7 && controller.smoothedGains[0] < 0.8);
assert.ok(controller.mags.every((magnitude) => magnitude > 0.35 && magnitude < 0.55));
assert.ok(ctrl.every(Number.isFinite), "all actuator targets should remain finite");
assert.ok(ctrl.slice(42).every((value) => value === 0 || value === 1));

const circularDistance = (a, b) => Math.abs(Math.atan2(Math.sin(a - b), Math.cos(a - b)));
assert.ok(circularDistance(controller.phases[0], controller.phases[2]) < 1e-6);
assert.ok(Math.abs(circularDistance(controller.phases[0], controller.phases[1]) - Math.PI) < 1e-6);

console.log("locomotion controller: stable 12 Hz tripod gait, neutral warmup, and smoothed commands verified");
