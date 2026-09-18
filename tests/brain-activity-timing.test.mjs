import assert from "node:assert/strict";

import {
  BrainActivityTiming,
  DEFAULT_ACTIVITY_INTERVAL_SECONDS,
  activityDecayFactor,
} from "../brain-activity-timing.mjs";

const closeTo = (actual, expected, tolerance = 1e-6) => {
  assert.ok(
    Math.abs(actual - expected) <= tolerance,
    `expected ${actual} to be within ${tolerance} of ${expected}`,
  );
};

function decayForFrames(frameRate) {
  const heat = new Float32Array([255]);
  for (let frame = 0; frame < frameRate; frame++) {
    heat[0] *= activityDecayFactor(1 / frameRate, DEFAULT_ACTIVITY_INTERVAL_SECONDS);
  }
  return heat[0];
}

const heatAt60Hz = decayForFrames(60);
const heatAt120Hz = decayForFrames(120);
const heatAfterOneSecond = 255 * activityDecayFactor(1, DEFAULT_ACTIVITY_INTERVAL_SECONDS);
closeTo(heatAt60Hz, heatAt120Hz, 1e-4);
closeTo(heatAt60Hz, heatAfterOneSecond, 1e-4);

const clock = new BrainActivityTiming(0);
assert.equal(clock.advanceTo(1), 1, "a one-second stall must decay a full second of activity");
assert.equal(clock.advanceTo(0.5), 0, "a non-monotonic timestamp must not move the clock backward");
assert.equal(clock.advanceTo(1.25), 0.25);

const cadence = new BrainActivityTiming(0);
cadence.observeActivity(0);
cadence.setCadenceActive(false); // Pause.
cadence.observeActivity(20); // A pending worker frame arrives while paused.
assert.equal(cadence.lastActivityAt, undefined);
cadence.setCadenceActive(true); // Resume.
cadence.observeActivity(21);
assert.equal(cadence.activityInterval, DEFAULT_ACTIVITY_INTERVAL_SECONDS);
cadence.observeActivity(22);
closeTo(cadence.activityInterval, 0.93);

cadence.advanceTo(25);
cadence.reset(30);
assert.equal(cadence.activityInterval, DEFAULT_ACTIVITY_INTERVAL_SECONDS);
assert.equal(cadence.lastActivityAt, undefined);
assert.equal(cadence.advanceTo(31), 1);

console.log("brain activity timing: wall-time decay and cadence resets verified");
