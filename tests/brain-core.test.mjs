import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import { fileURLToPath } from "node:url";
import path from "node:path";
import { performance } from "node:perf_hooks";

import { BrainEngine } from "../brain-core.mjs";

const here = path.dirname(fileURLToPath(import.meta.url));
const dataDir = path.resolve(here, "../data");

async function loadArray(file, Constructor) {
  const bytes = await readFile(path.join(dataDir, file));
  return new Constructor(bytes.buffer, bytes.byteOffset, bytes.byteLength / Constructor.BYTES_PER_ELEMENT);
}

const manifest = JSON.parse(await readFile(path.join(dataDir, "manifest.json"), "utf8"));
assert.equal(manifest.neuron_count, 138639);
assert.equal(manifest.edge_count, 15091983);
assert.equal(manifest.synapse_count, 54492922);

const [offsets, targets, weights, groups] = await Promise.all([
  loadArray(manifest.arrays.offsets.file, Uint32Array),
  loadArray(manifest.arrays.targets.file, Uint32Array),
  loadArray(manifest.arrays.weights.file, Int16Array),
  loadArray(manifest.arrays.groups.file, Uint8Array),
]);

assert.equal(offsets.length, manifest.neuron_count + 1);
assert.equal(offsets.at(-1), manifest.edge_count);
assert.equal(targets.length, manifest.edge_count);
assert.equal(weights.length, manifest.edge_count);
assert.equal(groups.length, manifest.neuron_count);

const engine = new BrainEngine(manifest, { offsets, targets, weights, groups });
const started = performance.now();
let frame;
let activeFrames = 0;
for (let i = 0; i < 8; i++) {
  frame = engine.step(15, {
    hungerHz: 55,
    odorLeftHz: 90,
    odorRightHz: 35,
    touchHz: 0,
    loomHz: 0,
    sugarLeftHz: 0,
    sugarRightHz: 0,
  });
  if (frame.spikeIndices.length) activeFrames++;
}
const elapsed = performance.now() - started;

assert.equal(Math.round(frame.brainTimeMs), 120);
assert.ok(activeFrames >= 6, "expected repeatable sensory activity");
assert.ok(frame.totalSpikes > 0, "expected spikes in the full graph");
assert.ok(frame.motor.forward > 0, "P9 drive should produce a forward readout");
assert.ok(Number.isFinite(frame.computeMs));

let threatFrame;
for (let i = 0; i < 12; i++) {
  threatFrame = engine.step(15, {
    hungerHz: 55,
    odorLeftHz: 90,
    odorRightHz: 35,
    touchHz: 0,
    loomHz: 220,
    sugarLeftHz: 0,
    sugarRightHz: 0,
  });
}
assert.ok(threatFrame.motor.escape > 0.8, "looming input should activate the giant-fiber escape readout");

console.log(JSON.stringify({
  neurons: manifest.neuron_count,
  edges: manifest.edge_count,
  simulatedMs: frame.brainTimeMs,
  wallMs: Number(elapsed.toFixed(1)),
  lastFrameComputeMs: Number(frame.computeMs.toFixed(1)),
  totalSpikes: frame.totalSpikes,
  activeLastFrame: frame.spikeIndices.length,
  motor: frame.motor,
  threatEscape: threatFrame.motor.escape,
}, null, 2));
