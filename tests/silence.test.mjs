import { BrainEngine } from "../brain-core.mjs";
import assert from "node:assert/strict";

// Toy brain: 0 -> 1 -> 2 (strong excitatory chain), plus 1 -> 3 (weak).
const manifest = {
  neuron_count: 4, edge_count: 3,
  group_names: ["sensory", "central", "motor"],
  lif: { dt_ms: 0.1, v_rest_mv: -52, v_reset_mv: -52, v_threshold_mv: -45, tau_mem_ms: 20,
         tau_syn_ms: 5, refractory_ms: 2.2, delay_ms: 1.8, weight_per_synapse_mv: 0.275, poisson_weight_scale: 250 },
  populations: { sugar_shiu_reference: [0], feed_mn9: [2] },
};
const arrays = {
  offsets: Uint32Array.from([0, 1, 3, 3, 3]),   // sender-sorted CSR
  targets: Uint32Array.from([1, 2, 3]),
  weights: Int16Array.from([400, 400, 5]),
  groups: Uint8Array.from([0, 1, 2, 1]),
};
const original = Int16Array.from(arrays.weights);
const brain = new BrainEngine(manifest, arrays);

const run = () => {
  brain.reset();
  const r = brain.step(500, { sugarReferenceHz: 100 });
  const fired = new Set(r.spikeIndices);
  return { sugar: fired.has(0), relay: fired.has(1), mn9: fired.has(2), feed: r.motor.feed };
};

const normal = run();
console.log("normal   ", normal);
assert.ok(normal.sugar && normal.relay && normal.mn9, "signal should reach MN9");

const count = brain.setSilenced(["sugar_shiu_reference"]);
assert.equal(count, 1);
assert.equal(brain.weights[0], 0, "sugar neuron's outgoing weight zeroed");
assert.equal(brain.weights[1], 400, "other neurons untouched");
const wax = run();
console.log("wax      ", wax);
assert.ok(wax.sugar, "silenced neuron still fires (tetanus-style)");
assert.ok(!wax.relay && !wax.mn9, "nothing downstream hears it");

brain.setSilenced([]);
assert.deepEqual(Array.from(brain.weights), Array.from(original), "weights fully restored");
const restored = run();
console.log("restored ", restored);
assert.ok(restored.mn9, "signal reaches MN9 again");

brain.setSilenced(["sugar_shiu_reference"]);
brain.setSilenced(["sugar_shiu_reference"]);   // toggling twice must not lose the originals
brain.setSilenced([]);
assert.deepEqual(Array.from(brain.weights), Array.from(original), "double silence still restores");

assert.throws(() => brain.setSilenced(["no_such_population"]));
console.log("ALL TESTS PASSED");
