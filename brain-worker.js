import { BrainEngine } from "./brain-core.mjs";

let engine = null;

const constructors = {
  uint8: Uint8Array,
  int16: Int16Array,
  uint32: Uint32Array,
};

async function loadArray(baseUrl, descriptor) {
  const response = await fetch(new URL(descriptor.file, baseUrl));
  if (!response.ok) throw new Error(`failed to load ${descriptor.file}: ${response.status}`);
  const buffer = await response.arrayBuffer();
  const Constructor = constructors[descriptor.dtype];
  if (!Constructor) throw new Error(`unsupported worker dtype ${descriptor.dtype}`);
  const result = new Constructor(buffer);
  if (result.length !== descriptor.length) {
    throw new Error(`${descriptor.file} length mismatch`);
  }
  return result;
}

self.onmessage = async ({ data }) => {
  try {
    if (data.type === "init") {
      const baseUrl = new URL(data.baseUrl, self.location.href);
      self.postMessage({ type: "status", message: "Loading whole-brain CSR connectivity" });
      const manifestResponse = await fetch(new URL("manifest.json", baseUrl));
      if (!manifestResponse.ok) throw new Error("Failed to load manifest.json");
      const manifest = await manifestResponse.json();
      const [offsets, targets, weights, groups] = await Promise.all([
        loadArray(baseUrl, manifest.arrays.offsets),
        loadArray(baseUrl, manifest.arrays.targets),
        loadArray(baseUrl, manifest.arrays.weights),
        loadArray(baseUrl, manifest.arrays.groups),
      ]);
      self.postMessage({ type: "status", message: "Initializing LIF state arrays" });
      engine = new BrainEngine(manifest, { offsets, targets, weights, groups });
      self.postMessage({
        type: "ready",
        neuronCount: manifest.neuron_count,
        edgeCount: manifest.edge_count,
        synapseCount: manifest.synapse_count,
      });
      return;
    }

    if (!engine) throw new Error("brain engine is not initialized");
    if (data.type === "reset") {
      engine.reset(data.seed);
      self.postMessage({ type: "reset-complete" });
      return;
    }
    if (data.type === "step") {
      const result = engine.step(data.durationMs, data.stimuli);
      self.postMessage(
        { type: "frame", requestId: data.requestId, ...result },
        [result.spikeIndices.buffer, result.spikeCounts.buffer],
      );
    }
  } catch (error) {
    self.postMessage({
      type: "error",
      message: error instanceof Error ? error.message : String(error),
      stack: error instanceof Error ? error.stack : "",
    });
  }
};
