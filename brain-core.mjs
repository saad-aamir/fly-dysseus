const READOUT_NAMES = [
  "forward_odn1",
  "walk_dnp09",
  "turn_left",
  "turn_right",
  "reverse_mdn",
  "groom_adn1",
  "escape_giant_fiber",
  "feed_mn9",
];

const STIMULUS_POPULATIONS = {
  odorLeftHz: "food_odor_left",
  odorRightHz: "food_odor_right",
  sugarLeftHz: "sugar_left",
  sugarRightHz: "sugar_right",
  // Shiu et al.'s reference sugar neurons, so the fly lab and the
  // published model receive the identical taste input.
  sugarReferenceHz: "sugar_shiu_reference",
  touchHz: "antennal_mechanosensory",
  loomHz: "looming_visual_proxy",
  hungerHz: "walk_dnp09",
};

const clamp = (value, low, high) => Math.max(low, Math.min(high, value));

export class BrainEngine {
  constructor(manifest, arrays) {
    this.manifest = manifest;
    this.offsets = arrays.offsets;
    this.targets = arrays.targets;
    this.weights = arrays.weights;
    this.groups = arrays.groups;
    this.n = manifest.neuron_count;

    if (this.offsets.length !== this.n + 1) throw new Error("invalid CSR offsets");
    if (this.targets.length !== manifest.edge_count) throw new Error("invalid targets");
    if (this.weights.length !== manifest.edge_count) throw new Error("invalid weights");
    if (this.groups.length !== this.n) throw new Error("invalid group codes");

    const p = manifest.lif;
    this.dt = p.dt_ms;
    this.vRest = p.v_rest_mv;
    this.vReset = p.v_reset_mv;
    this.vThreshold = p.v_threshold_mv;
    this.memFactor = this.dt / p.tau_mem_ms;
    this.synFactor = 1 - this.dt / p.tau_syn_ms;
    this.weightScale = p.weight_per_synapse_mv;
    this.poissonKick = p.weight_per_synapse_mv * p.poisson_weight_scale;
    this.refracSteps = Math.round(p.refractory_ms / this.dt);
    this.delaySteps = Math.round(p.delay_ms / this.dt);

    this.v = new Float32Array(this.n);
    this.g = new Float32Array(this.n);
    this.spikes = new Uint8Array(this.n);
    this.refrac = new Uint8Array(this.n);
    this.zeroRefrac = new Uint8Array(this.n);
    this.readoutMask = new Uint8Array(this.n);

    this.groupSizes = new Uint32Array(manifest.group_names.length);
    for (let i = 0; i < this.groups.length; i++) this.groupSizes[this.groups[i]]++;

    for (const populationName of Object.values(STIMULUS_POPULATIONS)) {
      for (const index of manifest.populations[populationName] || []) {
        this.zeroRefrac[index] = 1;
      }
    }
    READOUT_NAMES.forEach((name, bit) => {
      for (const index of manifest.populations[name] || []) {
        this.readoutMask[index] |= 1 << bit;
      }
    });

    this.delayRing = Array.from({ length: this.delaySteps + 1 }, () => []);
    this.intervalSpikeCounts = new Uint8Array(this.n);
    this.intervalTouched = [];
    this.smoothedReadouts = new Float64Array(READOUT_NAMES.length);
    this.smoothedGroups = new Float64Array(manifest.group_names.length);
    // neuron index -> its original outgoing weights, so silencing can be undone
    this.silencedEdges = new Map();
    this.reset();
  }

  reset(seed = 0x5eed1234) {
    this.v.fill(this.vRest);
    this.g.fill(0);
    this.spikes.fill(0);
    this.refrac.fill(this.refracSteps);
    this.delayRing.forEach((slot) => slot.splice(0));
    this.ringPosition = 0;
    this.intervalSpikeCounts.fill(0);
    this.intervalTouched.length = 0;
    this.smoothedReadouts.fill(0);
    this.smoothedGroups.fill(0);
    this.timeMs = 0;
    this.totalSpikes = 0;
    this.randomState = seed >>> 0;
  }

  // Tetanus-style silencing, the same cut as silence() in Shiu et al.'s
  // model.py: a silenced neuron can still fire, but every synapse FROM it
  // carries nothing. The graph is stored per sender, so a neuron's outgoing
  // synapses are the contiguous block offsets[i] .. offsets[i + 1].
  //
  // Pass the full list of populations that should be silenced right now.
  // Anything silenced before is restored first, so this works as a toggle.
  setSilenced(populationNames = []) {
    for (const [index, original] of this.silencedEdges) {
      this.weights.set(original, this.offsets[index]);
    }
    this.silencedEdges.clear();

    for (const name of populationNames) {
      const indices = this.manifest.populations[name];
      if (!indices) throw new Error(`unknown population ${name}`);
      for (const index of indices) {
        if (this.silencedEdges.has(index)) continue;
        const start = this.offsets[index];
        const end = this.offsets[index + 1];
        this.silencedEdges.set(index, this.weights.slice(start, end));
        this.weights.fill(0, start, end);
      }
    }
    return this.silencedEdges.size;
  }

  random() {
    let x = this.randomState;
    x ^= x << 13;
    x ^= x >>> 17;
    x ^= x << 5;
    this.randomState = x >>> 0;
    return this.randomState / 4294967296;
  }

  applyPoisson(indices, rateHz) {
    if (!indices || !indices.length || rateHz <= 0) return;
    const probability = Math.min(1, rateHz * this.dt / 1000);
    const v = this.v;
    for (let j = 0; j < indices.length; j++) {
      if (this.random() < probability) v[indices[j]] += this.poissonKick;
    }
  }

  step(durationMs = 15, stimuli = {}) {
    const started = performance.now();
    const steps = Math.max(1, Math.round(durationMs / this.dt));
    const readoutCounts = new Uint32Array(READOUT_NAMES.length);
    const groupCounts = new Uint32Array(this.groupSizes.length);

    for (let step = 0; step < steps; step++) {
      for (const [key, populationName] of Object.entries(STIMULUS_POPULATIONS)) {
        this.applyPoisson(this.manifest.populations[populationName], stimuli[key] || 0);
      }
      this.stepOnce(readoutCounts, groupCounts);
    }

    const intervalSeconds = steps * this.dt / 1000;
    const alpha = 1 - Math.exp(-(steps * this.dt) / 60);
    const readoutRates = {};
    READOUT_NAMES.forEach((name, i) => {
      const populationSize = Math.max(1, (this.manifest.populations[name] || []).length);
      const rawRate = readoutCounts[i] / populationSize / intervalSeconds;
      this.smoothedReadouts[i] += alpha * (rawRate - this.smoothedReadouts[i]);
      readoutRates[name] = this.smoothedReadouts[i];
    });

    const groupRates = {};
    this.manifest.group_names.forEach((name, i) => {
      const rawRate = groupCounts[i] / Math.max(1, this.groupSizes[i]) / intervalSeconds;
      this.smoothedGroups[i] += alpha * (rawRate - this.smoothedGroups[i]);
      groupRates[name] = this.smoothedGroups[i];
    });

    const spikeIndices = Uint32Array.from(this.intervalTouched);
    const spikeCounts = new Uint8Array(spikeIndices.length);
    for (let i = 0; i < spikeIndices.length; i++) {
      const index = spikeIndices[i];
      spikeCounts[i] = this.intervalSpikeCounts[index];
      this.intervalSpikeCounts[index] = 0;
    }
    this.intervalTouched.length = 0;

    const forwardHz = Math.max(readoutRates.forward_odn1, readoutRates.walk_dnp09);
    const reverseHz = readoutRates.reverse_mdn;
    const forward = clamp(forwardHz / 45, 0, 1.2);
    const reverse = clamp(reverseHz / 45, 0, 1.0);
    const turn = clamp(
      (readoutRates.turn_right - readoutRates.turn_left) / 35,
      -0.8,
      0.8,
    );

    return {
      brainTimeMs: this.timeMs,
      computeMs: performance.now() - started,
      simulatedMs: steps * this.dt,
      totalSpikes: this.totalSpikes,
      spikeIndices,
      spikeCounts,
      groupRates,
      readoutRates,
      motor: {
        forward,
        reverse,
        turn,
        feed: clamp(readoutRates.feed_mn9 / 35, 0, 1),
        groom: clamp(readoutRates.groom_adn1 / 35, 0, 1),
        escape: clamp(readoutRates.escape_giant_fiber / 35, 0, 1),
      },
    };
  }

  stepOnce(readoutCounts, groupCounts) {
    const n = this.n;
    const v = this.v;
    const g = this.g;
    const spikes = this.spikes;
    const refrac = this.refrac;
    const fired = [];

    for (let i = 0; i < n; i++) {
      if (spikes[i]) refrac[i] = 0;
      else if (refrac[i] < 255) refrac[i]++;

      let voltage = v[i];
      voltage += this.memFactor * (g[i] - (voltage - this.vRest));
      const didFire = voltage > this.vThreshold;
      spikes[i] = didFire ? 1 : 0;
      v[i] = didFire ? this.vReset : voltage;
      g[i] *= this.synFactor;

      if (!didFire) continue;
      fired.push(i);
      const previous = this.intervalSpikeCounts[i];
      if (previous === 0) this.intervalTouched.push(i);
      if (previous < 255) this.intervalSpikeCounts[i] = previous + 1;
      groupCounts[this.groups[i]]++;
      const mask = this.readoutMask[i];
      for (let bit = 0; bit < READOUT_NAMES.length; bit++) {
        if (mask & (1 << bit)) readoutCounts[bit]++;
      }
    }

    const due = this.delayRing[this.ringPosition];
    const offsets = this.offsets;
    const targets = this.targets;
    const weights = this.weights;
    for (let j = 0; j < due.length; j++) {
      const source = due[j];
      for (let edge = offsets[source]; edge < offsets[source + 1]; edge++) {
        const post = targets[edge];
        if (this.zeroRefrac[post] || refrac[post] >= this.refracSteps) {
          g[post] += weights[edge] * this.weightScale;
        }
      }
    }
    due.length = 0;

    for (let j = 0; j < fired.length; j++) g[fired[j]] = 0;
    const scheduled = (this.ringPosition + this.delaySteps) % this.delayRing.length;
    this.delayRing[scheduled] = fired;
    this.ringPosition = (this.ringPosition + 1) % this.delayRing.length;
    this.timeMs += this.dt;
    this.totalSpikes += fired.length;
  }
}

export const brainConstants = { READOUT_NAMES, STIMULUS_POPULATIONS };
