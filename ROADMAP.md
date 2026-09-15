# Embodied Fly Lab Roadmap

The current release is an interactive research prototype, not a biologically validated digital animal. This roadmap prioritizes changes that improve scientific usefulness before adding visual spectacle.

## Current release

- Simulates all 138,639 listed neurons and 15,091,983 weighted FlyWire v783 edges in a Web Worker.
- Couples connectome output to a FlyGym 2.1 body running in MuJoCo WASM.
- Uses a deterministic 12 Hz tripod CPG with recorded joint trajectories, phase-matched adhesion, smoothed commands, and a neutral-pose reset warmup.
- Displays the position and recent firing state of every neuron alongside the embodied world.

## Next priorities

### 1. Mechanosensory closed loop

Feed per-leg contact, joint angle, joint velocity, and force signals back into identified ascending and sensory populations. Use those signals for stance detection and limited CPG phase correction on uneven terrain.

**Validation:** perturb one leg or add a low obstacle, then compare recovery time, duty factor, and inter-leg phase against an open-loop baseline.

### 2. Calibrated sensory transduction

Replace scalar odor and looming proxies with a spatial odor plume, bilateral antenna sampling, and compound-eye sampling. Keep raw environmental signals separate from neuron-specific transduction so each stage can be tested independently.

**Validation:** measure orientation and turning curves across controlled plume and looming conditions.

### 3. Descending-neuron motor mapping

Replace hand-tuned population mixes with mappings supported by published perturbation data. Calibrate forward speed, turning curvature, stopping, retreat, and escape against NeuroMechFly reference trajectories.

**Validation:** replay fixed neural drives and report trajectory error, speed, yaw rate, and gait stability.

### 4. Reproducible experiments

Add deterministic seeds, scenario files, event recording, replay, and export of spikes, population rates, body pose, contacts, and stimuli. A run should be recoverable from one configuration file.

**Validation:** repeated runs with the same seed must produce byte-identical event logs.

## Later work

- Move graph integration toward WebGPU or partitioned workers and stream compressed graph chunks to reduce startup and brain-step latency.
- Add configurable neuron and synapse models without changing the environment/controller interface.
- Implement dedicated feeding, grooming, escape, flight, and terrain controllers as separable behavioral modules.
- Add experiment comparison views, spike raster inspection, neuron selection, and downloadable analysis bundles.
- Support multiple flies only after single-fly sensorimotor validation and performance targets are met.

## Performance targets

| Target | Goal |
| --- | ---: |
| First interactive frame on broadband | under 10 s |
| Whole-graph brain step | under 100 ms on a current laptop |
| Physics playback | at least 0.25x real time |
| Deterministic replay drift | zero logged-event differences |

Progress should be judged by measurable behavioral and computational tests, not animation smoothness alone.
