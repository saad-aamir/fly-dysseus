# Data, Model, and License Provenance

## Connectome

- Dataset: FlyWire FAFB v783, 138,639 neurons.
- Connectivity and completeness files: [Eon fly-brain](https://github.com/eonsystemspbc/fly-brain), GPL-2.0-or-later for the repository except where noted upstream.
- Cell classes, cell types, side labels, and representative coordinates: [FlyWire Codex downloads](https://codex.flywire.ai/api/download?dataset=fafb). Consult the FlyWire/Codex terms for reuse of these scientific data.
- LIF constants and the reference sugar-sensory population: [Drosophila_brain_model](https://github.com/philshiu/Drosophila_brain_model), MIT as recorded by the upstream Eon repository.
- The exact source hashes, generated-array hashes, neural populations, and numerical constants are embedded in `data/manifest.json`.

## Body and Runtime

- Body, CPG parameters, and preprogrammed leg trajectories: [FlyGym 2.1](https://github.com/NeLy-EPFL/flygym), Apache-2.0.
- Physics runtime: [MuJoCo](https://github.com/google-deepmind/mujoco), Apache-2.0. The browser WASM bundle follows the official FlyGym 2.1 browser example.
- Rendering runtime: [Three.js 0.169.0](https://github.com/mrdoob/three.js), MIT.

## Transformations

`tools/build_connectome.py` joins public annotations by FlyWire root ID, converts signed synapse counts into a source-sorted CSR graph, and writes little-endian typed arrays. Positions are clipped to the 1st-99th percentile bounding box and axis-reordered only for display. The simulation graph itself is not pruned. Visual line geometry contains the 100,000 strongest connections by absolute signed weight.

No claim is made that the hand-designed sensor and descending-neuron interfaces reproduce every biological pathway. See `README.md` for the current fidelity boundary.
