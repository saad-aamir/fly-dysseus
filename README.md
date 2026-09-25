# Fly-dysseus

**Odysseus and the Sirens, staged in a simulated fruit fly brain.**

A whole-brain model of *Drosophila*, wired from the FlyWire connectome (138,639 neurons), drives a physics-simulated fly body. The Sirens are sugar. The fly gets three versions of the myth: the song, wax in the ears, and tied to the mast.

[Video: drop the final cut here]

**Try it live:** [live demo link]

---

## The three acts

| Act | What I changed | Sugar circuit (378 neurons) | MN9 | Feeding command | Body |
|---|---|---|---|---|---|
| **1. The song** | Nothing | ~280 lit | firing | 1.00 | stops and eats until the banana is gone |
| **2. Wax in the ears** | Sugar neurons' outgoing synapses cut | ~60 lit (baseline) | silent | 0.00 | walks past |
| **3. Tied to the mast** | MN9's link to the body cut | ~280 lit | firing | 1.00 | walks straight through the banana |

Before the fly touches sugar, about 63 of the 378 circuit neurons are lit (driven by smell and walking). Act 2 returns the circuit to that baseline. Numbers are from representative runs; the live readout in the video shows them frame by frame.

- **MN9** is the motor neuron behind the fly's feeding mouthparts. In the published model, sugar input reliably drives it.
- **Wax** is a tetanus-toxin-style block: the silenced neurons still fire, but their outgoing synapses carry nothing. The taste neurons light up green in the video and nobody downstream listens.
- **Mast** leaves the brain untouched. MN9 fires, the feeding meter maxes out, and the body is not allowed to act on it. A model of a block between nerve and muscle, which the brain model doesn't simulate anyway.

---

## How it works

Two layers, run on the same neurons:

**1. The show: the Embodied Fly Lab (this repo).** A browser simulation that couples a leaky integrate-and-fire model of the whole FlyWire v783 brain to a NeuroMechFly body in MuJoCo. Touching sugar drives the lab's sugar-sensing neurons; MN9's firing becomes the body's feeding command.

**2. The proof: Shiu et al.'s published model** ([Nature, 2024](https://www.nature.com/articles/s41586-024-07763-9); [code](https://github.com/philshiu/Drosophila_brain_model)), run in [`proof/`](proof/). Same experiments, validated reference implementation, no body.

### What the proof layer shows

**Reproduction (FlyWire v630, the paper's setup).** 21 sugar neurons at 100 Hz, 30 trials of 1 s:

| | Published | Mine |
|---|---|---|
| MN9 firing rate | 67.0 Hz | 66.8 Hz |

**The acts, on the exact input the video uses (v783, the lab's 122 sugar neurons at 200 Hz):**

| | Normal | Wax | Mast |
|---|---|---|---|
| MN9 (left) | 77.0 Hz | 0 | 75.7 Hz |
| MN9 (right) | 106.4 Hz | 0 | 104.3 Hz |
| Average change across sugar-responsive neurons | | 51.1 Hz | 0.2 Hz |

Wax silences the circuit. Mast leaves it untouched: the brain still "wants" the sugar, only the output is cut.

**The Siren circuit.** With the paper's input on v783, 378 neurons respond to sugar (381 on v630, so the circuit is stable across connectome versions). These 378 are the neurons highlighted in the video's Siren circuit view.

---

## What I changed in the Embodied Fly Lab

- **`brain-core.mjs`**: `setSilenced()`, which zeroes a population's outgoing weights and restores them on toggle. Same cut as `silence()` in Shiu et al.'s `model.py`. Tested in `tests/silence.test.mjs`.
- **`app.js`**:
  - **Wax** and **Mast** buttons. Wax silences the stimulated sugar neurons inside the brain; Mast stops the body from acting on the feeding command.
  - A documented `SUGAR_INPUT` switch between the lab's original sugar input and Shiu et al.'s reference neurons.
  - A **Siren circuit** brain view that shows only the circuit neurons (sugar inputs green, circuit gold, MN9 red), with a large live readout: neurons lit, and whether MN9 is firing.
  - Flipped the brain display's vertical axis, so ventral structures sit at the bottom. MN9 now appears in the gnathal ganglion, matching its position in FlyWire Codex.
- **`data/siren-circuit.json`**: the circuit exported from the proof notebook.

---

## Honest limits

- **This is a demonstration, not a discovery.** Silence the taste neurons and the fly doesn't eat. Expected. The point is the method and the verification.
- **The "stand still while eating" rule is hand-written** in the lab. Whether the fly feeds is the brain's call (MN9 above a threshold); that feeding stops the legs is scripted.
- **The urge to walk is injected.** The lab's exploration drive feeds spikes directly into walking command neurons.
- **The lab's sugar input is much stronger than the paper's** (122 neurons at 200 Hz vs 20 at 100 Hz) and recruits a far broader response (9,235 neurons vs 378). With the paper's weaker input, the embodied fly never triggered feeding, even though MN9 fires at 48 to 62 Hz on the same input in the brain-only model. Why the busy embodied brain drowns out a weak taste signal is an open question.
- **Silencing is outgoing-only.** Shiu et al.'s README describes silencing as removing synapses "to and from" a neuron, but the code removes only outgoing ones, as already noted in [issue #10](https://github.com/philshiu/Drosophila_brain_model/issues/10). My experiments follow the code. It's also why MN9 keeps firing in Act 3.
- **The display flip fixes up and down only.** I haven't verified left/right orientation.
- **A connectome is not a brain.** Synapse counts stand in for weights, neurotransmitter signs are predicted, and there's no neuromodulation, plasticity or background activity.

---

## Run it

**The simulation:**

```bash
python3 serve.py
```

Open the address it prints, click **Siren circuit** in the brain panel, then **Add food**. Toggle **Wax** or **Mast** in the toolbar.

**The proof notebook:** clone [Shiu et al.'s repo](https://github.com/philshiu/Drosophila_brain_model) next to this one, then open `proof/odysseus_proof.ipynb`. Each whole-brain run takes a few minutes on a laptop CPU.

---

## Credits

Built entirely on other people's work:

- **Connectome:** [FlyWire](https://flywire.ai) (FAFB v783) and [FlyWire Codex](https://codex.flywire.ai)
- **Brain model:** Shiu et al., *A Drosophila computational brain model reveals sensorimotor processing*, Nature 2024, and [its code](https://github.com/philshiu/Drosophila_brain_model)
- **Connectivity files:** [Eon Systems](https://github.com/eonsystemspbc/fly-brain)
- **Simulation:** the [Embodied Fly Lab](https://github.com/statsleelab/embodied-fly-lab), which this repo forks
- **Body:** [NeuroMechFly / FlyGym](https://github.com/NeLy-EPFL/flygym), [MuJoCo](https://github.com/google-deepmind/mujoco), rendering by [Three.js](https://github.com/mrdoob/three.js)

Licenses and data terms for each are listed in [`PROVENANCE.md`](PROVENANCE.md). The original Embodied Fly Lab README is kept as [`README.upstream.md`](README.upstream.md).

No real flies were harmed. Several virtual bananas were.
