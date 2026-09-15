const TAU = Math.PI * 2;

export const DEFAULT_WALK_FREQUENCY_HZ = 12;
export const MAX_GAIT_AMPLITUDE = 1;
export const GROUND_ESCAPE_DURATION_SECONDS = 0.95;

const COMMAND_TIME_CONSTANT_SECONDS = 0.075;
const COMMAND_DEAD_ZONE = 0.04;
const ADHESION_AMPLITUDE_THRESHOLD = 0.06;

const clamp = (value, low, high) => Math.max(low, Math.min(high, value));

function applyDeadZone(value, deadZone = COMMAND_DEAD_ZONE) {
  const magnitude = Math.abs(value);
  if (magnitude <= deadZone) return 0;
  const scaled = (magnitude - deadZone) / (1 - deadZone);
  return Math.sign(value) * scaled;
}

export function mixDescendingCommands(base, turn) {
  const drive = clamp(applyDeadZone(base), -1, 1);
  const steering = clamp(applyDeadZone(turn) * 0.72, -0.62, 0.62);
  let left = drive + steering;
  let right = drive - steering;

  // During forward or reverse walking, preserve a small amount of stance-side
  // drive. This prevents noisy readouts from flipping one tripod backward.
  if (Math.abs(drive) > 0.12) {
    const sameDirectionFloor = Math.abs(drive) * 0.14;
    if (drive > 0) {
      left = Math.max(left, sameDirectionFloor);
      right = Math.max(right, sameDirectionFloor);
    } else {
      left = Math.min(left, -sameDirectionFloor);
      right = Math.min(right, -sameDirectionFloor);
    }
  }

  return {
    left: clamp(left, -MAX_GAIT_AMPLITUDE, MAX_GAIT_AMPLITUDE),
    right: clamp(right, -MAX_GAIT_AMPLITUDE, MAX_GAIT_AMPLITUDE),
  };
}

export function groundEscapeCommand(remainingSeconds, turnSign = 1) {
  const elapsed = GROUND_ESCAPE_DURATION_SECONDS - clamp(
    remainingSeconds,
    0,
    GROUND_ESCAPE_DURATION_SECONDS,
  );
  const direction = turnSign >= 0 ? 1 : -1;

  if (elapsed < 0.16) {
    return {
      left: -0.86,
      right: -0.86,
      frequencyScale: 1.15,
      responsiveness: 2.6,
      phase: "retreat",
    };
  }
  if (elapsed < 0.43) {
    return {
      left: direction * 0.94,
      right: -direction * 0.94,
      frequencyScale: 1.15,
      responsiveness: 2.6,
      phase: "turn",
    };
  }
  return {
    left: 1,
    right: 1,
    frequencyScale: 1.25,
    responsiveness: 2,
    phase: "sprint",
  };
}

export class LocomotionController {
  constructor(meta, options = {}) {
    this.dt = meta.timestep;
    this.legs = meta.control.leg_order;
    this.cmap = meta.ctrl_index_by_leg_dof;
    this.adh = meta.adhesion;
    this.tripodMap = meta.control.tripod_map;
    const cpg = meta.control.cpg;
    const walkFrequencyHz = options.walkFrequencyHz ?? DEFAULT_WALK_FREQUENCY_HZ;
    this.freqs0 = cpg.intrinsic_freqs.map((frequency) => (
      Math.min(Math.abs(frequency), walkFrequencyHz)
    ));
    this.W = cpg.coupling_weights;
    this.PB = cpg.phase_biases;
    this.conv = cpg.convergence_coefs;
    this.commandTau = options.commandTimeConstantSeconds ?? COMMAND_TIME_CONSTANT_SECONDS;
    this.phaseInc = (this.dt / meta.control.leg_step_time) * TAU;
    const pp = meta.preprogrammed;
    this.N = pp.n_samples;
    this.tab = this.legs.map((leg) => pp.legs[leg]);
    this._cpgAmps = new Float64Array(6);
    this._cpgFreqs = new Float64Array(6);
    this._d6 = new Float64Array(6);
    this._a7 = new Float64Array(7);
    this._groomActions = new Float64Array(6);
    this.smoothedGains = new Float64Array(2);
    this.reset();
  }

  reset() {
    // Start in a stable tripod relationship instead of waiting for randomly
    // initialized oscillators to converge while the body is already falling.
    this.phases = Float64Array.from(
      this.tripodMap,
      (tripod) => (tripod === 0 ? Math.PI : 0),
    );
    this.mags = new Float64Array(6);
    this.legPhases = new Float64Array(6);
    this.stepDir = new Float64Array(6);
    this.smoothedGains.fill(0);
  }

  _anglesInto(legIndex, phase, magnitude, output) {
    const table = this.tab[legIndex];
    const x = (((phase % TAU) + TAU) % TAU) / TAU * this.N;
    const i0 = Math.floor(x) % this.N;
    const i1 = (i0 + 1) % this.N;
    const fraction = x - Math.floor(x);
    const a0 = table.angles[i0];
    const a1 = table.angles[i1];
    for (let dof = 0; dof < 7; dof++) {
      const sample = a0[dof] * (1 - fraction) + a1[dof] * fraction;
      output[dof] = table.neutral[dof] + magnitude * (sample - table.neutral[dof]);
    }
  }

  _adhesionOn(legIndex, phase, magnitude) {
    if (magnitude < ADHESION_AMPLITUDE_THRESHOLD) return true;
    const [start, end] = this.tab[legIndex].swing;
    const wrapped = ((phase % TAU) + TAU) % TAU;
    return !(wrapped > start && wrapped < end);
  }

  _writeLeg(ctrl, legIndex, phase, magnitude, forceAdhesion = false) {
    this._anglesInto(legIndex, phase, magnitude, this._a7);
    const row = this.cmap[legIndex];
    for (let dof = 0; dof < 7; dof++) ctrl[row[dof]] = this._a7[dof];
    ctrl[this.adh[legIndex]] = forceAdhesion || this._adhesionOn(legIndex, phase, magnitude) ? 1 : 0;
  }

  holdNeutral(ctrl) {
    for (let i = 0; i < 6; i++) this._writeLeg(ctrl, i, this.phases[i], 0, true);
  }

  _smoothGain(index, target, responsiveness = 1) {
    const alpha = 1 - Math.exp(-this.dt * responsiveness / this.commandTau);
    this.smoothedGains[index] += alpha * (target - this.smoothedGains[index]);
    return this.smoothedGains[index];
  }

  stepCPG(ctrl, targetLeft, targetRight, options = {}) {
    const responsiveness = clamp(options.responsiveness ?? 1, 0.5, 4);
    const frequencyScale = clamp(options.frequencyScale ?? 1, 0.5, 1.35);
    const gainLeft = this._smoothGain(0, clamp(targetLeft, -1, 1), responsiveness);
    const gainRight = this._smoothGain(1, clamp(targetRight, -1, 1), responsiveness);
    const amps = this._cpgAmps;
    const freqs = this._cpgFreqs;
    const ampLeft = Math.abs(gainLeft);
    const ampRight = Math.abs(gainRight);
    amps[0] = amps[1] = amps[2] = ampLeft;
    amps[3] = amps[4] = amps[5] = ampRight;
    const signLeft = gainLeft >= 0 ? 1 : -1;
    const signRight = gainRight >= 0 ? 1 : -1;
    for (let i = 0; i < 6; i++) {
      freqs[i] = this.freqs0[i] * frequencyScale * (i < 3 ? signLeft : signRight);
    }

    const phases = this.phases;
    const magnitudes = this.mags;
    for (let i = 0; i < 6; i++) {
      let coupling = 0;
      for (let j = 0; j < 6; j++) {
        coupling += magnitudes[j] * this.W[i][j]
          * Math.sin(phases[j] - phases[i] - this.PB[i][j]);
      }
      this._d6[i] = TAU * freqs[i] + coupling;
    }
    for (let i = 0; i < 6; i++) {
      phases[i] = (phases[i] + this._d6[i] * this.dt + TAU) % TAU;
      magnitudes[i] += this.conv[i] * (amps[i] - magnitudes[i]) * this.dt;
      this._writeLeg(ctrl, i, phases[i], magnitudes[i]);
    }
  }

  _advance(phaseArray, directionArray, index, action) {
    if (phaseArray[index] >= TAU || (phaseArray[index] <= 0 && directionArray[index] < 0)) {
      phaseArray[index] = 0;
      directionArray[index] = 0;
    } else if (phaseArray[index] <= 0) {
      if (action > 0) {
        phaseArray[index] += this.phaseInc;
        directionArray[index] = 1;
      } else if (action < 0) {
        phaseArray[index] = TAU - this.phaseInc;
        directionArray[index] = -1;
      }
    } else {
      phaseArray[index] += this.phaseInc * directionArray[index];
    }
  }

  stepSingle(ctrl, actions) {
    for (let i = 0; i < 6; i++) {
      this._advance(this.legPhases, this.stepDir, i, actions[i]);
      const active = this.stepDir[i] !== 0 || Math.abs(actions[i]) > 0;
      this._writeLeg(ctrl, i, this.legPhases[i], active ? 1 : 0, !active);
    }
  }

  stepGroom(ctrl, simTime) {
    this._groomActions.fill(0);
    const phase = Math.floor(simTime * 8) % 2;
    this._groomActions[phase ? 0 : 3] = 1;
    this.stepSingle(ctrl, this._groomActions);
  }
}
