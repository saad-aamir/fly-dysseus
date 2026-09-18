export const DEFAULT_ACTIVITY_INTERVAL_SECONDS = 0.9;

const MIN_ACTIVITY_INTERVAL_SECONDS = 0.05;
const MAX_ACTIVITY_INTERVAL_SECONDS = 4;
const ACTIVITY_INTERVAL_SMOOTHING = 0.3;
const MIN_ACTIVITY_DECAY_SECONDS = 0.25;
const ACTIVITY_DECAY_INTERVAL_SCALE = 1.15;

const clamp = (value, low, high) => Math.max(low, Math.min(high, value));

export function activityDecayFactor(elapsedSeconds, activityIntervalSeconds) {
  const elapsed = Number.isFinite(elapsedSeconds) ? Math.max(0, elapsedSeconds) : 0;
  const interval = Number.isFinite(activityIntervalSeconds)
    ? activityIntervalSeconds
    : DEFAULT_ACTIVITY_INTERVAL_SECONDS;
  const timeConstant = Math.max(
    MIN_ACTIVITY_DECAY_SECONDS,
    interval * ACTIVITY_DECAY_INTERVAL_SCALE,
  );
  return Math.exp(-elapsed / timeConstant);
}

export class BrainActivityTiming {
  constructor(nowSeconds = 0) {
    this.cadenceActive = true;
    this.reset(nowSeconds);
  }

  advanceTo(nowSeconds) {
    if (!Number.isFinite(nowSeconds) || nowSeconds <= this.lastDecayAt) return 0;
    const elapsed = nowSeconds - this.lastDecayAt;
    this.lastDecayAt = nowSeconds;
    return elapsed;
  }

  observeActivity(arrivedAtSeconds) {
    if (!Number.isFinite(arrivedAtSeconds)) return;
    if (!this.cadenceActive) {
      this.resetCadence();
      return;
    }
    if (this.lastActivityAt !== undefined) {
      const gap = clamp(
        arrivedAtSeconds - this.lastActivityAt,
        MIN_ACTIVITY_INTERVAL_SECONDS,
        MAX_ACTIVITY_INTERVAL_SECONDS,
      );
      this.activityInterval += ACTIVITY_INTERVAL_SMOOTHING * (gap - this.activityInterval);
    }
    this.lastActivityAt = arrivedAtSeconds;
  }

  setCadenceActive(active) {
    const next = Boolean(active);
    if (next === this.cadenceActive) return;
    this.cadenceActive = next;
    this.resetCadence();
  }

  decayFactor(elapsedSeconds) {
    return activityDecayFactor(elapsedSeconds, this.activityInterval);
  }

  resetCadence() {
    this.activityInterval = DEFAULT_ACTIVITY_INTERVAL_SECONDS;
    this.lastActivityAt = undefined;
  }

  reset(nowSeconds = 0) {
    this.resetCadence();
    this.lastDecayAt = Number.isFinite(nowSeconds) ? nowSeconds : 0;
  }
}
