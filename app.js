import * as THREE from "three";
import {
  loadScene,
  makeStepper,
  makeStatsMeter,
  buildMeshes,
  syncMeshes,
} from "./shared/scene.js";

const TAU = Math.PI * 2;
const DATA_URL = new URL("./data/", import.meta.url);
const ASSETS_URL = "./assets";
const number = new Intl.NumberFormat("en-US");
const byId = (id) => document.getElementById(id);
const clamp = (value, low, high) => Math.max(low, Math.min(high, value));

const GROUP_COLORS = [
  [0.34, 0.39, 0.38],
  [0.38, 0.77, 0.55],
  [0.40, 0.65, 0.80],
  [0.67, 0.71, 0.69],
  [0.91, 0.74, 0.34],
  [0.88, 0.41, 0.37],
  [0.44, 0.79, 0.75],
  [0.72, 0.53, 0.75],
];

const GROUP_LABELS = {
  unknown: "unclassified",
  sensory: "sensory",
  optic: "visual",
  central: "central",
  descending: "descending",
  motor: "motor",
  ascending: "ascending",
  endocrine: "endocrine",
};

const typedArrayConstructors = {
  float32: Float32Array,
  uint8: Uint8Array,
  uint32: Uint32Array,
};

function setStatus(text, kind = "loading") {
  byId("status-text").textContent = text;
  byId("status-dot").className = `status-dot ${kind === "ready" ? "ready" : kind === "error" ? "error" : ""}`;
  byId("loading-message").textContent = text;
}

async function fetchArray(descriptor) {
  const response = await fetch(new URL(descriptor.file, DATA_URL));
  if (!response.ok) throw new Error(`Failed to load ${descriptor.file} (${response.status})`);
  const buffer = await response.arrayBuffer();
  const Constructor = typedArrayConstructors[descriptor.dtype];
  if (!Constructor) throw new Error(`Unsupported array type: ${descriptor.dtype}`);
  const array = new Constructor(buffer);
  if (array.length !== descriptor.length) throw new Error(`${descriptor.file} length mismatch`);
  return array;
}

async function loadBrainVisualData() {
  setStatus("Reading FlyWire whole-brain metadata");
  const response = await fetch(new URL("manifest.json", DATA_URL));
  if (!response.ok) throw new Error("Failed to load manifest.json");
  const manifest = await response.json();
  const [positions, groups, sides, displayEdges] = await Promise.all([
    fetchArray(manifest.arrays.positions),
    fetchArray(manifest.arrays.groups),
    fetchArray(manifest.arrays.sides),
    fetchArray(manifest.arrays.display_edges),
  ]);
  return { manifest, positions, groups, sides, displayEdges };
}

function startBrainWorker() {
  const worker = new Worker("./brain-worker.js", { type: "module" });
  let frameHandler = () => {};
  let rejectReady;
  const ready = new Promise((resolve, reject) => {
    rejectReady = reject;
    worker.onmessage = ({ data }) => {
      if (data.type === "status") setStatus(data.message);
      else if (data.type === "ready") resolve(data);
      else if (data.type === "frame") frameHandler(data);
      else if (data.type === "reset-complete") frameHandler(data);
      else if (data.type === "error") {
        const error = new Error(data.message);
        error.stack = data.stack || error.stack;
        reject(error);
        frameHandler({ type: "error", error });
      }
    };
    worker.onerror = (event) => reject(new Error(event.message));
  });
  worker.postMessage({ type: "init", baseUrl: DATA_URL.href });
  return {
    worker,
    ready,
    onFrame(handler) { frameHandler = handler; },
    fail(error) { rejectReady?.(error); },
  };
}

class Controller {
  constructor(meta) {
    this.dt = meta.timestep;
    this.legs = meta.control.leg_order;
    this.cmap = meta.ctrl_index_by_leg_dof;
    this.adh = meta.adhesion;
    this.tripodMap = meta.control.tripod_map;
    const cpg = meta.control.cpg;
    this.freqs0 = cpg.intrinsic_freqs.slice();
    this.W = cpg.coupling_weights;
    this.PB = cpg.phase_biases;
    this.conv = cpg.convergence_coefs;
    this.phaseInc = (this.dt / meta.control.leg_step_time) * TAU;
    const pp = meta.preprogrammed;
    this.N = pp.n_samples;
    this.tab = this.legs.map((leg) => pp.legs[leg]);
    this._cpgAmps = new Float64Array(6);
    this._cpgFreqs = new Float64Array(6);
    this._d6 = new Float64Array(6);
    this._a7 = new Float64Array(7);
    this._groomActions = new Float64Array(6);
    this.reset();
  }

  reset() {
    this.phases = new Float64Array(6).map(() => Math.random() * TAU);
    this.mags = new Float64Array(6);
    this.legPhases = new Float64Array(6);
    this.stepDir = new Float64Array(6);
    this.tripodPhases = new Float64Array(2);
    this.tripodDir = new Float64Array(2);
  }

  _anglesInto(legIndex, phase, magnitude, output) {
    const table = this.tab[legIndex];
    let x = ((phase % TAU) + TAU) % TAU / TAU * this.N;
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

  _adhesionOn(legIndex, phase) {
    const [start, end] = this.tab[legIndex].swing;
    const wrapped = ((phase % TAU) + TAU) % TAU;
    return !(wrapped > start && wrapped < end);
  }

  _writeLeg(ctrl, legIndex, phase, magnitude) {
    this._anglesInto(legIndex, phase, magnitude, this._a7);
    const row = this.cmap[legIndex];
    for (let dof = 0; dof < 7; dof++) ctrl[row[dof]] = this._a7[dof];
    ctrl[this.adh[legIndex]] = this._adhesionOn(legIndex, phase) ? 1 : 0;
  }

  stepCPG(ctrl, gainLeft, gainRight) {
    const amps = this._cpgAmps;
    const freqs = this._cpgFreqs;
    const ampLeft = Math.abs(gainLeft);
    const ampRight = Math.abs(gainRight);
    amps[0] = amps[1] = amps[2] = ampLeft;
    amps[3] = amps[4] = amps[5] = ampRight;
    const signLeft = gainLeft >= 0 ? 1 : -1;
    const signRight = gainRight >= 0 ? 1 : -1;
    for (let i = 0; i < 6; i++) freqs[i] = this.freqs0[i] * (i < 3 ? signLeft : signRight);

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
      phases[i] += this._d6[i] * this.dt;
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
      this._writeLeg(ctrl, i, this.legPhases[i], 1);
    }
  }

  stepGroom(ctrl, simTime) {
    this._groomActions.fill(0);
    const phase = Math.floor(simTime * 8) % 2;
    this._groomActions[phase ? 0 : 3] = 1;
    this.stepSingle(ctrl, this._groomActions);
  }
}

class BrainRenderer {
  constructor(container, positions, groups, sides, displayEdges, manifest) {
    this.container = container;
    this.positions = positions;
    this.groups = groups;
    this.sides = sides;
    this.manifest = manifest;
    this.scene = new THREE.Scene();
    this.camera = new THREE.PerspectiveCamera(38, 1, 0.1, 100);
    this.camera.position.set(0, 0, 28);
    this.renderer = new THREE.WebGLRenderer({ antialias: true, alpha: false });
    this.renderer.setPixelRatio(Math.min(devicePixelRatio, 2));
    this.renderer.setClearColor(0x111516, 1);
    container.appendChild(this.renderer.domElement);

    this.root = new THREE.Group();
    this.root.rotation.x = -0.14;
    this.scene.add(this.root);
    this.baseColors = new Float32Array(positions.length);
    for (let i = 0; i < groups.length; i++) {
      const color = GROUP_COLORS[groups[i]] || GROUP_COLORS[0];
      this.baseColors.set(color, i * 3);
    }

    const geometry = new THREE.BufferGeometry();
    geometry.setAttribute("position", new THREE.BufferAttribute(positions, 3));
    geometry.setAttribute("color", new THREE.BufferAttribute(this.baseColors, 3));
    const material = new THREE.PointsMaterial({
      size: 0.06,
      vertexColors: true,
      transparent: true,
      opacity: 0.38,
      depthWrite: false,
      sizeAttenuation: true,
    });
    this.points = new THREE.Points(geometry, material);
    this.root.add(this.points);

    const linePositions = new Float32Array(displayEdges.length * 3);
    for (let i = 0; i < displayEdges.length; i++) {
      const neuron = displayEdges[i];
      linePositions[i * 3] = positions[neuron * 3];
      linePositions[i * 3 + 1] = positions[neuron * 3 + 1];
      linePositions[i * 3 + 2] = positions[neuron * 3 + 2];
    }
    const lineGeometry = new THREE.BufferGeometry();
    lineGeometry.setAttribute("position", new THREE.BufferAttribute(linePositions, 3));
    this.lines = new THREE.LineSegments(
      lineGeometry,
      new THREE.LineBasicMaterial({ color: 0x8fa09a, transparent: true, opacity: 0.018 }),
    );
    this.root.add(this.lines);

    const glowCanvas = document.createElement("canvas");
    glowCanvas.width = glowCanvas.height = 64;
    const glowContext = glowCanvas.getContext("2d");
    const glowGradient = glowContext.createRadialGradient(32, 32, 0, 32, 32, 32);
    glowGradient.addColorStop(0, "rgba(255,255,255,1)");
    glowGradient.addColorStop(0.18, "rgba(255,255,255,0.96)");
    glowGradient.addColorStop(0.48, "rgba(255,255,255,0.34)");
    glowGradient.addColorStop(1, "rgba(255,255,255,0)");
    glowContext.fillStyle = glowGradient;
    glowContext.fillRect(0, 0, 64, 64);
    const glowTexture = new THREE.CanvasTexture(glowCanvas);

    this.activityPositions = new Float32Array(positions.length);
    this.activityColors = new Float32Array(positions.length);
    this.activityGeometry = new THREE.BufferGeometry();
    this.activityPositionAttribute = new THREE.BufferAttribute(this.activityPositions, 3);
    this.activityColorAttribute = new THREE.BufferAttribute(this.activityColors, 3);
    this.activityPositionAttribute.setUsage(THREE.DynamicDrawUsage);
    this.activityColorAttribute.setUsage(THREE.DynamicDrawUsage);
    this.activityGeometry.setAttribute("position", this.activityPositionAttribute);
    this.activityGeometry.setAttribute("color", this.activityColorAttribute);
    this.activityGeometry.setDrawRange(0, 0);

    this.activityHalo = new THREE.Points(
      this.activityGeometry,
      new THREE.PointsMaterial({
        size: 0.42,
        map: glowTexture,
        vertexColors: true,
        transparent: true,
        opacity: 0.52,
        blending: THREE.AdditiveBlending,
        depthTest: false,
        depthWrite: false,
        sizeAttenuation: true,
      }),
    );
    this.activityCore = new THREE.Points(
      this.activityGeometry,
      new THREE.PointsMaterial({
        size: 0.17,
        map: glowTexture,
        vertexColors: true,
        transparent: true,
        opacity: 1,
        blending: THREE.AdditiveBlending,
        depthTest: false,
        depthWrite: false,
        sizeAttenuation: true,
      }),
    );
    this.activityHalo.frustumCulled = false;
    this.activityCore.frustumCulled = false;
    this.root.add(this.activityHalo, this.activityCore);

    this.heat = new Uint8Array(groups.length);
    this.hot = [];
    this._bindPointerControls();
    this.resizeObserver = new ResizeObserver(() => this.resize());
    this.resizeObserver.observe(container);
    this.resize();

    this.raster = byId("activity-raster");
    this.rasterContext = this.raster.getContext("2d");
    this.rasterContext.imageSmoothingEnabled = false;
    document.querySelectorAll("[data-brain-mode]").forEach((button) => {
      button.addEventListener("click", () => this.setDisplayMode(button.dataset.brainMode));
    });
  }

  setDisplayMode(mode) {
    const activityOnly = mode === "activity";
    this.points.material.opacity = activityOnly ? 0.045 : 0.38;
    this.lines.material.opacity = activityOnly ? 0 : 0.018;
    document.querySelectorAll("[data-brain-mode]").forEach((button) => {
      button.setAttribute("aria-pressed", String(button.dataset.brainMode === mode));
    });
  }

  _bindPointerControls() {
    let dragging = false;
    let previousX = 0;
    let previousY = 0;
    const canvas = this.renderer.domElement;
    canvas.addEventListener("pointerdown", (event) => {
      dragging = true;
      previousX = event.clientX;
      previousY = event.clientY;
      canvas.setPointerCapture(event.pointerId);
    });
    canvas.addEventListener("pointermove", (event) => {
      if (!dragging) return;
      this.root.rotation.y += (event.clientX - previousX) * 0.006;
      this.root.rotation.x += (event.clientY - previousY) * 0.006;
      this.root.rotation.x = clamp(this.root.rotation.x, -1.2, 1.2);
      previousX = event.clientX;
      previousY = event.clientY;
    });
    canvas.addEventListener("pointerup", () => { dragging = false; });
    canvas.addEventListener("wheel", (event) => {
      event.preventDefault();
      this.camera.position.z = clamp(this.camera.position.z + event.deltaY * 0.012, 17, 44);
    }, { passive: false });
  }

  resize() {
    const width = Math.max(1, this.container.clientWidth);
    const height = Math.max(1, this.container.clientHeight);
    this.renderer.setSize(width, height, false);
    this.camera.aspect = width / height;
    this.camera.updateProjectionMatrix();
  }

  updateActivity(indices, counts, groupRates, simulatedMs) {
    let eventCount = 0;
    const sideEvents = new Uint32Array(4);
    for (let i = 0; i < indices.length; i++) {
      const index = indices[i];
      if (this.heat[index] === 0) this.hot.push(index);
      this.heat[index] = Math.min(255, this.heat[index] + 95 + counts[i] * 28);
      eventCount += counts[i];
      sideEvents[this.sides[index]] += counts[i];
    }
    const rate = eventCount / Math.max(simulatedMs / 1000, 0.001);
    byId("spike-rate").textContent = `${number.format(Math.round(rate))} spikes/s`;
    byId("active-count").textContent = `${number.format(indices.length)} active neurons`;
    const [leadingGroup] = Object.entries(groupRates)
      .filter(([name]) => name !== "unknown")
      .sort((a, b) => b[1] - a[1])[0] || ["unknown", 0];
    const left = sideEvents[1];
    const right = sideEvents[2];
    const sideLabel = left > right * 1.18 ? "left" : right > left * 1.18 ? "right" : "bilateral";
    byId("active-focus").textContent = `${sideLabel} | ${GROUP_LABELS[leadingGroup]} dominant`;
    this.drawRaster(groupRates);
  }

  drawRaster(rates) {
    const ctx = this.rasterContext;
    const width = this.raster.width;
    const height = this.raster.height;
    ctx.drawImage(this.raster, -2, 0);
    ctx.fillStyle = "#0d1011";
    ctx.fillRect(width - 2, 0, 2, height);
    const names = ["sensory", "optic", "central", "descending", "motor", "ascending"];
    const laneHeight = height / names.length;
    names.forEach((name, lane) => {
      const color = GROUP_COLORS[this.manifest.group_names.indexOf(name)];
      const alpha = clamp((rates[name] || 0) / 55, 0.05, 1);
      ctx.fillStyle = `rgba(${color.map((v) => Math.round(v * 255)).join(",")},${alpha})`;
      ctx.fillRect(width - 2, lane * laneHeight + 1, 2, laneHeight - 2);
    });
  }

  resetActivity() {
    this.heat.fill(0);
    this.hot.length = 0;
    this.activityGeometry.setDrawRange(0, 0);
    this.rasterContext.fillStyle = "#0d1011";
    this.rasterContext.fillRect(0, 0, this.raster.width, this.raster.height);
    byId("spike-rate").textContent = "0 spikes/s";
    byId("active-count").textContent = "0 active neurons";
    byId("active-focus").textContent = "Locating activity";
  }

  render() {
    const hadActivity = this.hot.length > 0;
    let write = 0;
    for (let i = 0; i < this.hot.length; i++) {
      const index = this.hot[i];
      const heat = this.heat[index] * 0.92;
      this.heat[index] = heat;
      if (heat < 3) {
        this.heat[index] = 0;
        continue;
      }
      const sourceOffset = index * 3;
      const activityOffset = write * 3;
      this.activityPositions[activityOffset] = this.positions[sourceOffset];
      this.activityPositions[activityOffset + 1] = this.positions[sourceOffset + 1];
      this.activityPositions[activityOffset + 2] = this.positions[sourceOffset + 2];
      const intensity = clamp(heat / 190, 0, 1);
      this.activityColors[activityOffset] = 1;
      this.activityColors[activityOffset + 1] = 0.2 + intensity * 0.72;
      this.activityColors[activityOffset + 2] = 0.035;
      this.hot[write++] = index;
    }
    this.hot.length = write;
    this.activityGeometry.setDrawRange(0, write);
    if (hadActivity) {
      this.activityPositionAttribute.needsUpdate = true;
      this.activityColorAttribute.needsUpdate = true;
    }
    this.activityHalo.material.opacity = 0.46 + Math.sin(performance.now() * 0.009) * 0.08;
    this.renderer.render(this.scene, this.camera);
  }
}

class WorldRenderer {
  constructor(container, mj, model, data, meta) {
    this.container = container;
    this.mj = mj;
    this.model = model;
    this.data = data;
    this.meta = meta;
    this.cameraMode = "follow";
    this.scene = new THREE.Scene();
    this.scene.background = new THREE.Color(0x171c1c);
    this.scene.fog = new THREE.Fog(0x171c1c, 35, 100);
    this.camera = new THREE.PerspectiveCamera(46, 1, 0.04, 240);
    this.camera.up.set(0, 0, 1);
    this.renderer = new THREE.WebGLRenderer({ antialias: true });
    this.renderer.setPixelRatio(Math.min(devicePixelRatio, 2));
    this.renderer.shadowMap.enabled = true;
    container.appendChild(this.renderer.domElement);

    THREE.Object3D.DEFAULT_UP.set(0, 0, 1);
    this.scene.add(new THREE.HemisphereLight(0xdce8e2, 0x364039, 1.4));
    const light = new THREE.DirectionalLight(0xfff3d5, 2.3);
    light.position.set(-10, -14, 24);
    light.castShadow = true;
    this.scene.add(light);

    const ground = new THREE.Mesh(
      new THREE.PlaneGeometry(120, 90),
      new THREE.MeshStandardMaterial({ color: 0x27312d, roughness: 0.98 }),
    );
    ground.position.z = -0.02;
    ground.receiveShadow = true;
    this.scene.add(ground);
    const grid = new THREE.GridHelper(120, 30, 0x52605a, 0x39423f);
    grid.rotation.x = Math.PI / 2;
    grid.position.z = 0.005;
    this.scene.add(grid);

    this.meshGroup = buildMeshes(model, meta);
    for (const item of this.meshGroup.userData.items) {
      item.mesh.castShadow = true;
      item.mesh.receiveShadow = true;
    }
    this.scene.add(this.meshGroup);
    this.bodyId = this.findFlyRootBody();
    this.foods = [];
    this.addFood(15, 7);
    this.addFood(34, -9);
    this.addFood(-16, 12);
    this.createLoomObject();

    this.resizeObserver = new ResizeObserver(() => this.resize());
    this.resizeObserver.observe(container);
    this.resize();
  }

  findFlyRootBody() {
    for (let joint = 0; joint < this.model.njnt; joint++) {
      if (this.model.jnt_type[joint] === 0) return this.model.jnt_bodyid[joint];
    }
    return 1;
  }

  resize() {
    const width = Math.max(1, this.container.clientWidth);
    const height = Math.max(1, this.container.clientHeight);
    this.renderer.setSize(width, height, false);
    this.camera.aspect = width / height;
    this.camera.updateProjectionMatrix();
  }

  makeBanana() {
    const group = new THREE.Group();
    const material = new THREE.MeshStandardMaterial({ color: 0xe8bd57, roughness: 0.72 });
    for (let i = 0; i < 3; i++) {
      const segment = new THREE.Mesh(
        new THREE.TorusGeometry(0.62, 0.14, 8, 22, Math.PI * 1.15),
        material,
      );
      segment.rotation.set(Math.PI / 2, 0.25 * i, -0.5);
      segment.position.set(0.18 * i, 0.16 * i, 0.25 + i * 0.04);
      segment.castShadow = true;
      group.add(segment);
    }
    const odor = new THREE.Mesh(
      new THREE.RingGeometry(1.7, 1.78, 48),
      new THREE.MeshBasicMaterial({ color: 0x62c48d, transparent: true, opacity: 0.25, side: THREE.DoubleSide }),
    );
    odor.position.z = 0.015;
    group.add(odor);
    return group;
  }

  addFood(x, y) {
    const mesh = this.makeBanana();
    mesh.position.set(x, y, 0);
    this.scene.add(mesh);
    this.foods.push({ x, y, energy: 1, mesh });
  }

  addFoodAhead() {
    const pose = this.pose();
    const distance = 9 + Math.random() * 11;
    const lateral = (Math.random() - 0.5) * 15;
    this.addFood(
      pose.x + Math.cos(pose.yaw) * distance - Math.sin(pose.yaw) * lateral,
      pose.y + Math.sin(pose.yaw) * distance + Math.cos(pose.yaw) * lateral,
    );
  }

  createLoomObject() {
    this.loomObject = new THREE.Mesh(
      new THREE.SphereGeometry(1.1, 20, 12),
      new THREE.MeshStandardMaterial({ color: 0x2b3030, roughness: 0.9 }),
    );
    this.loomObject.visible = false;
    this.scene.add(this.loomObject);
  }

  updateLoom(timer) {
    if (timer <= 0) {
      this.loomObject.visible = false;
      return;
    }
    const pose = this.pose();
    this.loomObject.visible = true;
    const progress = 1 - timer / 0.8;
    const distance = 11 - progress * 8;
    this.loomObject.position.set(
      pose.x + Math.cos(pose.yaw) * distance,
      pose.y + Math.sin(pose.yaw) * distance,
      5 - progress * 2,
    );
    this.loomObject.scale.setScalar(0.6 + progress * 2.2);
  }

  pose() {
    const body = this.bodyId;
    const matrix = this.data.xmat;
    return {
      x: this.data.xpos[body * 3],
      y: this.data.xpos[body * 3 + 1],
      z: this.data.xpos[body * 3 + 2],
      yaw: Math.atan2(matrix[body * 9 + 3], matrix[body * 9]),
    };
  }

  updateCamera() {
    const pose = this.pose();
    if (this.cameraMode === "top") {
      this.camera.position.set(pose.x, pose.y - 0.01, 23);
      this.camera.up.set(0, 1, 0);
      this.camera.lookAt(pose.x, pose.y, 0);
      return;
    }
    this.camera.up.set(0, 0, 1);
    this.camera.position.set(
      pose.x - Math.cos(pose.yaw) * 8,
      pose.y - Math.sin(pose.yaw) * 8,
      5.2,
    );
    this.camera.lookAt(
      pose.x + Math.cos(pose.yaw) * 2.1,
      pose.y + Math.sin(pose.yaw) * 2.1,
      0.65,
    );
  }

  render() {
    syncMeshes(this.meshGroup, this.data);
    this.updateCamera();
    this.renderer.render(this.scene, this.camera);
  }
}

class EmbodiedFlyLab {
  constructor(body, brainData, brainConnection) {
    this.mj = body.mj;
    this.model = body.model;
    this.data = body.data;
    this.bodyMeta = body.meta;
    this.manifest = brainData.manifest;
    this.brainConnection = brainConnection;
    this.world = new WorldRenderer(byId("world-stage"), body.mj, body.model, body.data, body.meta);
    this.brain = new BrainRenderer(
      byId("brain-stage"),
      brainData.positions,
      brainData.groups,
      brainData.sides,
      brainData.displayEdges,
      brainData.manifest,
    );
    this.controller = new Controller(body.meta);
    this.physicsStepper = makeStepper(body.meta.timestep, 90);
    this.statsMeter = makeStatsMeter(body.meta.timestep, ({ rtf }) => {
      byId("metric-ratio").textContent = `${rtf.toFixed(2)}x`;
    });
    this.playbackSpeed = 0.08;
    this.driveHz = 55;
    this.running = true;
    this.motor = { forward: 0, reverse: 0, turn: 0, feed: 0, groom: 0, escape: 0 };
    this.sensors = {};
    this.hunger = 0.72;
    this.dust = 0.08;
    this.loomTimer = 0;
    this.simTime = 0;
    this.brainAccumulatorMs = 0;
    this.brainBusy = false;
    this.requestId = 0;
    this.ignoreFramesThrough = 0;
    this.lastWallTime = undefined;
    this.nearestFood = null;
    this.resetBody();
    this.bindControls();
    brainConnection.onFrame((frame) => this.onBrainFrame(frame));
  }

  bindControls() {
    byId("play-button").addEventListener("click", () => {
      this.running = !this.running;
      byId("play-button").textContent = this.running ? "Pause" : "Resume";
    });
    byId("food-button").addEventListener("click", () => this.world.addFoodAhead());
    byId("dust-button").addEventListener("click", () => { this.dust = 1; });
    byId("loom-button").addEventListener("click", () => { this.loomTimer = 0.8; });
    byId("reset-button").addEventListener("click", () => this.reset());
    byId("settings-button").addEventListener("click", () => {
      const panel = byId("settings");
      panel.hidden = !panel.hidden;
      byId("settings-button").setAttribute("aria-expanded", String(!panel.hidden));
    });
    byId("speed-range").addEventListener("input", (event) => {
      this.playbackSpeed = Number(event.target.value);
      byId("speed-value").textContent = `${this.playbackSpeed.toFixed(2)}x`;
    });
    byId("drive-range").addEventListener("input", (event) => {
      this.driveHz = Number(event.target.value);
      byId("drive-value").textContent = `${this.driveHz.toFixed(0)} Hz`;
    });
    document.querySelectorAll("[data-camera]").forEach((button) => {
      button.addEventListener("click", () => {
        this.world.cameraMode = button.dataset.camera;
        document.querySelectorAll("[data-camera]").forEach((peer) => {
          peer.setAttribute("aria-pressed", String(peer === button));
        });
      });
    });
  }

  resetBody() {
    this.mj.mj_resetDataKeyframe(this.model, this.data, 0);
    this.controller.reset();
    this.mj.mj_forward(this.model, this.data);
  }

  reset() {
    this.resetBody();
    this.hunger = 0.72;
    this.dust = 0.08;
    this.loomTimer = 0;
    this.simTime = 0;
    this.brainAccumulatorMs = 0;
    this.ignoreFramesThrough = this.requestId;
    this.brainBusy = true;
    this.motor = { forward: 0, reverse: 0, turn: 0, feed: 0, groom: 0, escape: 0 };
    this.brain.resetActivity();
    this.updateMotorHud();
    byId("metric-brain-time").textContent = "0.00 s";
    this.brainConnection.worker.postMessage({ type: "reset", seed: 0x5eed1234 });
  }

  computeSensors() {
    const pose = this.world.pose();
    let nearest = null;
    for (const food of this.world.foods) {
      if (food.energy <= 0) continue;
      const dx = food.x - pose.x;
      const dy = food.y - pose.y;
      const distance = Math.hypot(dx, dy);
      if (!nearest || distance < nearest.distance) nearest = { food, dx, dy, distance };
    }
    this.nearestFood = nearest;
    let odor = 0;
    let lateral = 0;
    let contact = 0;
    if (nearest) {
      odor = Math.exp(-nearest.distance / 16) * nearest.food.energy;
      const bearing = Math.atan2(nearest.dy, nearest.dx) - pose.yaw;
      lateral = Math.sin(Math.atan2(Math.sin(bearing), Math.cos(bearing)));
      contact = nearest.distance < 2.1 ? 1 : 0;
    }
    const left = clamp(odor * (1 + lateral * 0.72), 0, 1);
    const right = clamp(odor * (1 - lateral * 0.72), 0, 1);
    return {
      odor,
      contact,
      odorLeftHz: left * 145,
      odorRightHz: right * 145,
      sugarLeftHz: contact ? 200 : 0,
      sugarRightHz: contact ? 200 : 0,
      touchHz: this.dust * 150,
      loomHz: this.loomTimer > 0 ? 220 : 0,
      hungerHz: this.driveHz * this.hunger,
    };
  }

  updateInternalState(dt) {
    this.hunger = clamp(this.hunger + dt * 0.0025, 0, 1);
    this.dust = clamp(this.dust + dt * 0.0015, 0, 1);
    this.loomTimer = Math.max(0, this.loomTimer - dt);
    this.world.updateLoom(this.loomTimer);
    if (this.sensors.contact && this.motor.feed > 0.18 && this.nearestFood) {
      this.hunger = clamp(this.hunger - dt * 0.24, 0, 1);
      this.nearestFood.food.energy = Math.max(0, this.nearestFood.food.energy - dt * 0.12);
      this.nearestFood.food.mesh.scale.setScalar(0.35 + this.nearestFood.food.energy * 0.65);
      if (this.nearestFood.food.energy <= 0) this.nearestFood.food.mesh.visible = false;
    }
    if (this.motor.groom > 0.15) this.dust = clamp(this.dust - dt * 0.18, 0, 1);
  }

  physicsStep() {
    let base = this.motor.forward - this.motor.reverse;
    let turn = this.motor.turn;
    if (this.motor.escape > 0.2 || this.loomTimer > 0.15) base = Math.max(base, 1.2);
    if (this.sensors.contact && this.motor.feed > 0.18) base = 0;

    if (this.motor.groom > 0.22 && this.dust > 0.25) {
      this.controller.stepGroom(this.data.ctrl, this.simTime);
    } else {
      const gainLeft = clamp(base + turn, -1.2, 1.2);
      const gainRight = clamp(base - turn, -1.2, 1.2);
      this.controller.stepCPG(this.data.ctrl, gainLeft, gainRight);
    }
    this.mj.mj_step(this.model, this.data);
    this.simTime += this.bodyMeta.timestep;

    const pose = this.world.pose();
    if (Math.abs(pose.x) > 58 || Math.abs(pose.y) > 43 || pose.z < -1) this.resetBody();
  }

  requestBrainStep() {
    if (this.brainBusy || this.brainAccumulatorMs < 15) return;
    const durationMs = Math.min(45, Math.floor(this.brainAccumulatorMs / 15) * 15);
    this.brainAccumulatorMs -= durationMs;
    this.sensors = this.computeSensors();
    this.brainBusy = true;
    this.brainConnection.worker.postMessage({
      type: "step",
      requestId: ++this.requestId,
      durationMs,
      stimuli: this.sensors,
    });
  }

  onBrainFrame(frame) {
    if (frame.type === "error") {
      this.running = false;
      setStatus(frame.error.message, "error");
      return;
    }
    if (frame.type === "reset-complete") {
      this.brainBusy = false;
      return;
    }
    if (frame.requestId <= this.ignoreFramesThrough) return;
    this.brainBusy = false;
    this.motor = frame.motor;
    this.brain.updateActivity(frame.spikeIndices, frame.spikeCounts, frame.groupRates, frame.simulatedMs);
    byId("metric-brain-time").textContent = `${(frame.brainTimeMs / 1000).toFixed(2)} s`;
    byId("metric-compute").textContent = `${frame.computeMs.toFixed(0)} ms`;
    this.updateMotorHud();
  }

  updateMotorHud() {
    const values = {
      forward: this.motor.forward,
      turn: this.motor.turn,
      feed: this.motor.feed,
      groom: this.motor.groom,
    };
    for (const [name, value] of Object.entries(values)) {
      byId(`motor-${name}`).textContent = value.toFixed(2);
      byId(`motor-${name}-meter`).style.width = `${clamp(Math.abs(value), 0, 1) * 100}%`;
    }
  }

  updateWorldHud() {
    const pose = this.world.pose();
    byId("pose-readout").textContent = `x ${pose.x.toFixed(1)} | y ${pose.y.toFixed(1)} | ${Math.round(pose.yaw * 180 / Math.PI)} deg`;
    const odor = clamp(this.sensors.odor || 0, 0, 1);
    const sugar = this.sensors.contact || 0;
    const touch = this.dust;
    for (const [name, value] of Object.entries({ odor, sugar, touch })) {
      byId(`${name}-value`).textContent = `${Math.round(value * 100)}%`;
      byId(`${name}-meter`).style.width = `${value * 100}%`;
    }
    let behavior = "Sensor integration";
    if (this.motor.escape > 0.2 || this.loomTimer > 0.1) behavior = "Escape";
    else if (this.motor.groom > 0.22 && this.dust > 0.25) behavior = "Antennal grooming";
    else if (sugar && this.motor.feed > 0.18) behavior = "Feeding";
    else if (this.motor.forward > 0.08) behavior = "Food seeking";
    byId("behavior-label").textContent = behavior;
  }

  start() {
    byId("loading").hidden = true;
    setStatus("Closed loop running", "ready");
    requestAnimationFrame((time) => this.frame(time));
  }

  frame(nowMs) {
    requestAnimationFrame((time) => this.frame(time));
    const now = nowMs / 1000;
    const wallDt = this.lastWallTime === undefined ? 0 : Math.min(now - this.lastWallTime, 0.1);
    this.lastWallTime = now;
    let physicsSteps = 0;
    if (this.running) {
      physicsSteps = this.physicsStepper.advance(wallDt * this.playbackSpeed, () => this.physicsStep());
      const simulatedSeconds = physicsSteps * this.bodyMeta.timestep;
      this.brainAccumulatorMs += simulatedSeconds * 1000;
      this.updateInternalState(simulatedSeconds);
      this.requestBrainStep();
    }
    this.world.render();
    this.brain.render();
    this.updateWorldHud();
    this.statsMeter(now, physicsSteps);
  }
}

async function main() {
  const brainConnection = startBrainWorker();
  try {
    const bodyPromise = loadScene({
      assetsDir: ASSETS_URL,
      xmlName: "fly.xml",
      onStage: (message) => setStatus(message),
    });
    const [body, brainData] = await Promise.all([bodyPromise, loadBrainVisualData(), brainConnection.ready]);
    byId("metric-neurons").textContent = number.format(brainData.manifest.neuron_count);
    byId("metric-edges").textContent = number.format(brainData.manifest.edge_count);
    const lab = new EmbodiedFlyLab(body, brainData, brainConnection);
    lab.start();
  } catch (error) {
    console.error(error);
    setStatus(error instanceof Error ? error.message : String(error), "error");
    byId("loading-message").textContent = `Initialization failed: ${error instanceof Error ? error.message : error}`;
  }
}

main();
