#!/usr/bin/env python3
"""Validate graph binaries and the generated NeuroMechFly model."""

from __future__ import annotations

import argparse
import array
import hashlib
import json
import math
import sys
import xml.etree.ElementTree as ET
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DTYPE_BYTES = {
    "uint8": 1,
    "int16": 2,
    "uint32": 4,
    "uint64": 8,
    "float32": 4,
}


def digest(path: Path) -> str:
    result = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            result.update(chunk)
    return result.hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--physics",
        action="store_true",
        help="also compile and step the XML with the Python MuJoCo package",
    )
    return parser.parse_args()


def verify_graph() -> dict[str, int]:
    manifest = json.loads((ROOT / "data/manifest.json").read_text())
    for name, descriptor in manifest["arrays"].items():
        path = ROOT / "data" / descriptor["file"]
        expected_size = descriptor["length"] * DTYPE_BYTES[descriptor["dtype"]]
        assert path.stat().st_size == expected_size == descriptor["bytes"], name
        assert digest(path) == descriptor["sha256"], name

    offsets = array.array("I")
    with (ROOT / "data" / manifest["arrays"]["offsets"]["file"]).open("rb") as handle:
        offsets.fromfile(handle, manifest["arrays"]["offsets"]["length"])
    if sys.byteorder != "little":
        offsets.byteswap()
    assert len(offsets) == manifest["neuron_count"] + 1
    assert offsets[0] == 0 and offsets[-1] == manifest["edge_count"]
    assert all(a <= b for a, b in zip(offsets, offsets[1:])), "CSR offsets not sorted"
    assert sum(manifest["group_counts"].values()) == manifest["neuron_count"]
    return manifest


def verify_xml(run_physics: bool) -> dict[str, float | int]:
    xml_path = ROOT / "assets/model/fly.xml"
    root = ET.parse(xml_path).getroot()
    option = root.find("option")
    assert option is not None and float(option.attrib["timestep"]) == 0.0001
    actuator = root.find("actuator")
    assert actuator is not None and len(actuator) == 48
    for mesh in root.findall("./asset/mesh"):
        assert (xml_path.parent / mesh.attrib["file"]).is_file(), mesh.attrib["file"]

    result: dict[str, float | int] = {
        "actuators": len(actuator),
        "timestep": float(option.attrib["timestep"]),
    }
    if run_physics:
        import mujoco

        model = mujoco.MjModel.from_xml_path(str(xml_path))
        data = mujoco.MjData(model)
        mujoco.mj_resetDataKeyframe(model, data, 0)
        for _ in range(100):
            mujoco.mj_step(model, data)
        assert all(math.isfinite(float(value)) for value in data.qpos)
        result.update({"nq": model.nq, "nu": model.nu, "steps": 100})
    return result


def main() -> int:
    args = parse_args()
    graph = verify_graph()
    physics = verify_xml(args.physics)
    print(json.dumps({
        "graph": {
            "neurons": graph["neuron_count"],
            "edges": graph["edge_count"],
            "synapses": graph["synapse_count"],
        },
        "physics": physics,
        "status": "ok",
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
