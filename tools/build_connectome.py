#!/usr/bin/env python3
"""Build compact browser-ready arrays from the public FlyWire v783 model."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


GROUP_NAMES = [
    "unknown",
    "sensory",
    "optic",
    "central",
    "descending",
    "motor",
    "ascending",
    "endocrine",
]

SHIU_SUGAR_IDS = [
    720575940624963786,
    720575940630233916,
    720575940637568838,
    720575940638202345,
    720575940617000768,
    720575940630797113,
    720575940632889389,
    720575940621754367,
    720575940621502051,
    720575940640649691,
    720575940639332736,
    720575940616885538,
    720575940639198653,
    720575940620900446,
    720575940617937543,
    720575940632425919,
    720575940633143833,
    720575940612670570,
    720575940628853239,
    720575940629176663,
    720575940611875570,
]


def parse_args() -> argparse.Namespace:
    project = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser()
    parser.add_argument("--brain-data", type=Path, required=True)
    parser.add_argument("--annotations", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=project / "data")
    parser.add_argument("--display-edges", type=int, default=100_000)
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_array(path: Path, values: np.ndarray, dtype: str) -> dict[str, object]:
    array = np.asarray(values, dtype=np.dtype(dtype).newbyteorder("<"))
    array.tofile(path)
    return {
        "file": path.name,
        "dtype": dtype,
        "length": int(array.size),
        "bytes": int(path.stat().st_size),
        "sha256": sha256(path),
    }


def parse_positions(series: pd.Series) -> np.ndarray:
    result = np.full((len(series), 3), np.nan, dtype=np.float64)
    for row, value in enumerate(series):
        if isinstance(value, str):
            parsed = np.fromstring(value.strip("[]"), sep=" ", dtype=np.float64)
            if parsed.size == 3:
                result[row] = parsed
    return result


def normalized_positions(raw: np.ndarray) -> tuple[np.ndarray, dict[str, list[float]]]:
    valid = np.isfinite(raw).all(axis=1)
    low = np.nanpercentile(raw[valid], 1.0, axis=0)
    high = np.nanpercentile(raw[valid], 99.0, axis=0)
    center = (low + high) / 2
    scale = np.maximum(high - low, 1.0)
    normalized = np.clip((raw - center) / scale, -0.65, 0.65)
    # FlyWire xyz is rearranged for a familiar dorsal brain view in Three.js.
    normalized = normalized[:, [0, 2, 1]]
    normalized[:, 1] *= -1
    normalized[~valid] = 0
    normalized *= np.array([12.0, 8.5, 7.0])
    return normalized.astype(np.float32), {
        "percentile_low_xyz": low.tolist(),
        "percentile_high_xyz": high.tolist(),
        "valid": int(valid.sum()),
    }


def indices_for(meta: pd.DataFrame, mask: pd.Series) -> list[int]:
    return meta.loc[mask.fillna(False), "index"].astype(int).tolist()


def build_populations(meta: pd.DataFrame) -> dict[str, list[int]]:
    primary = meta["primary_type"].fillna("")
    extra = meta["additional_type(s)"].fillna("")
    side = meta["side"].fillna("")
    superclass = meta["super_class"].fillna("")
    cell_class = meta["class"].fillna("")
    subclass = meta["sub_class"].fillna("")

    sugar = (primary == "LB3") & (subclass == "sugar/water")
    food_odor = primary.isin(["ORN_DM1", "ORN_DM2", "ORN_DM4", "ORN_VA2"])
    johnstons = primary.str.startswith("JO-") | (
        (superclass == "sensory") & (cell_class == "mechanosensory")
    )
    looming = primary.str.match(r"^(LC4|LC6|LPLC2)(_|$)")

    named = lambda value: (primary == value) | extra.str.split(", ").apply(
        lambda values: value in values
    )
    root_ids = meta.index.to_series()

    steering = named("DNa01") | named("DNa02")
    populations = {
        "sugar_left": indices_for(meta, sugar & (side == "left")),
        "sugar_right": indices_for(meta, sugar & (side == "right")),
        "sugar_shiu_reference": indices_for(meta, root_ids.isin(SHIU_SUGAR_IDS)),
        "food_odor_left": indices_for(meta, food_odor & (side == "left")),
        "food_odor_right": indices_for(meta, food_odor & (side == "right")),
        "antennal_mechanosensory": indices_for(meta, johnstons),
        "looming_visual_proxy": indices_for(meta, looming),
        "forward_odn1": indices_for(meta, named("DNg97")),
        "walk_dnp09": indices_for(meta, named("DNp09")),
        "turn_dna01": indices_for(meta, named("DNa01")),
        "turn_dna02": indices_for(meta, named("DNa02")),
        "turn_left": indices_for(meta, steering & (side == "left")),
        "turn_right": indices_for(meta, steering & (side == "right")),
        "reverse_mdn": indices_for(meta, named("MDN")),
        "groom_adn1": indices_for(meta, named("DNg62")),
        "escape_giant_fiber": indices_for(meta, named("DNp01")),
        "feed_mn9": indices_for(
            meta,
            named("CB0701")
            | root_ids.isin([720575940660219265, 720575940645521262]),
        ),
    }
    return populations


def main() -> int:
    args = parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    completeness_path = args.brain_data / "2025_Completeness_783.csv"
    connectivity_path = args.brain_data / "2025_Connectivity_783.parquet"
    classification_path = args.annotations / "classification.csv.gz"
    coordinates_path = args.annotations / "coordinates.csv.gz"
    cell_types_path = args.annotations / "consolidated_cell_types.csv.gz"

    completeness = pd.read_csv(completeness_path, index_col=0)
    root_ids = completeness.index.to_numpy(dtype=np.uint64)
    neuron_count = len(root_ids)
    id_to_index = pd.Series(np.arange(neuron_count, dtype=np.int64), index=root_ids)

    classification = pd.read_csv(classification_path).set_index("root_id")
    cell_types = pd.read_csv(cell_types_path).set_index("root_id")
    coordinates = (
        pd.read_csv(coordinates_path)
        .drop_duplicates("root_id", keep="first")
        .set_index("root_id")
    )
    meta = pd.DataFrame(index=root_ids)
    meta = meta.join(classification).join(cell_types).join(coordinates)
    meta["index"] = id_to_index

    raw_positions = parse_positions(meta["position"])
    positions, position_info = normalized_positions(raw_positions)

    superclass = meta["super_class"].fillna("unknown")
    group_lookup = {name: idx for idx, name in enumerate(GROUP_NAMES)}
    groups = superclass.map(group_lookup).fillna(0).to_numpy(dtype=np.uint8)
    side_lookup = {"left": 1, "right": 2, "center": 3}
    sides = meta["side"].map(side_lookup).fillna(0).to_numpy(dtype=np.uint8)
    populations = build_populations(meta)

    edges = pd.read_parquet(
        connectivity_path,
        columns=[
            "Presynaptic_Index",
            "Postsynaptic_Index",
            "Excitatory x Connectivity",
        ],
    )
    source = edges["Presynaptic_Index"].to_numpy(dtype=np.uint32, copy=False)
    target = edges["Postsynaptic_Index"].to_numpy(dtype=np.uint32, copy=False)
    signed_weight = edges["Excitatory x Connectivity"].to_numpy(dtype=np.int64, copy=False)
    if signed_weight.min() < np.iinfo(np.int16).min or signed_weight.max() > np.iinfo(np.int16).max:
        raise ValueError("signed synapse counts do not fit int16")
    weight = signed_weight.astype(np.int16)

    if np.any(source[1:] < source[:-1]):
        order = np.argsort(source, kind="stable")
        source, target, weight = source[order], target[order], weight[order]

    offsets = np.zeros(neuron_count + 1, dtype=np.uint32)
    np.cumsum(np.bincount(source, minlength=neuron_count), out=offsets[1:])

    n_display = min(args.display_edges, len(weight))
    strongest = np.argpartition(np.abs(weight.astype(np.int32)), -n_display)[-n_display:]
    strongest = strongest[np.argsort(np.abs(weight[strongest].astype(np.int32)))[::-1]]
    display_edges = np.column_stack((source[strongest], target[strongest])).astype(np.uint32)

    arrays = {
        "offsets": write_array(args.out / "offsets.u32", offsets, "uint32"),
        "targets": write_array(args.out / "targets.u32", target, "uint32"),
        "weights": write_array(args.out / "weights.i16", weight, "int16"),
        "root_ids": write_array(args.out / "root-ids.u64", root_ids, "uint64"),
        "positions": write_array(args.out / "positions.f32", positions, "float32"),
        "groups": write_array(args.out / "groups.u8", groups, "uint8"),
        "sides": write_array(args.out / "sides.u8", sides, "uint8"),
        "display_edges": write_array(
            args.out / "display-edges.u32", display_edges, "uint32"
        ),
    }

    manifest = {
        "schema_version": 1,
        "dataset": "FlyWire FAFB v783",
        "neuron_count": neuron_count,
        "edge_count": int(len(weight)),
        "synapse_count": int(np.abs(weight.astype(np.int64)).sum()),
        "display_edge_count": n_display,
        "group_names": GROUP_NAMES,
        "group_counts": {
            name: int((groups == idx).sum()) for idx, name in enumerate(GROUP_NAMES)
        },
        "position_normalization": position_info,
        "populations": populations,
        "population_counts": {key: len(value) for key, value in populations.items()},
        "lif": {
            "dt_ms": 0.1,
            "v_rest_mv": -52.0,
            "v_reset_mv": -52.0,
            "v_threshold_mv": -45.0,
            "tau_mem_ms": 20.0,
            "tau_syn_ms": 5.0,
            "refractory_ms": 2.2,
            "delay_ms": 1.8,
            "weight_per_synapse_mv": 0.275,
            "poisson_weight_scale": 250.0,
        },
        "arrays": arrays,
        "sources": {
            "lif_model": "https://github.com/philshiu/Drosophila_brain_model",
            "brain_backend": "https://github.com/eonsystemspbc/fly-brain",
            "codex": "https://codex.flywire.ai/api/download?dataset=fafb",
            "connectivity_file_sha256": sha256(connectivity_path),
            "completeness_file_sha256": sha256(completeness_path),
        },
        "model_boundary": {
            "simulation": "All listed neurons and weighted edges are stepped.",
            "display": "All neurons and the strongest anatomical edge subset are rendered.",
            "motor_interface": "Published descending-neuron groups are converted to low-dimensional FlyGym controller gains.",
            "vision": "Looming cell types are used as a proxy input; the Lappalainen visual model is not included.",
        },
    }
    manifest_path = args.out / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    print(f"neurons={neuron_count:,}")
    print(f"edges={len(weight):,}")
    print(f"synapses={manifest['synapse_count']:,}")
    print(f"display_edges={n_display:,}")
    for name, values in populations.items():
        print(f"population.{name}={len(values):,}")
    print(f"output={args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
